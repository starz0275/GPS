"""
train_e2e.py — CovAdapterNet 端到端训练（论文 Section V-A）
==========================================================

训练逻辑
--------
1. 加载标定数据 → 归一化 IMU
2. CovAdapterNet 从 IMU 滑窗预测 [z_lat, z_up]
3. 穿过可微 EKF 得到状态轨迹
4. Loss = GPS 有效帧之间的相对平移误差（论文 t_rel）
5. 梯度反向传播穿过 EKF → CovAdapterNet 参数更新
6. 导出权重为 Keras .weights.h5 → 供 NumPy EKF 推理
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter

# ---- 复用现有的数据加载 ----
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_ekf import load_calibration_seq
from data_preprocessing_v2 import DATA_DIR_CALIB, CALIB_TRAIN_IDS, CALIB_VAL_ID
from config import DEFAULT_EKF_CONFIG

from ekf_torch import (
    DifferentiableEKF, CovAdapterNetTorch, export_torch_to_keras,
    N_STATE, IDX_PX, IDX_PY,
)

# ============================================================================
# 配置
# ============================================================================
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
TARGET_DT = 0.1
WINDOW_SIZE = 30

LR = 3e-4
EPOCHS = 20
GRAD_CLIP = 1.0
CHUNK_SIZE = 80

OUTPUT_DIR = Path(__file__).parent / 'trained_models'
OUTPUT_DIR.mkdir(exist_ok=True)
COV_WEIGHTS_PT = OUTPUT_DIR / 'cov_adapter_torch.pt'
COV_WEIGHTS_H5 = OUTPUT_DIR / 'cov_adapter_weights.weights.h5'
NORM_JSON = Path(__file__).parent / 'preprocessed_data' / 'normalization_stats.json'
BIASNET_WEIGHTS = Path(__file__).parent / 'trained_models' / 'biasnet_weights.weights.h5'


def load_norm_stats():
    with open(NORM_JSON) as f:
        raw = json.load(f)
    stats = raw.get('stats', raw)
    keys = ['AccX_g', 'AccY_g', 'AccZ_g', 'GyroX_degs', 'GyroY_degs', 'GyroZ_degs']
    mu = np.array([stats[k]['mean'] for k in keys], np.float32)
    std = np.array([stats[k]['std'] for k in keys], np.float32) + 1e-8
    return mu, std


def normalize_imu(imu, mu, std):
    return (imu - mu) / std


def compute_biasnet_predictions(seqs, mu, std):
    """用训练好的 BiasNet 对全部序列做零偏预测。"""
    from ekf_navigator import BiasNet, clip_biasnet_output
    W = 30
    model = BiasNet(window_size=W)
    dummy = np.zeros((1, W, 6), dtype=np.float32)
    model(dummy)
    model.load_weights(str(BIASNET_WEIGHTS))
    DEG2RAD_LOCAL = np.pi / 180.0

    biases = []
    for seq in seqs:
        imu_norm = normalize_imu(seq['imu'], mu, std)
        T = len(imu_norm)
        bias = np.zeros(T, dtype=np.float32)
        if T >= W:
            windows = np.stack([imu_norm[i:i+W] for i in range(T - W + 1)], axis=0)
            raw = model(windows.astype(np.float32), training=False).numpy().flatten()
            raw_clipped = clip_biasnet_output(raw, 2.0)
            bias[W - 1:] = raw_clipped * DEG2RAD_LOCAL
        biases.append(bias.astype(np.float32))
    return biases


def seq_to_tensors(seq, mu, std, device, net_bias=None, sim_outage=True):
    """将一段轨迹转为 PyTorch tensor，可选模拟 GPS outage。"""
    T = len(seq['Time_s'])
    imu_norm = normalize_imu(seq['imu'], mu, std)

    # 时间步长
    dt_raw = np.diff(seq['Time_s'], prepend=seq['Time_s'][0])
    dt_raw[0] = TARGET_DT
    dt_raw = np.clip(dt_raw, DEFAULT_EKF_CONFIG.dt_min, DEFAULT_EKF_CONFIG.dt_max)

    # 初始状态（复用 _init_state 的逻辑）
    gps_valid_truth = seq['gps_valid']
    v_ms = seq['v_ms']
    gps_x = seq['enu_x']
    gps_y = seq['enu_y']

    ok = np.where(gps_valid_truth & np.isfinite(gps_x) & np.isfinite(gps_y))[0]
    if len(ok) == 0:
        return None
    k0 = int(ok[0])
    px0 = float(gps_x[k0])
    py0 = float(gps_y[k0])

    # 找首段运动计算航向
    min_hdg = 5.0 / 3.6
    moving = np.where((v_ms >= min_hdg) & np.isfinite(gps_x) & np.isfinite(gps_y))[0]
    yaw0 = 0.0
    if len(moving) >= 5:
        i0 = int(moving[0])
        i1 = min(i0 + 10, len(gps_x) - 1)
        dx = float(gps_x[i1] - gps_x[i0])
        dy = float(gps_y[i1] - gps_y[i0])
        if dx * dx + dy * dy > 4.0:
            yaw0 = float(np.arctan2(dy, dx))

    # 静止零偏估计
    still = (np.abs(v_ms) < 0.1) & np.isfinite(seq['gyro_z_rad']) & gps_valid_truth
    still_idx = np.where(still)[0]
    bg0 = 0.0
    if len(still_idx) >= 10:
        bg0 = float(np.median(seq['gyro_z_rad'][still_idx]))
        bg0 = np.clip(bg0, -DEFAULT_EKF_CONFIG.bg_init_max_bg_rads,
                      DEFAULT_EKF_CONFIG.bg_init_max_bg_rads)

    x0 = torch.tensor([px0, py0, 0.0, 0.0, yaw0, bg0], device=device)

    # EKF 起始帧：找第一帧运动
    motion_frames = np.where((v_ms >= DEFAULT_EKF_CONFIG.min_speed_wheel_ms)
                             & np.isfinite(gps_x) & np.isfinite(gps_y))[0]
    if len(motion_frames) > 0:
        k_start = int(motion_frames[0])
    else:
        k_start = k0

    # BiasNet 输出（如果提供则使用，否则用 0）
    if net_bias is not None:
        net_bias_arr = net_bias.astype(np.float32)
    else:
        net_bias_arr = np.zeros(T, dtype=np.float32)

    # ★ GPS outage 模拟 ★
    gps_valid_sim = gps_valid_truth.copy()
    outage_mask = np.zeros(T, dtype=bool)
    if sim_outage:
        # 找运动后 10s 开始，持续 60-90s（根据数据长度自适应）
        outage_start = min(k_start + 100, T - 200) if T > 300 else k_start + 50
        outage_dur = min(600, T - outage_start - 50)  # 最多 60s
        if outage_dur > 50:
            outage_end = outage_start + outage_dur
            gps_valid_sim[outage_start:outage_end] = False
            outage_mask[outage_start:outage_end] = True

    return {
        'imu_norm': torch.from_numpy(imu_norm).float().to(device),
        'gyro_z': torch.from_numpy(seq['gyro_z_rad'].astype(np.float32)).to(device),
        'net_bias': torch.from_numpy(net_bias_arr).to(device),
        'v_ms': torch.from_numpy(seq['v_ms'].astype(np.float32)).to(device),
        'gps_valid': gps_valid_sim,                        # ← EKF 用模拟的
        'gps_valid_truth': gps_valid_truth,                 # ← Loss 用原始的
        'outage_mask': outage_mask,                         # ← 标记 outage 帧
        'gps_x': gps_x.astype(np.float32),
        'gps_y': gps_y.astype(np.float32),
        'dt_arr': torch.from_numpy(dt_raw.astype(np.float32)).to(device),
        'x0': x0,
        'k_start': k_start,
        'T': T,
    }


def compute_loss(states, gps_x, gps_y, gps_valid_truth, outage_mask):
    """
    论文损失：仅计算 outage 段 GPS 有效帧之间的相对平移误差。

    Loss = mean( (||Δpos_pred|| - ||Δpos_true||)² )  over outage GPS frames
    """
    gps_x_t = torch.from_numpy(gps_x).float().to(states.device)
    gps_y_t = torch.from_numpy(gps_y).float().to(states.device)
    gps_v_t = torch.from_numpy(gps_valid_truth).to(states.device)
    outage_t = torch.from_numpy(outage_mask).to(states.device)

    pred_pos = states[:, :2]
    true_pos = torch.stack([gps_x_t, gps_y_t], dim=-1)

    # ★ GPS 真值有效 AND 在 outage 段内
    target_frames = torch.where(gps_v_t & outage_t)[0]
    if len(target_frames) < 2:
        # outage 段不够 → 回退到全轨迹
        target_frames = torch.where(gps_v_t)[0]
        if len(target_frames) < 2:
            return torch.tensor(0.0, device=states.device, requires_grad=True)

    # 相邻 GPS 有效帧之间的相对位移
    rel_pred = pred_pos[target_frames[1:]] - pred_pos[target_frames[:-1]]
    rel_true = true_pos[target_frames[1:]] - true_pos[target_frames[:-1]]

    d_pred = torch.norm(rel_pred, dim=-1)
    d_true = torch.norm(rel_true, dim=-1)

    # Huber 混合
    err = torch.abs(d_pred - d_true)
    huber = torch.where(err < 5.0, 0.5 * err ** 2, 5.0 * (err - 2.5))
    return huber.mean()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print("=" * 60)
    print("CovAdapterNet 端到端训练（穿过可微 EKF）")
    print("=" * 60)

    # 1. 加载归一化参数
    mu, std = load_norm_stats()

    # 2. 加载训练数据
    train_ids = list(CALIB_TRAIN_IDS)
    print(f"\n[1] 加载训练序列 ({', '.join(train_ids)}) ...")
    seqs_tr = load_calibration_seq(DATA_DIR_CALIB, train_ids)
    print(f"  训练序列: {len(seqs_tr)} 段")

    # ★ 用 BiasNet 为所有序列计算零偏预测
    print("\n  [BiasNet] 计算零偏预测...")
    net_biases = compute_biasnet_predictions(seqs_tr, mu, std)
    print(f"  [BiasNet] 完成 ({len(net_biases)} 段)")

    print(f"\n[2] 构建模型 ...")
    cov_net = CovAdapterNetTorch(window_size=WINDOW_SIZE, in_channels=6).to(device)
    ekf = DifferentiableEKF(device=device).to(device)
    optimizer = torch.optim.Adam(cov_net.parameters(), lr=LR)

    # 从 config 构建 cfg dict
    c = DEFAULT_EKF_CONFIG
    cfg_dict = {
        'q_pos': c.q_pos, 'q_vel': c.q_vel, 'q_yaw': c.q_yaw, 'q_bg': c.q_bg,
        'r_gps_xy': c.r_gps_xy, 'r_nhc': c.r_nhc, 'r_wheel': c.r_wheel,
        'p_init_pos': c.p_init_pos, 'p_init_vel': c.p_init_vel,
        'p_init_yaw': c.p_init_yaw, 'p_init_bg': c.p_init_bg,
        'min_speed_wheel_ms': c.min_speed_wheel_ms,
        'min_speed_nhc_ms': c.min_speed_nhc_ms,
        'freeze_yaw_below_ms': c.freeze_yaw_below_ms,
        'beta_noise_scale': c.beta_noise_scale,
    }

    print(f"\n[3] 开始端到端训练 ({EPOCHS} epochs) ...")
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        epoch_losses = []
        for seq_idx, seq in enumerate(seqs_tr):
            data = seq_to_tensors(seq, mu, std, device, net_bias=net_biases[seq_idx])
            if data is None:
                continue

            T = data['T']
            k0 = data['k_start']
            imu_norm = data['imu_norm']  # (T, 6)

            end_frame = T

            # ----- CovAdapterNet: 滑窗预测 z_lat, z_up -----
            if end_frame < WINDOW_SIZE + 1:
                continue

            windows = torch.stack([imu_norm[i:i+WINDOW_SIZE] for i in range(end_frame - WINDOW_SIZE + 1)])
            z_all = cov_net(windows)  # (N, 2)
            # 填充前 WINDOW_SIZE-1 帧为零（无历史窗口）
            z_full = torch.zeros(end_frame, 2, device=device)
            z_full[WINDOW_SIZE - 1:] = z_all
            z_lat = z_full[:, 0]
            z_up = z_full[:, 1]

            # ----- 穿过可微 EKF（用模拟的 gps_valid）-----
            P0 = ekf._build_P0(cfg_dict)
            states_full = ekf(
                data['x0'].clone(), P0.clone(),
                data['gyro_z'][k0:end_frame], data['net_bias'][k0:end_frame],
                data['v_ms'][k0:end_frame],
                data['gps_valid'][k0:end_frame],   # ← 模拟 outage 的掩码
                data['gps_x'][k0:end_frame], data['gps_y'][k0:end_frame],
                z_lat[k0:end_frame], z_up[k0:end_frame],
                data['dt_arr'][k0:end_frame],
                cfg_dict, chunk_size=CHUNK_SIZE,
            )

            full_states = torch.zeros(end_frame, N_STATE, device=device)
            full_states[k0:] = states_full
            full_states[:k0] = data['x0'].unsqueeze(0)

            # ----- Loss：只看 outage 段 GPS 真值的相对平移误差 -----
            loss = compute_loss(
                full_states,
                data['gps_x'][:end_frame], data['gps_y'][:end_frame],
                data['gps_valid_truth'][:end_frame],     # ← 用原始 GPS 真值
                data['outage_mask'][:end_frame],          # ← 只看 outage 段
            )

            if torch.isnan(loss) or loss.item() == 0.0:
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cov_net.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_losses.append(loss.item())

        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            print(f"  Epoch {epoch+1:2d}/{EPOCHS}  "
                  f"loss={avg_loss:.4f}  ({len(epoch_losses)} seqs)")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(cov_net.state_dict(), COV_WEIGHTS_PT)
                print(f"    ★ 最优模型已保存 (loss={best_loss:.4f})")
        else:
            print(f"  Epoch {epoch+1:2d}/{EPOCHS}  (no valid sequences)")

    # 4. 导出权重
    print(f"\n[4] 导出权重到 Keras .weights.h5 ...")
    best_cov = CovAdapterNetTorch(window_size=WINDOW_SIZE).to(device)
    if COV_WEIGHTS_PT.exists():
        best_cov.load_state_dict(torch.load(COV_WEIGHTS_PT, map_location=device))
    export_torch_to_keras(best_cov, COV_WEIGHTS_H5, WINDOW_SIZE)

    # 5. 保存训练日志
    from datetime import datetime
    log = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'config': {
            'lr': LR, 'epochs': EPOCHS, 'grad_clip': GRAD_CLIP,
            'chunk_size': CHUNK_SIZE,
        },
        'ekf_config': {k: float(v) for k, v in cfg_dict.items()},
        'training': {
            'datasets': train_ids,
            'sequences': len(seqs_tr),
            'best_loss': float(best_loss),
            'final_epoch': epoch + 1,
        },
    }
    log_dir = Path(__file__).parent / 'training_logs'
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"train_cov_{log['timestamp']}.json"
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\n[日志] 已保存到 {log_path}")
    print(f"\n完成！运行 validate_ekf.py 查看效果")

    # 重新启用 CovAdapterNet
    print("\n提示：运行 validate_ekf.py 前确认已恢复 CovAdapterNet 加载")
    print("  检查 validate_ekf.py 中 cov_weights_path 不为 None")


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    np.random.seed(42)
    torch.manual_seed(42)
    main()
