"""
train_biasnet_e2e.py — BiasNet 端到端微调（两阶段训练）
======================================================

阶段一（已完成）: 监督学习 BiasNet 预测零偏 MAE=0.11°/s
阶段二（本脚本）: 加载阶段一权重，穿过可微 EKF，用 outage 定位误差做 loss 微调

每次迭代:
  60s 有 GPS → 丢 10s GPS → EKF 纯推算 → GPS 恢复 → 算位置误差 → 反向传播
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_ekf import load_calibration_seq, load_or_compute_norm, normalize_imu
from data_preprocessing_v2 import DATA_DIR_CALIB, CALIB_TRAIN_IDS
from config import DEFAULT_EKF_CONFIG

from ekf_torch import (
    DifferentiableEKF, BiasNetTorch, export_biasnet_torch_to_keras,
    N_STATE, IDX_PX, IDX_PY, IDX_YAW,
)

# ============================================================================
# 配置
# ============================================================================
DEG2RAD = np.pi / 180.0
TARGET_DT = 0.1
WINDOW_SIZE = 30

LR = 1e-5              # 微调用小学习率
EPOCHS = 10
GRAD_CLIP = 1.0
CHUNK_SIZE = 80        # 协方差梯度截断步长

GPS_INIT_S = 60        # 有 GPS 初始化时长（秒）
OUTAGE_S = 10           # 模拟丢 GPS 时长（秒）
RECOVERY_S = 5          # GPS 恢复后评估时长（秒）

OUTPUT_DIR = Path(__file__).parent / 'trained_models'
OUTPUT_DIR.mkdir(exist_ok=True)
BIASNET_WEIGHTS = OUTPUT_DIR / 'biasnet_weights.weights.h5'
BIASNET_FT_WEIGHTS = OUTPUT_DIR / 'biasnet_finetuned.weights.h5'
NORM_JSON = Path(__file__).parent / 'preprocessed_data' / 'normalization_stats.json'


def load_biasnet_weights_to_torch(torch_model, weights_path, in_channels=7):
    """加载 Keras 训练的 BiasNet 权重到 PyTorch 模型。"""
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    class KerasBiasNet(Model):
        def __init__(self):
            super().__init__(name='BiasNet')
            self.conv1 = layers.Conv1D(32, 5, padding='causal', activation='relu')
            self.conv2 = layers.Conv1D(32, 5, dilation_rate=4, padding='causal', activation='relu')
            self.conv3 = layers.Conv1D(16, 3, padding='causal', activation='relu')
            self.pool = layers.GlobalAveragePooling1D()
            self.fc1 = layers.Dense(32, activation='relu')
            self.drop = layers.Dropout(0.2)
            self.out = layers.Dense(1, activation='linear')

        def call(self, x, training=False):
            h = self.conv1(x)
            h = self.conv2(h)
            h = self.conv3(h)
            h = self.pool(h)
            h = self.fc1(h)
            h = self.drop(h, training=training)
            return self.out(h)

    keras_model = KerasBiasNet()
    dummy = np.zeros((1, WINDOW_SIZE, in_channels), dtype=np.float32)
    keras_model(dummy)
    keras_model.load_weights(str(weights_path))

    # Keras → PyTorch 权重映射
    with torch.no_grad():
        torch_model.conv1.weight.copy_(torch.from_numpy(
            keras_model.conv1.get_weights()[0].transpose(2, 1, 0)))
        torch_model.conv1.bias.copy_(torch.from_numpy(
            keras_model.conv1.get_weights()[1]))
        torch_model.conv2.weight.copy_(torch.from_numpy(
            keras_model.conv2.get_weights()[0].transpose(2, 1, 0)))
        torch_model.conv2.bias.copy_(torch.from_numpy(
            keras_model.conv2.get_weights()[1]))
        torch_model.conv3.weight.copy_(torch.from_numpy(
            keras_model.conv3.get_weights()[0].transpose(2, 1, 0)))
        torch_model.conv3.bias.copy_(torch.from_numpy(
            keras_model.conv3.get_weights()[1]))
        torch_model.fc1.weight.copy_(torch.from_numpy(
            keras_model.fc1.get_weights()[0].T))
        torch_model.fc1.bias.copy_(torch.from_numpy(
            keras_model.fc1.get_weights()[1]))
        torch_model.out.weight.copy_(torch.from_numpy(
            keras_model.out.get_weights()[0].T))
        torch_model.out.bias.copy_(torch.from_numpy(
            keras_model.out.get_weights()[1]))

    print(f"[权重] 已加载 Keras 权重: {weights_path}")
    return torch_model


def seq_to_tensors(seq, mu, std, device):
    """转为 PyTorch tensor。"""
    T = len(seq['Time_s'])
    imu_norm = normalize_imu(seq['imu'], mu, std)

    dt_raw = np.diff(seq['Time_s'], prepend=seq['Time_s'][0])
    dt_raw[0] = TARGET_DT
    dt_raw = np.clip(dt_raw, DEFAULT_EKF_CONFIG.dt_min, DEFAULT_EKF_CONFIG.dt_max)

    gps_valid = seq['gps_valid'].copy()
    gps_x = seq['enu_x']
    gps_y = seq['enu_y']

    ok = np.where(gps_valid & np.isfinite(gps_x) & np.isfinite(gps_y))[0]
    k0 = int(ok[0]) if len(ok) > 0 else 0
    px0 = float(gps_x[k0]) if k0 < len(gps_x) else 0.0
    py0 = float(gps_y[k0]) if k0 < len(gps_y) else 0.0

    # 初始航向
    min_hdg = 5.0 / 3.6
    moving = np.where((seq['v_ms'] >= min_hdg) & np.isfinite(gps_x) & np.isfinite(gps_y))[0]
    yaw0 = 0.0
    if len(moving) > 1:
        i0 = int(moving[min(5, len(moving)-1)])
        i1 = min(i0 + 10, len(gps_x) - 1)
        dx = float(gps_x[i1] - gps_x[i0])
        dy = float(gps_y[i1] - gps_y[i0])
        if dx * dx + dy * dy > 1.0:
            yaw0 = float(np.arctan2(dy, dx))

    # 静止零偏
    still = (np.abs(seq['v_ms']) < 0.1) & np.isfinite(seq['gyro_z_rad']) & gps_valid
    still_idx = np.where(still)[0]
    bg0 = 0.0
    if len(still_idx) >= 10:
        bg0 = float(np.median(seq['gyro_z_rad'][still_idx]))

    x0 = torch.tensor([px0, py0, 0.0, 0.0, yaw0, bg0], device=device)

    k_start = int(moving[0]) if len(moving) > 0 else k0

    return {
        'imu_norm': torch.from_numpy(imu_norm).float().to(device),
        'gyro_z': torch.from_numpy(seq['gyro_z_rad'].astype(np.float32)).to(device),
        'v_ms': torch.from_numpy(seq['v_ms'].astype(np.float32)).to(device),
        'gps_valid': gps_valid,
        'gps_x': gps_x.astype(np.float32),
        'gps_y': gps_y.astype(np.float32),
        'dt_arr': torch.from_numpy(dt_raw.astype(np.float32)).to(device),
        'x0': x0,
        'k_start': k_start,
        'T': T,
    }


def simulate_outage(data, k_start, init_s=GPS_INIT_S, outage_s=OUTAGE_S):
    """在 60s 初始化后模拟 10s GPS 丢失。"""
    init_frames = int(init_s / TARGET_DT)
    outage_frames = int(outage_s / TARGET_DT)

    start_idx = k_start + init_frames
    end_idx = start_idx + outage_frames

    if end_idx >= data['T'] - 10:
        return None  # 数据不够长

    gps_sim = np.ones(data['T'], dtype=bool)
    gps_sim[start_idx:end_idx] = False

    return gps_sim, start_idx, end_idx


def compute_outage_loss(states, gps_x, gps_y, start_idx, end_idx):
    """
    GPS 恢复后，比较 EKF 位置与 GPS 真值的偏差。
    Loss = GPS恢复后5秒内的位置RMSE
    """
    gps_x_t = torch.from_numpy(gps_x).float().to(states.device)
    gps_y_t = torch.from_numpy(gps_y).float().to(states.device)

    eval_end = min(end_idx + 50, len(gps_x))  # 恢复后5秒
    if eval_end - end_idx < 5:
        return None

    pred = states[end_idx:eval_end, :2]      # (N, 2)  EKF位置
    truth = torch.stack([gps_x_t[end_idx:eval_end],
                         gps_y_t[end_idx:eval_end]], dim=-1)

    err = torch.norm(pred - truth, dim=-1)    # (N,)
    return err.mean()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print("=" * 60)
    print("BiasNet 端到端微调（两阶段训练）")
    print(f"  模式: {GPS_INIT_S}s有GPS → {OUTAGE_S}s丢失 → {RECOVERY_S}s恢复评估")
    print("=" * 60)

    # 1. 加载归一化参数
    from ekf_navigator import load_norm_stats
    norm_stats = load_norm_stats(str(NORM_JSON))
    keys = ['AccX_g','AccY_g','AccZ_g','GyroX_degs','GyroY_degs','GyroZ_degs','VehicleSpeed_ms']
    mu = np.array([norm_stats[k]['mean'] for k in keys], np.float32)
    std = np.array([norm_stats[k]['std'] for k in keys], np.float32) + 1e-8

    # 2. 加载训练数据
    print(f"\n[1] 加载训练序列 ...")
    seqs_tr = load_calibration_seq(DATA_DIR_CALIB, CALIB_TRAIN_IDS)
    print(f"  训练序列: {len(seqs_tr)} 段")

    # 3. 构建 BiasNet + 可微 EKF，加载已有权重
    print(f"\n[2] 构建模型，加载已有权重 ...")
    biasnet = BiasNetTorch(window_size=WINDOW_SIZE, in_channels=7).to(device)
    biasnet = load_biasnet_weights_to_torch(biasnet, BIASNET_WEIGHTS)
    biasnet.train()

    ekf = DifferentiableEKF(device=device).to(device)
    optimizer = torch.optim.Adam(biasnet.parameters(), lr=LR)

    # EKF 配置 dict
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

    # 4. 端到端微调
    print(f"\n[3] 开始端到端微调 ({EPOCHS} epochs) ...")
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        epoch_losses = []
        for seq_idx, seq in enumerate(seqs_tr):
            data = seq_to_tensors(seq, mu, std, device)
            k0 = data['k_start']
            T = data['T']

            # 模拟 outage
            outage = simulate_outage(data, k0)
            if outage is None:
                continue
            gps_sim, out_start, out_end = outage

            # 只用 k0 到 out_end+50 这段
            end_frame = min(out_end + 50, T)
            if end_frame > T or end_frame < WINDOW_SIZE + 1:
                continue

            imu_norm = data['imu_norm']

            # BiasNet 预测零偏（滑窗）
            windows = torch.stack([imu_norm[i:i+WINDOW_SIZE]
                                  for i in range(end_frame - WINDOW_SIZE + 1)])
            bias_pred = biasnet(windows)  # (N,)
            net_bias = torch.zeros(end_frame, device=device)
            net_bias[WINDOW_SIZE - 1:] = bias_pred

            # 穿过可微 EKF（模拟 outage）
            P0 = ekf._build_P0(cfg_dict)
            states_full = ekf(
                data['x0'].clone(), P0.clone(),
                data['gyro_z'][k0:end_frame], net_bias[k0:end_frame],
                data['v_ms'][k0:end_frame],
                torch.from_numpy(gps_sim[k0:end_frame]).to(device),
                torch.from_numpy(data['gps_x'][k0:end_frame]).to(device),
                torch.from_numpy(data['gps_y'][k0:end_frame]).to(device),
                torch.zeros(end_frame - k0, device=device),  # z_lat=0
                torch.zeros(end_frame - k0, device=device),  # z_up=0
                data['dt_arr'][k0:end_frame],
                cfg_dict, chunk_size=CHUNK_SIZE,
            )

            # 补齐完整状态
            full_states = torch.zeros(end_frame, N_STATE, device=device)
            full_states[k0:] = states_full
            full_states[:k0] = data['x0'].unsqueeze(0)

            # Loss = GPS 恢复后的位置误差
            loss = compute_outage_loss(
                full_states, data['gps_x'][:end_frame], data['gps_y'][:end_frame],
                out_start, out_end)

            if loss is None or torch.isnan(loss) or loss.item() == 0.0:
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(biasnet.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_losses.append(loss.item())

        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            print(f"  Epoch {epoch+1:2d}/{EPOCHS}  "
                  f"loss={avg_loss:.4f}  ({len(epoch_losses)} seqs)")
            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f"    ★ 最优 (loss={best_loss:.4f})")
        else:
            print(f"  Epoch {epoch+1:2d}/{EPOCHS}  (no valid)")

    # 5. 导出微调后的权重
    print(f"\n[4] 导回 Keras 权重 ...")
    biasnet.eval()
    export_biasnet_torch_to_keras(biasnet, BIASNET_FT_WEIGHTS, WINDOW_SIZE, 7)

    # 同时覆盖原始权重路径（使 validate_ekf.py 默认加载微调版）
    export_biasnet_torch_to_keras(biasnet, BIASNET_WEIGHTS, WINDOW_SIZE, 7)

    # 6. 日志
    from datetime import datetime
    log = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'config': {
            'lr': LR, 'epochs': EPOCHS, 'grad_clip': GRAD_CLIP,
            'gps_init_s': GPS_INIT_S, 'outage_s': OUTAGE_S,
            'recovery_s': RECOVERY_S,
        },
        'training': {'datasets': CALIB_TRAIN_IDS, 'best_loss': float(best_loss)},
    }
    log_dir = Path(__file__).parent / 'training_logs'
    log_dir.mkdir(exist_ok=True)
    with open(log_dir / f"train_biasnet_e2e_{log['timestamp']}.json", 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\n完成！微调权重已保存到 {BIASNET_FT_WEIGHTS}")
    print("运行 validate_ekf.py 查看效果")


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    np.random.seed(42)
    torch.manual_seed(42)
    main()
