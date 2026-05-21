"""
train_cov_adapter.py — CovAdapterNet 训练脚本
=============================================

训练逻辑
--------
1. 先跑一遍固定噪声的 EKF，记录每帧的 NHC/轮速残差
2. 将残差映射为目标 z 值（论文公式 (17) 的逆映射）
3. 用 IMU 窗口 → z_lat, z_up 监督训练 CovAdapterNet
4. （可选）迭代：用训练好的 CovAdapterNet 重跑 EKF，更新标签

损失函数
--------
论文方法（方案 A）：Loss = 相对平移误差 t_rel，反向传播穿过整个 EKF
   → 需要 PyTorch 可微 EKF，详见本文档末尾的方案 A 说明

本脚本方法（方案 B）：启发式残差标签
   → 立即可用，不需要可微 EKF
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (ModelCheckpoint, ReduceLROnPlateau,
                                        EarlyStopping)
from pathlib import Path
from scipy.ndimage import median_filter
import json, warnings
warnings.filterwarnings('ignore')

from ekf_navigator import (
    CovAdapterNet, EKFNavigatorNP, load_norm_stats, enu_to_body, wrap_angle,
)
from config import DEFAULT_EKF_CONFIG

# ============================================================================
# 配置
# ============================================================================

MODEL_DIR       = Path(__file__).parent / "trained_models"
MODEL_DIR.mkdir(exist_ok=True)
NORM_JSON       = Path(__file__).parent / "preprocessed_data" / "normalization_stats.json"
WEIGHTS_PATH    = MODEL_DIR / "biasnet_weights.weights.h5"
COV_WEIGHTS_PATH = MODEL_DIR / "cov_adapter_weights.weights.h5"

from data_preprocessing_v2 import DATA_DIR_CALIB, CALIB_TRAIN_IDS, CALIB_VAL_ID

WINDOW_SIZE     = 30
TARGET_DT       = 0.1
DEG2RAD         = np.pi / 180.0
RAD2DEG         = 180.0 / np.pi
BATCH_SIZE      = 256
EPOCHS          = 60
LR              = 3e-4
PATIENCE        = 12
BETA_SCALE      = 3.0          # 公式 (17) 的 β
Z_CLIP          = 2.5          # z 值限幅，对应噪声缩放 10^(±β*tanh(2.5)) ≈ 10^(±2.96)

# ============================================================================
# 数据加载（复用 train_ekf.py 的 load_calibration_seq）
# ============================================================================

from train_ekf import load_calibration_seq


def compute_residual_labels(seq, norm_stats, weights_path):
    """
    用固定噪声跑一次 EKF，记录每帧的 NHC 和轮速残差，
    然后反算目标 z 值。

    返回
    ----
    z_labels: (T, 2) — [z_lat_target, z_up_target]
    """
    nav = EKFNavigatorNP(
        weights_path=str(weights_path),
        norm_stats=norm_stats,
        window_size=WINDOW_SIZE,
        ekf_config=DEFAULT_EKF_CONFIG,
        cov_weights_path=None,  # 不用 CovAdapterNet
    )

    gps_valid = seq['gps_valid']
    gx = seq['enu_x']
    gy = seq['enu_y']

    # 模拟无 outage（用全量 GPS 让 EKF 得到最优状态）
    enu_x, enu_y, ekf_h, net_bias, ekf_vx, ekf_vy, ekf_bg = nav.run(
        imu_raw=seq['imu'],
        v_ms=seq['v_ms'],
        gyro_z_rad=seq['gyro_z_rad'],
        gps_enu_x=gx,
        gps_enu_y=gy,
        gps_valid=gps_valid,
        dt=TARGET_DT,
        time_s=seq['Time_s'],
        gps_theta=seq['gps_theta'],
    )

    T = len(seq['Time_s'])
    z_lat = np.zeros(T, dtype=np.float32)
    z_up = np.zeros(T, dtype=np.float32)

    cfg = DEFAULT_EKF_CONFIG
    sigma_lat = np.sqrt(cfg.r_nhc)    # ~0.08 m/s
    sigma_up = np.sqrt(cfg.r_wheel)   # ~0.12 m/s

    for k in range(T):
        # 车体横向速度（NHC 残差）
        v_lat = float(-ekf_vx[k] * np.sin(ekf_h[k]) + ekf_vy[k] * np.cos(ekf_h[k]))
        v_fwd = float(ekf_vx[k] * np.cos(ekf_h[k]) + ekf_vy[k] * np.sin(ekf_h[k]))

        # 轮速残差
        wheel_err = abs(float(seq['v_ms'][k]) - v_fwd) if seq['v_ms'][k] > 0.5 else 0.0

        # 公式 (17) 逆映射：residual → target z
        # σ_dyn = σ_base * 10^(β * tanh(z))
        # z = atanh(log10(σ_dyn / σ_base) / β)
        # 用残差 = |v_lat| 作为目标 σ_dyn
        if k > 0 and abs(v_lat) > 1e-6:
            log_ratio = np.log10(max(abs(v_lat), 1e-4) / sigma_lat)
            z_lat[k] = np.clip(np.arctanh(np.clip(log_ratio / BETA_SCALE, -0.99, 0.99)),
                               -Z_CLIP, Z_CLIP)
        if wheel_err > 1e-6:
            log_ratio = np.log10(max(wheel_err, 1e-4) / sigma_up)
            z_up[k] = np.clip(np.arctanh(np.clip(log_ratio / BETA_SCALE, -0.99, 0.99)),
                              -Z_CLIP, Z_CLIP)

    # 平滑标签（去掉单帧尖刺）
    if T > 11:
        z_lat = median_filter(z_lat, size=11)
        z_up = median_filter(z_up, size=11)

    return np.stack([z_lat, z_up], axis=1).astype(np.float32)


def build_samples(seqs, norm_stats, weights_path, window_size=WINDOW_SIZE):
    """
    对所有数据段计算残差标签，构造滑窗训练样本。

    返回 X (N, W, 6), Y (N, 2)
    """
    # 加载归一化参数
    keys = ['AccX_g', 'AccY_g', 'AccZ_g', 'GyroX_degs', 'GyroY_degs', 'GyroZ_degs']
    mu = np.array([norm_stats[k]['mean'] for k in keys], np.float32)
    std = np.array([norm_stats[k]['std'] for k in keys], np.float32) + 1e-8

    X_list, Y_list = [], []
    for seq in seqs:
        imu_norm = (seq['imu'] - mu) / std
        z_labels = compute_residual_labels(seq, norm_stats, weights_path)

        T = len(imu_norm)
        if T < window_size:
            continue
        for i in range(T - window_size + 1):
            X_list.append(imu_norm[i: i + window_size])
            Y_list.append(z_labels[i + window_size - 1])

    X = np.stack(X_list, axis=0).astype(np.float32)
    Y = np.stack(Y_list, axis=0).astype(np.float32)
    return X, Y


# ============================================================================
# 主训练流程
# ============================================================================

def main():
    print("=" * 60)
    print("CovAdapterNet 训练：AI 噪声参数适配器")
    print("=" * 60)

    # 1. 加载数据
    print(f"\n[1] 加载数据 ({', '.join(CALIB_TRAIN_IDS)} 训练 / {CALIB_VAL_ID} 验证) ...")
    seqs_tr = load_calibration_seq(DATA_DIR_CALIB, CALIB_TRAIN_IDS)
    seqs_val = load_calibration_seq(DATA_DIR_CALIB, [CALIB_VAL_ID])
    if not seqs_tr:
        raise RuntimeError("未找到训练数据")

    norm_stats = load_norm_stats(str(NORM_JSON))

    # 2. 构造训练样本（需要先跑一次固定噪声 EKF 生成标签）
    print("\n[2] 计算残差标签 + 构造训练样本（需要先跑固定噪声 EKF）...")
    X_tr, Y_tr = build_samples(seqs_tr, norm_stats, WEIGHTS_PATH)
    print(f"  训练样本: {len(X_tr)}")
    print(f"  z_lat: mean={Y_tr[:,0].mean():.3f} std={Y_tr[:,0].std():.3f}")
    print(f"  z_up:  mean={Y_tr[:,1].mean():.3f} std={Y_tr[:,1].std():.3f}")

    if seqs_val:
        X_val, Y_val = build_samples(seqs_val, norm_stats, WEIGHTS_PATH)
        print(f"  验证样本: {len(X_val)}")
    else:
        n_val = max(1, int(0.15 * len(X_tr)))
        X_val, Y_val = X_tr[:n_val], Y_tr[:n_val]
        X_tr, Y_tr = X_tr[n_val:], Y_tr[n_val:]

    # 3. 构建模型
    print("\n[3] 构建 CovAdapterNet (输出 [z_lat, z_up]) ...")
    model = CovAdapterNet(window_size=WINDOW_SIZE)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss='huber',        # Huber 对标签噪声鲁棒
        metrics=['mae'],
    )
    model(X_tr[:1])
    model.summary()

    # 4. 训练
    print("\n[4] 开始训练 ...")
    callbacks = [
        ModelCheckpoint(str(COV_WEIGHTS_PATH), save_weights_only=True,
                        monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6,
                          min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=PATIENCE,
                      restore_best_weights=True, verbose=1),
    ]

    history = model.fit(
        X_tr, Y_tr,
        validation_data=(X_val, Y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )

    # 5. 检验
    print("\n[5] 验证集快速检验 ...")
    Y_pred = model(X_val, training=False).numpy()
    for i, name in enumerate(['z_lat', 'z_up']):
        err = Y_pred[:, i] - Y_val[:, i]
        print(f"  {name}: mean_err={err.mean():.4f}  std={err.std():.4f}  "
              f"MAE={np.abs(err).mean():.4f}")

    # 检查噪声缩放范围
    sigma_lat = np.sqrt(DEFAULT_EKF_CONFIG.r_nhc)
    sigma_up = np.sqrt(DEFAULT_EKF_CONFIG.r_wheel)
    factor_lat = 10.0 ** (BETA_SCALE * np.tanh(Y_pred[:, 0]))
    factor_up = 10.0 ** (BETA_SCALE * np.tanh(Y_pred[:, 1]))
    print(f"  噪声缩放因子 lat: [{factor_lat.min():.3f}x, {factor_lat.max():.3f}x]")
    print(f"  噪声缩放因子 up:  [{factor_up.min():.3f}x, {factor_up.max():.3f}x]")

    # 6. 保存信息
    info = {
        'window_size': WINDOW_SIZE,
        'beta_scale': BETA_SCALE,
        'sigma_lat_ms': float(sigma_lat),
        'sigma_up_ms': float(sigma_up),
        'train_samples': int(len(X_tr)),
        'val_samples': int(len(X_val)),
    }
    with open(MODEL_DIR / 'cov_adapter_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n权重已保存到：{COV_WEIGHTS_PATH}")
    print("下一步：python validate_ekf.py 查看效果")


if __name__ == '__main__':
    np.random.seed(42)
    tf.random.set_seed(42)
    main()
