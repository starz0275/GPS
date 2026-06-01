#!/usr/bin/env python3
"""scripts/train_bias_angle.py — 端到端学习零偏(6) + 安装角(1)。

思路 (参考 AI-IMU Dead-Reckoning 论文):
  IMU窗口 → TCN → 零偏(ba_x/y/z, bg_x/y/z) + 安装角(ψ_install)
  → 修正IMU → 航位推算积分 → 预测轨迹
  → 与 GPS 轨迹真值对比 → 损失反传优化网络

输出: 6轴零偏 (acc 3 + gyro 3) + 1个安装角 (yaw, IMU→车体)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
)

from data0109_loader import (
    DATA0109_TRAIN_SEGMENTS,
    DATA0109_VAL_SEGMENT,
    load_data0109_segments,
)
from train_ekf import load_or_compute_norm, normalize_imu

ROOT = Path(__file__).resolve().parent.parent
NORM_JSON_0109 = ROOT / "preprocessed_data" / "normalization_stats_data0109.json"
MODEL_DIR = ROOT / "trained_models"
MODEL_DIR.mkdir(exist_ok=True)

DEG2RAD = np.pi / 180.0
TARGET_DT = 0.1  # 10 Hz


# ====================================================================
# 不同iable 航位推算
# ====================================================================

@tf.function
def rotate_imu_to_body(acc_imu, gyro_imu_z, install_angle):
    """将 IMU 系加速度/角速度旋转换算到车体系。

    install_angle (ψ): IMU→车体的 yaw 旋转角 [rad]。
    假设 pitch/install roll 为 0 (仅 yaw 偏差)。

    旋转矩阵:
        R = [[ cosψ, sinψ, 0],
             [-sinψ, cosψ, 0],
             [    0,    0, 1]]
    """
    ca = tf.cos(install_angle)  # (B, 1)
    sa = tf.sin(install_angle)
    ax = acc_imu[:, 0:1]  # (B,)
    ay = acc_imu[:, 1:2]
    az = acc_imu[:, 2:3]
    acc_body_x = ca * ax + sa * ay       # (B, 1)
    acc_body_y = -sa * ax + ca * ay
    acc_body_z = az  # Z 轴不变
    acc_body = tf.concat([acc_body_x, acc_body_y, acc_body_z], axis=-1)  # (B, 3)
    # 陀螺仪 Z 轴: 假设安装偏航角不影响 Z 轴角速度
    gyro_body_z = gyro_imu_z  # (B,)
    return acc_body, gyro_body_z


@tf.function
def integrate_trajectory(init_east, init_north, init_heading,
                         acc_corrected, gyro_corrected_z, v_ms, dt):
    """可微分的航位推算积分。

    Args:
        init_east, init_north: 初始位置 (B,)
        init_heading:          初始航向角, 即 psi (B,) [rad], 0=East
        acc_corrected:         去零偏加速度 (B, T, 3) [g]
        gyro_corrected_z:      去零偏陀螺仪 Z (B, T) [deg/s]
        v_ms:                  车速 (B, T) [m/s]
        dt:                    时间步长 [s]

    Returns:
        pred_east, pred_north: 预测轨迹 (B, T+1)
        pred_heading:          预测航向 (B, T+1) [rad]
    """
    batch_size = tf.shape(acc_corrected)[0]
    seq_len = tf.shape(acc_corrected)[1]  # T

    # 初始化
    east = init_east       # (B,)
    north = init_north
    psi = init_heading     # rad, 从 GPS 航向得到

    east_list = tf.TensorArray(dtype=tf.float32, size=seq_len + 1,
                               dynamic_size=False)
    north_list = tf.TensorArray(dtype=tf.float32, size=seq_len + 1,
                                dynamic_size=False)
    psi_list = tf.TensorArray(dtype=tf.float32, size=seq_len + 1,
                              dynamic_size=False)

    east_list = east_list.write(0, east)
    north_list = north_list.write(0, north)
    psi_list = psi_list.write(0, psi)

    g = 9.80665  # m/s²

    def body(i, east, north, psi, east_list, north_list, psi_list):
        # 当前时刻的输入
        ax = acc_corrected[:, i, 0]   # (B,) [g]
        ay = acc_corrected[:, i, 1]   # (B,) [g]
        dpsi = gyro_corrected_z[:, i] # (B,) [deg/s]
        vi = v_ms[:, i]               # (B,) [m/s]

        # 加速度: g → m/s²
        ax_ms2 = ax * g
        ay_ms2 = ay * g

        # 机体系加速度 → 导航系 (East, North)
        cp = tf.cos(psi)
        sp = tf.sin(psi)
        acc_e = cp * ax_ms2 - sp * ay_ms2   # (B,)
        acc_n = sp * ax_ms2 + cp * ay_ms2

        # 积分速度 (以车速为基准, 加速度做修正)
        dv_e = acc_e * dt   # (B,)
        dv_n = acc_n * dt

        # 积分位置: 用速度积分 (车速沿航向)
        v_e = vi * cp + dv_e  # (B,)
        v_n = vi * sp + dv_n

        east = east + v_e * dt
        north = north + v_n * dt
        psi = psi + dpsi * DEG2RAD * dt  # rad

        east_list = east_list.write(i + 1, east)
        north_list = north_list.write(i + 1, north)
        psi_list = psi_list.write(i + 1, psi)

        return i + 1, east, north, psi, east_list, north_list, psi_list

    def cond(i, *_):
        return i < seq_len

    _, east, north, psi, east_list, north_list, psi_list = tf.while_loop(
        cond, body,
        [tf.constant(0), east, north, psi, east_list, north_list, psi_list],
        parallel_iterations=1,
    )

    pred_east = tf.transpose(east_list.stack())    # (B, T+1)
    pred_north = tf.transpose(north_list.stack())
    pred_heading = tf.transpose(psi_list.stack())
    return pred_east, pred_north, pred_heading


# ====================================================================
# 网络: TCN → 零偏(6) + 安装角(1)
# ====================================================================

class _TCNBlock(layers.Layer):
    def __init__(self, channels, kernel, dilation, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv1D(channels, kernel, dilation_rate=dilation,
                                   padding='causal')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(channels, kernel, dilation_rate=dilation,
                                   padding='causal')
        self.bn2 = layers.BatchNormalization()

    def call(self, x, training=False):
        residual = x
        h = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        h = self.bn2(self.conv2(h), training=training)
        return tf.nn.relu(h + residual)


class BiasAngleNet(Model):
    """TCN 网络: 从 IMU 窗口估计零偏(6) + 安装角(1)。"""

    def __init__(self, window_size: int = 200, channels: int = 48, **kwargs):
        super().__init__(**kwargs)
        self.stem_conv = layers.Conv1D(channels, 5, padding='causal')
        self.stem_bn = layers.BatchNormalization()
        dilations = (1, 2, 4, 8, 16, 32)
        self.tcn_blocks = [
            _TCNBlock(channels, 3, d, name=f'tcn_d{d}') for d in dilations
        ]
        self.pool = layers.GlobalAveragePooling1D()
        self.fc1 = layers.Dense(64, activation='relu')
        self.drop = layers.Dropout(0.2)
        # 输出: [ba_x, ba_y, ba_z, bg_x, bg_y, bg_z, install_angle]
        #       acc_bias[g], gyro_bias[deg/s], install_angle[rad]
        self.out_layer = layers.Dense(7, activation='linear')

    def call(self, x, training=False):
        h = self.stem_bn(self.stem_conv(x), training=training)
        h = tf.nn.relu(h)
        for blk in self.tcn_blocks:
            h = blk(h, training=training)
        h = self.pool(h)
        h = self.drop(self.fc1(h), training=training)
        out = self.out_layer(h)  # (B, 7)
        # 分离零偏和安装角
        bias = out[:, :6]           # (B, 6): [ba_x, ba_y, ba_z, bg_x, bg_y, bg_z]
        install_angle = out[:, 6:7] # (B, 1): install yaw [rad]
        return bias, install_angle


# ====================================================================
# 窗口构造
# ====================================================================

def build_windows(seqs, mu, std, window_size=200, use_install_gt=True):
    """构造训练窗口: IMU + GPS 轨迹真值 + 安装角真值。"""
    X, V, GPS_EAST, GPS_NORTH, GPS_HEAD, GPS_VALID = [], [], [], [], [], []
    INSTALL_PITCH_GT, INSTALL_YAW_GT = [], []  # 安装角真值

    for seq in seqs:
        imu_norm = normalize_imu(seq['imu'], mu, std)
        v_ms = seq['v_ms']
        enu_e = seq['enu_x']
        enu_n = seq['enu_y']
        gps_theta = seq['gps_theta']
        gps_valid = seq['gps_valid']
        T = len(seq['imu'])

        # 安装角真值
        has_install = 'cmcc_install_deg' in seq
        cmcc_ok = seq.get('cmcc_ok', np.zeros(T, dtype=bool))
        if has_install:
            install_deg = seq['cmcc_install_deg']  # (T, 2): rbv_pitch, rbv_yaw [deg]

        for i in range(0, T - window_size):
            # 窗口起点的 GPS 信息作为初始条件
            if not gps_valid[i]:
                continue
            # 窗口内需要有足够的 GPS 有效点
            end = i + window_size
            gps_slice = gps_valid[i:end]
            if gps_slice.sum() < window_size * 0.5:
                continue

            X.append(imu_norm[i:end])
            V.append(v_ms[i:end])
            GPS_EAST.append(enu_e[i:end + 1])    # T+1 个点
            GPS_NORTH.append(enu_n[i:end + 1])
            GPS_HEAD.append(gps_theta[i:end + 1])
            GPS_VALID.append(gps_valid[i:end + 1])

            # 安装角真值: 取窗口内 cmcc_ok 的均值
            if has_install and use_install_gt:
                win_ok = cmcc_ok[i:end]
                if win_ok.sum() > 0:
                    pitch_gt = np.mean(install_deg[i:end, 0][win_ok])  # rbv_pitch
                    yaw_gt = np.mean(install_deg[i:end, 1][win_ok])    # rbv_yaw
                else:
                    pitch_gt = 0.0
                    yaw_gt = 0.0
                INSTALL_PITCH_GT.append(pitch_gt)
                INSTALL_YAW_GT.append(yaw_gt)

    if len(X) == 0:
        raise RuntimeError('没有足够的 GPS 有效窗口')

    result = [
        np.stack(X).astype(np.float32),       # (N, T, 7)
        np.stack(V).astype(np.float32),        # (N, T)
        np.stack(GPS_EAST).astype(np.float32), # (N, T+1)
        np.stack(GPS_NORTH).astype(np.float32),
        np.stack(GPS_HEAD).astype(np.float32),
        np.stack(GPS_VALID).astype(bool),
    ]

    if has_install and use_install_gt and len(INSTALL_PITCH_GT) > 0:
        result.append(np.array(INSTALL_PITCH_GT, dtype=np.float32))
        result.append(np.array(INSTALL_YAW_GT, dtype=np.float32))

    return tuple(result)


# ====================================================================
# 损失函数
# ====================================================================

def trajectory_loss(pred_east, pred_north, pred_heading,
                    true_east, true_north, true_heading, true_valid,
                    pred_install_angle=None, true_install_yaw=None,
                    install_weight=1.0):
    """轨迹 RMSE + 航向 RMSE + 安装角监督损失。"""
    # 位置误差
    de = pred_east - true_east         # (B, T+1)
    dn = pred_north - true_north
    pos_err_sq = de * de + dn * dn     # (B, T+1)

    # 用 GPS 有效帧计算均值
    valid_f = tf.cast(true_valid, tf.float32)  # (B, T+1)
    # 有效帧至少为 1
    n_valid = tf.maximum(tf.reduce_sum(valid_f, axis=1), 1.0)  # (B,)
    pos_mse = tf.reduce_sum(pos_err_sq * valid_f, axis=1) / n_valid  # (B,)
    pos_rmse = tf.sqrt(tf.maximum(pos_mse, 1e-12))  # (B,)

    # 航向误差 (角度环绕)
    head_diff = pred_heading - true_heading
    head_err = tf.abs(tf.atan2(tf.sin(head_diff), tf.cos(head_diff)))  # (B, T+1)
    head_mse = tf.reduce_sum(head_err * valid_f, axis=1) / n_valid     # (B,)

    # 安装角监督损失 (如果提供)
    install_loss = tf.constant(0.0, dtype=tf.float32)
    if pred_install_angle is not None and true_install_yaw is not None:
        # pred_install_angle: (B, 1) [rad]
        # true_install_yaw: (B,) [deg]
        pred_yaw_deg = pred_install_angle[:, 0] * (180.0 / np.pi)  # (B,) [deg]
        # 角度差 (考虑环绕)
        angle_diff = pred_yaw_deg - true_install_yaw
        angle_diff = tf.atan2(tf.sin(angle_diff * DEG2RAD), tf.cos(angle_diff * DEG2RAD)) / DEG2RAD
        install_loss = tf.reduce_mean(tf.square(angle_diff))  # MSE in degrees

    total_loss = tf.reduce_mean(pos_rmse) + 0.1 * tf.reduce_mean(head_mse) + install_weight * install_loss
    return total_loss, tf.reduce_mean(pos_rmse), tf.reduce_mean(head_mse), install_loss


# ====================================================================
# 单步积分 (batch 内并行)
# ====================================================================

def batch_integrate(imu_batch, v_batch, bias, install_angle,
                    init_east, init_north, init_heading):
    """对一个 batch 做航位推算积分。

    Args:
        imu_batch:     (B, T, 7) 归一化 IMU
        v_batch:       (B, T) 车速 [m/s]
        bias:          (B, 6) [ba_x, ba_y, ba_z, bg_x, bg_y, bg_z]
        install_angle: (B, 1) [rad]
        init_east:     (B,) 初始东向位置
        init_north:    (B,) 初始北向位置
        init_heading:  (B,) 初始航向 [rad]

    Returns:
        pred_east, pred_north, pred_heading: (B, T+1)
    """
    B = tf.shape(imu_batch)[0]
    T = tf.shape(imu_batch)[1]

    # 提取 IMU 原始值 (归一化后的, 需要先还原, 然后去零偏)
    # 但这里我们直接在归一化域操作:
    # 归一化 IMU = (raw - mu) / std
    # 去零偏: raw_corrected = (raw - bias) = (norm * std + mu - bias)
    # 为了端到端学习, 我们让网络直接学归一化域的偏移:
    # norm_corrected = norm - bias_norm, 其中 bias_norm = bias / std
    # 但为了简化, 直接在原始值上操作更清晰
    # → 不归一化, 直接用原始 IMU 值
    # 然而网络输入还是归一化的, 输出的零偏也应该是物理单位
    pass


# ====================================================================
# 数据准备
# ====================================================================

def prepare_dataset(X, V, ge, gn, gh, gv, batch_size, shuffle=True,
                     install_pitch_gt=None, install_yaw_gt=None):
    if install_pitch_gt is not None and install_yaw_gt is not None:
        ds = tf.data.Dataset.from_tensor_slices((X, V, ge, gn, gh, gv, install_pitch_gt, install_yaw_gt))
    else:
        ds = tf.data.Dataset.from_tensor_slices((X, V, ge, gn, gh, gv))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(X), 4096), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main():
    parser = argparse.ArgumentParser(
        description='End-to-end train bias(6) + install_angle(1) via trajectory loss + install GT')
    parser.add_argument('--window-size', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--model-name', type=str, default='bias_angle')
    parser.add_argument('--install-weight', type=float, default=1.0,
                        help='安装角监督损失权重 (默认1.0)')
    parser.add_argument('--no-install-gt', action='store_true',
                        help='不使用安装角真值监督')
    args = parser.parse_args()

    use_install_gt = not args.no_install_gt

    print('[Info] Loading Data0109 segments...')
    seqs_tr = load_data0109_segments(DATA0109_TRAIN_SEGMENTS)
    seqs_val = load_data0109_segments([DATA0109_VAL_SEGMENT])
    seqs_tr = [s for s in seqs_tr if s is not None]
    seqs_val = [s for s in seqs_val if s is not None]

    mu, std, _ = load_or_compute_norm(seqs_tr, NORM_JSON_0109)

    print('[Info] Building windows...')
    W = args.window_size

    # 构造训练窗口 (含安装角真值)
    train_result = build_windows(seqs_tr, mu, std, W, use_install_gt=use_install_gt)
    val_result = build_windows(seqs_val, mu, std, W, use_install_gt=use_install_gt)

    if use_install_gt and len(train_result) > 6:
        X_tr, V_tr, ge_tr, gn_tr, gh_tr, gv_tr, ip_tr, iy_tr = train_result
        X_val, V_val, ge_val, gn_val, gh_val, gv_val, ip_val, iy_val = val_result
        print(f'  Train windows: {len(X_tr)} (install_gt: {len(ip_tr)})')
        print(f'  Val   windows: {len(X_val)} (install_gt: {len(ip_val)})')
        print(f'  Install yaw GT (train): mean={np.mean(iy_tr):.2f}° std={np.std(iy_tr):.2f}°')
        print(f'  Install yaw GT (val):   mean={np.mean(iy_val):.2f}° std={np.std(iy_val):.2f}°')
    else:
        X_tr, V_tr, ge_tr, gn_tr, gh_tr, gv_tr = train_result[:6]
        X_val, V_val, ge_val, gn_val, gh_val, gv_val = val_result[:6]
        ip_tr, iy_tr = None, None
        ip_val, iy_val = None, None
        print(f'  Train windows: {len(X_tr)}')
        print(f'  Val   windows: {len(X_val)}')

    train_ds = prepare_dataset(X_tr, V_tr, ge_tr, gn_tr, gh_tr, gv_tr,
                               args.batch_size, shuffle=True,
                               install_pitch_gt=ip_tr, install_yaw_gt=iy_tr)
    val_ds = prepare_dataset(X_val, V_val, ge_val, gn_val, gh_val, gv_val,
                             args.batch_size, shuffle=False,
                             install_pitch_gt=ip_val, install_yaw_gt=iy_val)

    print('[Info] Creating model...')
    model = BiasAngleNet(window_size=W)
    model(tf.zeros((1, W, 7), dtype=tf.float32))
    print(f'  Output: 6 bias + 1 install_angle = 7 params')
    print(f'  Total params: {model.count_params()}')
    print(f'  Install GT supervision: {use_install_gt}')
    print(f'  Install loss weight: {args.install_weight}')

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    @tf.function
    def train_step(imu_norm, v_ms, true_e, true_n, true_h, true_v,
                   true_pitch_gt=None, true_yaw_gt=None):
        with tf.GradientTape() as tape:
            bias, install_angle = model(imu_norm, training=True)

            std_t = tf.constant(std, dtype=tf.float32)
            mu_t = tf.constant(mu, dtype=tf.float32)

            acc_raw = imu_norm[:, :, :3] * std_t[:3] + mu_t[:3]
            gyro_z_raw = (imu_norm[:, :, 5] * std_t[5] + mu_t[5])

            ba = bias[:, :3]
            bg_z = bias[:, 5]
            acc_corr = acc_raw - tf.expand_dims(ba, axis=1)
            gyro_corr_z = gyro_z_raw - tf.expand_dims(bg_z, axis=1)

            B_s = tf.shape(acc_corr)[0]
            T_s = tf.shape(acc_corr)[1]
            acc_corr_flat = tf.reshape(acc_corr, (-1, 3))
            gyro_corr_z_flat = tf.reshape(gyro_corr_z, (-1,))
            angle_flat = tf.repeat(install_angle, T_s, axis=0)
            acc_body_flat, gyro_body_z_flat = rotate_imu_to_body(
                acc_corr_flat, gyro_corr_z_flat, angle_flat)
            acc_body = tf.reshape(acc_body_flat, (B_s, T_s, 3))
            gyro_body_z = tf.reshape(gyro_body_z_flat, (B_s, T_s))

            init_e = true_e[:, 0]
            init_n = true_n[:, 0]
            init_h = true_h[:, 0]

            pred_e, pred_n, pred_h = integrate_trajectory(
                init_e, init_n, init_h,
                acc_body, gyro_body_z, v_ms, TARGET_DT)

            # 损失 (含安装角监督)
            loss, pos_loss, head_loss, install_loss = trajectory_loss(
                pred_e, pred_n, pred_h,
                true_e, true_n, true_h, true_v,
                pred_install_angle=install_angle,
                true_install_yaw=true_yaw_gt,
                install_weight=args.install_weight)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, pos_loss, head_loss, install_loss

    @tf.function
    def val_step(imu_norm, v_ms, true_e, true_n, true_h, true_v,
                 true_pitch_gt=None, true_yaw_gt=None):
        bias, install_angle = model(imu_norm, training=False)
        std_t = tf.constant(std, dtype=tf.float32)
        mu_t = tf.constant(mu, dtype=tf.float32)

        acc_raw = imu_norm[:, :, :3] * std_t[:3] + mu_t[:3]
        gyro_z_raw = (imu_norm[:, :, 5] * std_t[5] + mu_t[5])

        ba = bias[:, :3]
        bg_z = bias[:, 5]
        acc_corr = acc_raw - tf.expand_dims(ba, axis=1)
        gyro_corr_z = gyro_z_raw - tf.expand_dims(bg_z, axis=1)

        B_s = tf.shape(acc_corr)[0]
        T_s = tf.shape(acc_corr)[1]
        acc_corr_flat = tf.reshape(acc_corr, (-1, 3))
        gyro_corr_z_flat = tf.reshape(gyro_corr_z, (-1,))
        angle_flat = tf.repeat(install_angle, T_s, axis=0)
        acc_body_flat, gyro_body_z_flat = rotate_imu_to_body(
            acc_corr_flat, gyro_corr_z_flat, angle_flat)
        acc_body = tf.reshape(acc_body_flat, (B_s, T_s, 3))
        gyro_body_z = tf.reshape(gyro_body_z_flat, (B_s, T_s))

        init_e = true_e[:, 0]
        init_n = true_n[:, 0]
        init_h = true_h[:, 0]

        pred_e, pred_n, pred_h = integrate_trajectory(
            init_e, init_n, init_h,
            acc_body, gyro_body_z, v_ms, TARGET_DT)

        loss, pos_loss, head_loss, install_loss = trajectory_loss(
            pred_e, pred_n, pred_h,
            true_e, true_n, true_h, true_v,
            pred_install_angle=install_angle,
            true_install_yaw=true_yaw_gt,
            install_weight=args.install_weight)
        return loss, pos_loss, head_loss, install_loss, bias, install_angle

    # === 训练循环 ===
    print('\n[Info] Training...')
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15

    for epoch in range(args.epochs):
        # Train
        train_loss_sum, train_pos_sum, train_head_sum, train_inst_sum = 0.0, 0.0, 0.0, 0.0
        n_train = 0
        for batch in train_ds:
            if use_install_gt and len(batch) == 8:
                imu, v, ge, gn, gh, gv, ip, iy = batch
                loss, pos_l, head_l, inst_l = train_step(imu, v, ge, gn, gh, gv, ip, iy)
            else:
                imu, v, ge, gn, gh, gv = batch[:6]
                loss, pos_l, head_l, inst_l = train_step(imu, v, ge, gn, gh, gv)
            bs = tf.shape(imu)[0].numpy()
            train_loss_sum += loss.numpy() * bs
            train_pos_sum += pos_l.numpy() * bs
            train_head_sum += head_l.numpy() * bs
            train_inst_sum += inst_l.numpy() * bs
            n_train += bs

        # Val
        val_loss_sum, val_pos_sum, val_head_sum, val_inst_sum = 0.0, 0.0, 0.0, 0.0
        n_val = 0
        last_bias, last_angle = None, None
        for batch in val_ds:
            if use_install_gt and len(batch) == 8:
                imu, v, ge, gn, gh, gv, ip, iy = batch
                loss, pos_l, head_l, inst_l, b, a = val_step(imu, v, ge, gn, gh, gv, ip, iy)
            else:
                imu, v, ge, gn, gh, gv = batch[:6]
                loss, pos_l, head_l, inst_l, b, a = val_step(imu, v, ge, gn, gh, gv)
            bs = tf.shape(imu)[0].numpy()
            val_loss_sum += loss.numpy() * bs
            val_pos_sum += pos_l.numpy() * bs
            val_head_sum += head_l.numpy() * bs
            val_inst_sum += inst_l.numpy() * bs
            n_val += bs
            last_bias = b.numpy()
            last_angle = a.numpy()

        tl = train_loss_sum / n_train
        vl = val_loss_sum / n_val
        tpl = train_pos_sum / n_train
        vpl = val_pos_sum / n_val
        thl = train_head_sum / n_train
        vhl = val_head_sum / n_val
        til = train_inst_sum / n_train
        vil = val_inst_sum / n_val

        if (epoch + 1) % 5 == 0 or epoch == 0:
            angle_deg = np.mean(last_angle) * 180 / np.pi if last_angle is not None else 0
            print(f'Epoch {epoch+1:3d}/{args.epochs}  '
                  f'train: loss={tl:.2f} pos={tpl:.2f}m head={thl:.4f}rad inst={til:.2f}°  |  '
                  f'val: loss={vl:.2f} pos={vpl:.2f}m head={vhl:.4f}rad inst={vil:.2f}°  |  '
                  f'pred_angle={angle_deg:.2f}°')

        if vl < best_val_loss:
            best_val_loss = vl
            patience_counter = 0
            weights_path = MODEL_DIR / f'{args.model_name}.weights.h5'
            model.save_weights(str(weights_path))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'[Info] Early stopping at epoch {epoch+1}')
                break

    # === 加载最优权重, 输出结果 ===
    model.load_weights(str(MODEL_DIR / f'{args.model_name}.weights.h5'))

    print('\n[Result] Best model on validation set:')
    print(f'  Best val loss: {best_val_loss:.2f} m')

    # 在验证集上取平均零偏和安装角
    all_bias, all_angle = [], []
    for batch in val_ds:
        imu = batch[0]
        bias, angle = model(imu, training=False)
        all_bias.append(bias.numpy())
        all_angle.append(angle.numpy())

    bias_mean = np.concatenate(all_bias, axis=0).mean(axis=0)
    angle_mean = np.concatenate(all_angle, axis=0).mean()

    print(f'\n  安装角 (yaw): {angle_mean * 180 / np.pi:.4f}°')
    print(f'  加速度零偏 [g]:    ba_x={bias_mean[0]:.6f}  ba_y={bias_mean[1]:.6f}  ba_z={bias_mean[2]:.6f}')
    print(f'  陀螺仪零偏 [°/s]:  bg_x={bias_mean[3]:.6f}  bg_y={bias_mean[4]:.6f}  bg_z={bias_mean[5]:.6f}')

    if use_install_gt and iy_val is not None:
        print(f'\n  安装角真值 (yaw): {np.mean(iy_val):.4f}°')
        print(f'  安装角误差: {abs(angle_mean * 180 / np.pi - np.mean(iy_val)):.4f}°')

    # 保存
    result = {
        'install_angle_deg': float(angle_mean * 180 / np.pi),
        'install_angle_rad': float(angle_mean),
        'acc_bias_g': {'x': float(bias_mean[0]), 'y': float(bias_mean[1]), 'z': float(bias_mean[2])},
        'gyro_bias_degs': {'x': float(bias_mean[3]), 'y': float(bias_mean[4]), 'z': float(bias_mean[5])},
        'use_install_gt': use_install_gt,
        'install_weight': args.install_weight,
    }
    if use_install_gt and iy_val is not None:
        result['install_gt_yaw_deg'] = float(np.mean(iy_val))
        result['install_error_deg'] = abs(float(angle_mean * 180 / np.pi) - float(np.mean(iy_val)))
    result_path = MODEL_DIR / f'{args.model_name}_result.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\n  Saved: {result_path}')


if __name__ == '__main__':
    main()
