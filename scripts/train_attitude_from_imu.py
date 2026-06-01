#!/usr/bin/env python3
"""scripts/train_attitude_from_imu.py — 从 Data0109 IMU 学习全姿态并输出姿态角。"""

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
    ModelCheckpoint,
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

TARGET_DT = 0.1
WINDOW_SIZE = 200
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4


def quat_norm(q: tf.Tensor) -> tf.Tensor:
    norm = tf.linalg.norm(q, axis=-1, keepdims=True)
    return q / tf.maximum(norm, 1e-8)


def quat_mul(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """四元数乘法，输入格式 [qw, qx, qy, qz]。"""
    aw, ax, ay, az = tf.split(a, 4, axis=-1)
    bw, bx, by, bz = tf.split(b, 4, axis=-1)
    rw = aw * bw - ax * bx - ay * by - az * bz
    rx = aw * bx + ax * bw + ay * bz - az * by
    ry = aw * by - ax * bz + ay * bw + az * bx
    rz = aw * bz + ax * by - ay * bx + az * bw
    return tf.concat([rw, rx, ry, rz], axis=-1)


def quat_conj(q: tf.Tensor) -> tf.Tensor:
    qw, qx, qy, qz = tf.split(q, 4, axis=-1)
    return tf.concat([qw, -qx, -qy, -qz], axis=-1)


def quat_rotate_nav_to_body(q: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
    """将导航系向量 v 旋转到机体系。"""
    q_conj = quat_conj(q)
    v_q = tf.concat([tf.zeros_like(v[..., :1]), v], axis=-1)
    return quat_mul(quat_mul(q_conj, v_q), q)[..., 1:]


def quat_to_yaw(q: tf.Tensor) -> tf.Tensor:
    qw, qx, qy, qz = tf.split(q, 4, axis=-1)
    return tf.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def quat_to_euler(q: tf.Tensor) -> tf.Tensor:
    """四元数 [qw,qx,qy,qz] → 欧拉角 [roll, pitch, yaw] (rad)，ZYX 内旋。"""
    q = quat_norm(q)
    qw, qx, qy, qz = tf.split(q, 4, axis=-1)
    roll = tf.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = tf.asin(2.0 * tf.clip_by_value(qw * qy - qz * qx, -1.0, 1.0))
    yaw = tf.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return tf.concat([roll, pitch, yaw], axis=-1)


def euler_to_quat_zyx(euler_rad: tf.Tensor) -> tf.Tensor:
    """欧拉角 [roll, pitch, yaw] (rad) → 四元数 [qw,qx,qy,qz]，ZYX 内旋。"""
    r, p, y = tf.split(euler_rad, 3, axis=-1)
    cy = tf.cos(y * 0.5)
    sy = tf.sin(y * 0.5)
    cp = tf.cos(p * 0.5)
    sp = tf.sin(p * 0.5)
    cr = tf.cos(r * 0.5)
    sr = tf.sin(r * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return tf.concat([qw, qx, qy, qz], axis=-1)


class _TCNBlock(layers.Layer):
    """两层 causal conv + BN + ReLU + 残差。"""

    def __init__(self, channels: int, kernel: int, dilation: int, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv1D(channels, kernel, dilation_rate=dilation,
                                   padding='causal')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(channels, kernel, dilation_rate=dilation,
                                   padding='causal')
        self.bn2 = layers.BatchNormalization()

    def call(self, x, training=False):
        residual = x
        h = self.conv1(x)
        h = self.bn1(h, training=training)
        h = tf.nn.relu(h)
        h = self.conv2(h)
        h = self.bn2(h, training=training)
        return tf.nn.relu(h + residual)


class AttitudeNet(Model):
    """TCN 姿态网络：从 IMU 窗口回归四元数。"""

    def __init__(self, window_size: int = WINDOW_SIZE, channels: int = 48):
        super().__init__(name='AttitudeNet')
        self.stem_conv = layers.Conv1D(channels, 5, padding='causal')
        self.stem_bn = layers.BatchNormalization()
        dilations = (1, 2, 4, 8, 16, 32)
        self.tcn_blocks = [
            _TCNBlock(channels, 3, d, name=f'tcn_d{d}') for d in dilations
        ]
        self.pool = layers.GlobalAveragePooling1D()
        self.fc1 = layers.Dense(64, activation='relu')
        self.drop = layers.Dropout(0.2)
        self.out_layer = layers.Dense(4, activation='linear')

    def call(self, x, training=False):
        h = self.stem_conv(x)
        h = self.stem_bn(h, training=training)
        h = tf.nn.relu(h)
        for blk in self.tcn_blocks:
            h = blk(h, training=training)
        h = self.pool(h)
        h = self.fc1(h)
        h = self.drop(h, training=training)
        return quat_norm(self.out_layer(h))


def build_windows(seqs, mu, std, window_size=WINDOW_SIZE):
    """从 CMCC 姿态真值构造有监督窗口。仅保留 cmcc_stable 为真的窗口。
    输入通道: IMU 归一化(7) + GPS sin/cos(2) = 9 通道。
    """
    X, Y = [], []
    for seq in seqs:
        imu_norm = normalize_imu(seq['imu'], mu, std)
        gps_theta = seq['gps_theta']
        gps_sin = np.sin(gps_theta).astype(np.float32)
        gps_cos = np.cos(gps_theta).astype(np.float32)
        att_deg = seq['cmcc_attitude_deg']  # (T, 3): pitch, roll, yaw [deg]
        cmcc_stable = seq['cmcc_stable']
        T = len(seq['imu'])
        for i in range(0, T - window_size + 1):
            idx = i + window_size - 1
            if not cmcc_stable[idx]:
                continue
            pitch, roll, yaw = att_deg[idx] * (np.pi / 180.0)
            imu_win = imu_norm[i: i + window_size]
            gps_feat = np.stack([gps_sin[i: i + window_size],
                                 gps_cos[i: i + window_size]], axis=-1)
            x_win = np.concatenate([imu_win, gps_feat], axis=-1)
            X.append(x_win)
            Y.append([roll, pitch, yaw])
    if len(X) == 0:
        raise RuntimeError('没有 cmcc_stable 窗口，请检查数据')
    euler_rad = np.array(Y, dtype=np.float32)  # (N, 3): roll, pitch, yaw [rad]
    return np.stack(X, axis=0).astype(np.float32), euler_rad


def attitude_loss(y_true, y_pred):
    """四元数角度距离损失。y_true: (batch, 3) roll/pitch/yaw [rad]，y_pred: (batch, 4) 四元数。"""
    y_true = tf.cast(y_true, tf.float32)
    q_pred = quat_norm(y_pred)
    q_true = euler_to_quat_zyx(y_true)
    dot = tf.abs(tf.reduce_sum(q_pred * q_true, axis=-1))
    return tf.reduce_mean(1.0 - dot)


def parse_args():
    parser = argparse.ArgumentParser(description='Train CMCC-supervised attitude network from Data0109 IMU')
    parser.add_argument('--window-size', type=int, default=WINDOW_SIZE)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--model-name', type=str, default='attitude_from_imu')
    return parser.parse_args()


def prepare_dataset_from_arrays(X: np.ndarray, Y: np.ndarray, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(X), 8192), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main():
    args = parse_args()
    print('[Info] Loading Data0109 segments...')
    seqs_tr = load_data0109_segments(DATA0109_TRAIN_SEGMENTS)
    seqs_val = load_data0109_segments([DATA0109_VAL_SEGMENT])
    seqs_tr = [s for s in seqs_tr if s is not None]
    seqs_val = [s for s in seqs_val if s is not None]

    mu, std, stats = load_or_compute_norm(seqs_tr, NORM_JSON_0109)
    print(f'[Info] Normalization mean/std loaded for {len(mu)} channels')

    print('[Info] Building training windows...')
    X_train, Y_train = build_windows(seqs_tr, mu, std, args.window_size)
    print(f'[Info] training windows: {len(X_train)}')
    print('[Info] Building validation windows...')
    X_val, Y_val = build_windows(seqs_val, mu, std, args.window_size)
    print(f'[Info] validation windows: {len(X_val)}')

    train_ds = prepare_dataset_from_arrays(X_train, Y_train, args.batch_size, shuffle=True)
    val_ds = prepare_dataset_from_arrays(X_val, Y_val, args.batch_size, shuffle=False)

    print('[Info] Creating model...')
    model = AttitudeNet(window_size=args.window_size)
    model(tf.zeros((1, args.window_size, 9), dtype=tf.float32))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=attitude_loss,
    )

    print(model.summary())

    weights_path = MODEL_DIR / f'{args.model_name}.weights.h5'
    callbacks = [
        ModelCheckpoint(str(weights_path), save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        EarlyStopping(patience=12, restore_best_weights=True, monitor='val_loss'),
    ]

    print('[Info] Training...')
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    print('[Info] Training finished.')
    print(f'[Info] Saving best weights to {weights_path}')
    model.save_weights(str(weights_path))

    print('[Info] Evaluating validation set...')
    all_roll_err, all_pitch_err, all_yaw_err = [], [], []
    for X_batch, Y_batch in val_ds:
        q_pred = model(X_batch, training=False)
        euler_pred = quat_to_euler(q_pred).numpy()  # (B, 3): roll, pitch, yaw [rad]
        euler_true = Y_batch.numpy()                 # (B, 3): roll, pitch, yaw [rad]
        for i in range(3):
            diff = euler_pred[:, i] - euler_true[:, i]
            err = np.abs(np.arctan2(np.sin(diff), np.cos(diff)))  # 角度环绕处理
            [all_roll_err, all_pitch_err, all_yaw_err][i].extend(err.tolist())
    if all_yaw_err:
        roll_rmse = np.sqrt(np.mean(np.array(all_roll_err) ** 2)) * 180.0 / np.pi
        pitch_rmse = np.sqrt(np.mean(np.array(all_pitch_err) ** 2)) * 180.0 / np.pi
        yaw_rmse = np.sqrt(np.mean(np.array(all_yaw_err) ** 2)) * 180.0 / np.pi
        print(f'[Eval] Roll  RMSE = {roll_rmse:.3f} deg')
        print(f'[Eval] Pitch RMSE = {pitch_rmse:.3f} deg')
        print(f'[Eval] Yaw   RMSE = {yaw_rmse:.3f} deg')
    else:
        print('[Eval] No valid CMCC windows in validation set.')


if __name__ == '__main__':
    main()
