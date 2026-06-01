#!/usr/bin/env python3
"""端到端学习 6 轴全局零偏 + 3 轴全局安装角，并允许小零偏残差微调。

训练链路:
  IMU窗口 -> BiasAngleNet 输出小零偏残差
  全局零偏 + 残差 -> 去零偏
  全局安装角 roll/pitch/yaw -> IMU系旋到车体系
  航位推算 -> 轨迹与 GNSS ENU 真值对比 -> 反向优化

最终标定输出以全局 6 零偏和全局 3 安装角为准；残差只作为慢漂补偿诊断。
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
from tensorflow.keras import Model, layers

from data0109_loader import (
    DATA0109_ALL_SEGMENTS,
    DATA0109_TRAIN_SEGMENTS,
    DATA0109_VAL_SEGMENT,
    load_data0109_segments,
)
from train_ekf import load_or_compute_norm, normalize_imu

NORM_JSON_0109 = ROOT / "preprocessed_data" / "normalization_stats_data0109.json"
MODEL_DIR = ROOT / "trained_models"
MODEL_DIR.mkdir(exist_ok=True)

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
TARGET_DT = 0.1


def wrap_angle_rad(x):
    return tf.atan2(tf.sin(x), tf.cos(x))


def wrap_angle_np(x):
    return np.arctan2(np.sin(x), np.cos(x))


def rotation_imu_to_body(v, install_rpy):
    """用 ZYX 欧拉角将 IMU 系向量旋到车体系。

    install_rpy: [roll, pitch, yaw] rad，全局固定安装角。
    v: (..., 3)
    """
    roll, pitch, yaw = tf.unstack(install_rpy)
    cr, sr = tf.cos(roll), tf.sin(roll)
    cp, sp = tf.cos(pitch), tf.sin(pitch)
    cy, sy = tf.cos(yaw), tf.sin(yaw)

    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr
    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr
    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr

    x, y, z = tf.unstack(v, axis=-1)
    out_x = r00 * x + r01 * y + r02 * z
    out_y = r10 * x + r11 * y + r12 * z
    out_z = r20 * x + r21 * y + r22 * z
    return tf.stack([out_x, out_y, out_z], axis=-1)


@tf.function
def integrate_trajectory(init_east, init_north, init_heading,
                         acc_body, gyro_body_z, v_ms, dt):
    """可微航位推算，返回 (B,T+1) 的 east/north/heading。"""
    seq_len = tf.shape(acc_body)[1]
    east = init_east
    north = init_north
    psi = init_heading

    east_list = tf.TensorArray(tf.float32, size=seq_len + 1)
    north_list = tf.TensorArray(tf.float32, size=seq_len + 1)
    psi_list = tf.TensorArray(tf.float32, size=seq_len + 1)
    east_list = east_list.write(0, east)
    north_list = north_list.write(0, north)
    psi_list = psi_list.write(0, psi)

    g = tf.constant(9.80665, tf.float32)

    def body(i, east, north, psi, east_list, north_list, psi_list):
        ax = acc_body[:, i, 0] * g
        ay = acc_body[:, i, 1] * g
        wz = gyro_body_z[:, i]
        vi = v_ms[:, i]

        cp = tf.cos(psi)
        sp = tf.sin(psi)
        acc_e = cp * ax - sp * ay
        acc_n = sp * ax + cp * ay

        # 轮速提供主速度约束，加速度只做短时修正项。
        v_e = vi * cp + acc_e * dt
        v_n = vi * sp + acc_n * dt

        east = east + v_e * dt
        north = north + v_n * dt
        psi = wrap_angle_rad(psi + wz * DEG2RAD * dt)

        east_list = east_list.write(i + 1, east)
        north_list = north_list.write(i + 1, north)
        psi_list = psi_list.write(i + 1, psi)
        return i + 1, east, north, psi, east_list, north_list, psi_list

    def cond(i, *_):
        return i < seq_len

    _, _, _, _, east_list, north_list, psi_list = tf.while_loop(
        cond,
        body,
        [tf.constant(0), east, north, psi, east_list, north_list, psi_list],
        parallel_iterations=1,
    )

    return (
        tf.transpose(east_list.stack()),
        tf.transpose(north_list.stack()),
        tf.transpose(psi_list.stack()),
    )


class _TCNBlock(layers.Layer):
    def __init__(self, channels, kernel, dilation, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv1D(channels, kernel, dilation_rate=dilation, padding="causal")
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(channels, kernel, dilation_rate=dilation, padding="causal")
        self.bn2 = layers.BatchNormalization()

    def call(self, x, training=False):
        residual = x
        h = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        h = self.bn2(self.conv2(h), training=training)
        return tf.nn.relu(h + residual)


class BiasAngleNet(Model):
    """全局 6 零偏 + 全局 3 安装角 + 窗口零偏残差网络。"""

    def __init__(
        self,
        window_size: int = 200,
        channels: int = 48,
        install_limit_deg: float = 10.0,
        residual_acc_scale_g: float = 0.02,
        residual_gyro_scale_degs: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.install_limit_rad = float(install_limit_deg * DEG2RAD)
        self.residual_scale = tf.constant(
            [residual_acc_scale_g] * 3 + [residual_gyro_scale_degs] * 3,
            dtype=tf.float32,
        )

        self.bias_base = self.add_weight(
            name="bias_base",
            shape=(6,),
            initializer="zeros",
            trainable=True,
        )
        self.install_raw = self.add_weight(
            name="install_raw",
            shape=(3,),
            initializer="zeros",
            trainable=True,
        )

        self.stem_conv = layers.Conv1D(channels, 5, padding="causal")
        self.stem_bn = layers.BatchNormalization()
        self.tcn_blocks = [
            _TCNBlock(channels, 3, d, name=f"tcn_d{d}") for d in (1, 2, 4, 8, 16, 32)
        ]
        self.pool = layers.GlobalAveragePooling1D()
        self.fc1 = layers.Dense(64, activation="relu")
        self.drop = layers.Dropout(0.2)
        self.residual_out = layers.Dense(6, activation="tanh")

    @property
    def install_rpy(self):
        return tf.tanh(self.install_raw) * self.install_limit_rad

    def call(self, x, training=False):
        h = tf.nn.relu(self.stem_bn(self.stem_conv(x), training=training))
        for block in self.tcn_blocks:
            h = block(h, training=training)
        h = self.pool(h)
        h = self.drop(self.fc1(h), training=training)
        residual = self.residual_out(h) * self.residual_scale
        bias_used = tf.expand_dims(self.bias_base, axis=0) + residual
        return bias_used, residual, self.install_rpy

    def fixed_bias_batch(self, batch_size):
        return tf.tile(tf.expand_dims(self.bias_base, axis=0), [batch_size, 1])


def build_windows(seqs, mu, std, window_size=200):
    X, V, GPS_EAST, GPS_NORTH, GPS_HEAD = [], [], [], [], []
    POS_VALID, HEAD_VALID = [], []

    for seq in seqs:
        imu_norm = normalize_imu(seq["imu"], mu, std)
        v_ms = seq["v_ms"]
        enu_e = seq["enu_x"]
        enu_n = seq["enu_y"]
        gps_theta = seq["gps_theta"]
        pos_valid = seq.get("gps_pos_valid", seq["gps_valid"])
        head_valid = seq["gps_valid"]
        T = len(seq["imu"])

        for i in range(0, T - window_size):
            end = i + window_size
            if not head_valid[i]:
                continue
            if pos_valid[i:end + 1].sum() < (window_size + 1) * 0.5:
                continue

            X.append(imu_norm[i:end])
            V.append(v_ms[i:end])
            GPS_EAST.append(enu_e[i:end + 1])
            GPS_NORTH.append(enu_n[i:end + 1])
            GPS_HEAD.append(gps_theta[i:end + 1])
            POS_VALID.append(pos_valid[i:end + 1])
            HEAD_VALID.append(head_valid[i:end + 1])

    if not X:
        raise RuntimeError("没有足够的 GNSS 有效窗口")

    return (
        np.stack(X).astype(np.float32),
        np.stack(V).astype(np.float32),
        np.stack(GPS_EAST).astype(np.float32),
        np.stack(GPS_NORTH).astype(np.float32),
        np.stack(GPS_HEAD).astype(np.float32),
        np.stack(POS_VALID).astype(bool),
        np.stack(HEAD_VALID).astype(bool),
    )


def prepare_dataset(X, V, ge, gn, gh, pos_v, head_v, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, V, ge, gn, gh, pos_v, head_v))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(X), 4096), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def apply_bias_and_install(imu_norm, bias_used, install_rpy, mu, std):
    std_t = tf.constant(std, dtype=tf.float32)
    mu_t = tf.constant(mu, dtype=tf.float32)

    acc_raw = imu_norm[:, :, :3] * std_t[:3] + mu_t[:3]
    gyro_raw = imu_norm[:, :, 3:6] * std_t[3:6] + mu_t[3:6]

    ba = bias_used[:, :3]
    bg = bias_used[:, 3:6]
    acc_corr = acc_raw - tf.expand_dims(ba, axis=1)
    gyro_corr = gyro_raw - tf.expand_dims(bg, axis=1)

    acc_body = rotation_imu_to_body(acc_corr, install_rpy)
    gyro_body = rotation_imu_to_body(gyro_corr, install_rpy)
    return acc_body, gyro_body[:, :, 2]


def predict_trajectory(model, imu_norm, v_ms, true_e, true_n, true_h, mu, std, use_residual=True, training=False):
    if use_residual:
        bias_used, residual, install_rpy = model(imu_norm, training=training)
    else:
        batch_size = tf.shape(imu_norm)[0]
        bias_used = model.fixed_bias_batch(batch_size)
        residual = tf.zeros_like(bias_used)
        install_rpy = model.install_rpy

    acc_body, gyro_body_z = apply_bias_and_install(imu_norm, bias_used, install_rpy, mu, std)
    pred_e, pred_n, pred_h = integrate_trajectory(
        true_e[:, 0],
        true_n[:, 0],
        true_h[:, 0],
        acc_body,
        gyro_body_z,
        v_ms,
        TARGET_DT,
    )
    return pred_e, pred_n, pred_h, bias_used, residual, install_rpy


def trajectory_loss(pred_e, pred_n, pred_h, true_e, true_n, true_h,
                    pos_valid, head_valid, residual,
                    head_weight=2.0, residual_l2_weight=50.0, base_l2=0.0):
    pos_valid_f = tf.cast(pos_valid, tf.float32)
    head_valid_f = tf.cast(head_valid, tf.float32)

    pos_err = tf.sqrt(tf.maximum((pred_e - true_e) ** 2 + (pred_n - true_n) ** 2, 1e-12))
    n_pos = tf.maximum(tf.reduce_sum(pos_valid_f, axis=1), 1.0)
    pos_rmse = tf.sqrt(tf.reduce_sum((pos_err ** 2) * pos_valid_f, axis=1) / n_pos)

    head_err = tf.abs(wrap_angle_rad(pred_h - true_h))
    n_head = tf.maximum(tf.reduce_sum(head_valid_f, axis=1), 1.0)
    head_rmse = tf.sqrt(tf.reduce_sum((head_err ** 2) * head_valid_f, axis=1) / n_head)

    residual_l2 = tf.reduce_mean(tf.square(residual))
    total = (
        tf.reduce_mean(pos_rmse)
        + head_weight * tf.reduce_mean(head_rmse)
        + residual_l2_weight * residual_l2
        + base_l2
    )
    return total, tf.reduce_mean(pos_rmse), tf.reduce_mean(head_rmse), residual_l2


def collect_metrics(model, ds, mu, std, use_residual=True):
    pos_errs, head_errs, residuals = [], [], []
    for batch in ds:
        imu, v, ge, gn, gh, pos_v, head_v = batch
        pred_e, pred_n, pred_h, _, residual, _ = predict_trajectory(
            model, imu, v, ge, gn, gh, mu, std,
            use_residual=use_residual,
            training=False,
        )
        pos_v_np = pos_v.numpy().astype(bool)
        head_v_np = head_v.numpy().astype(bool)
        pos_err = np.sqrt((pred_e.numpy() - ge.numpy()) ** 2 + (pred_n.numpy() - gn.numpy()) ** 2)
        head_err = np.abs(wrap_angle_np(pred_h.numpy() - gh.numpy())) * RAD2DEG
        pos_errs.append(pos_err[pos_v_np])
        head_errs.append(head_err[head_v_np])
        residuals.append(residual.numpy())

    pos_all = np.concatenate(pos_errs) if pos_errs else np.array([], dtype=np.float32)
    head_all = np.concatenate(head_errs) if head_errs else np.array([], dtype=np.float32)
    res_all = np.concatenate(residuals, axis=0) if residuals else np.zeros((0, 6), dtype=np.float32)

    def stats(arr):
        if len(arr) == 0:
            return {"mean": None, "rmse": None, "p95": None, "max": None}
        return {
            "mean": float(np.mean(arr)),
            "rmse": float(np.sqrt(np.mean(arr ** 2))),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr)),
        }

    return {
        "position_m": stats(pos_all),
        "heading_deg": stats(head_all),
        "residual": {
            "mean": res_all.mean(axis=0).astype(float).tolist() if len(res_all) else [0.0] * 6,
            "std": res_all.std(axis=0).astype(float).tolist() if len(res_all) else [0.0] * 6,
            "max_abs": np.max(np.abs(res_all), axis=0).astype(float).tolist() if len(res_all) else [0.0] * 6,
        },
    }


def install_gt_eval(seqs, install_rpy_deg):
    pitch_vals, yaw_vals = [], []
    for seq in seqs:
        install = seq.get("cmcc_install_deg")
        stable = seq.get("cmcc_stable")
        if install is None or stable is None or not stable.any():
            continue
        pitch_vals.append(install[stable, 0])
        yaw_vals.append(install[stable, 1])
    if not pitch_vals:
        return {}
    pitch = np.concatenate(pitch_vals)
    yaw = np.concatenate(yaw_vals)
    return {
        "cmcc_pitch_mean_deg": float(np.mean(pitch)),
        "cmcc_yaw_mean_deg": float(np.mean(yaw)),
        "pitch_error_deg": float(install_rpy_deg[1] - np.mean(pitch)),
        "yaw_error_deg": float(install_rpy_deg[2] - np.mean(yaw)),
        "roll_note": "CMCC result has rbv_pitch/rbv_yaw only; no roll GT is available.",
    }


def main():
    parser = argparse.ArgumentParser(description="Train global bias(6) + global install RPY(3) with trajectory loss.")
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--model-name", type=str, default="bias_angle_v3")
    parser.add_argument("--head-weight", type=float, default=2.0)
    parser.add_argument("--residual-l2-weight", type=float, default=50.0)
    parser.add_argument("--bias-l2-weight", type=float, default=0.1)
    parser.add_argument("--install-l2-weight", type=float, default=1.0)
    parser.add_argument("--install-prior-scale-deg", type=float, default=5.0)
    parser.add_argument("--install-limit-deg", type=float, default=10.0)
    parser.add_argument("--residual-acc-scale-g", type=float, default=0.02)
    parser.add_argument("--residual-gyro-scale-degs", type=float, default=0.02)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    print("[Info] Loading Data0109 segments...")
    seqs_tr = [s for s in load_data0109_segments(DATA0109_TRAIN_SEGMENTS) if s is not None]
    seqs_val = [s for s in load_data0109_segments([DATA0109_VAL_SEGMENT]) if s is not None]
    if not seqs_tr or not seqs_val:
        raise RuntimeError("训练或验证数据为空")

    mu, std, _ = load_or_compute_norm(seqs_tr, NORM_JSON_0109)

    print("[Info] Building windows...")
    train_data = build_windows(seqs_tr, mu, std, args.window_size)
    val_data = build_windows(seqs_val, mu, std, args.window_size)
    X_tr, V_tr, ge_tr, gn_tr, gh_tr, posv_tr, headv_tr = train_data
    X_val, V_val, ge_val, gn_val, gh_val, posv_val, headv_val = val_data
    print(f"  Train windows: {len(X_tr)}")
    print(f"  Val   windows: {len(X_val)}")

    train_ds = prepare_dataset(*train_data, batch_size=args.batch_size, shuffle=True)
    val_ds = prepare_dataset(*val_data, batch_size=args.batch_size, shuffle=False)

    print("[Info] Creating model...")
    model = BiasAngleNet(
        window_size=args.window_size,
        install_limit_deg=args.install_limit_deg,
        residual_acc_scale_g=args.residual_acc_scale_g,
        residual_gyro_scale_degs=args.residual_gyro_scale_degs,
    )
    model(tf.zeros((1, args.window_size, 7), dtype=tf.float32))
    print(f"  Output: global bias(6) + global install_rpy(3) + residual_bias(6)")
    print(f"  Total params: {model.count_params()}")
    print(f"  Install angle limit: ±{args.install_limit_deg:.1f}°")

    optimizer = tf.keras.optimizers.Adam(args.lr)
    weights_path = MODEL_DIR / f"{args.model_name}.weights.h5"

    @tf.function
    def train_step(imu, v, ge, gn, gh, pos_v, head_v):
        with tf.GradientTape() as tape:
            pred_e, pred_n, pred_h, _, residual, _ = predict_trajectory(
                model, imu, v, ge, gn, gh, mu, std,
                use_residual=True,
                training=True,
            )
            install_deg = model.install_rpy * RAD2DEG
            base_l2 = (
                args.bias_l2_weight * tf.reduce_mean(tf.square(model.bias_base))
                + args.install_l2_weight
                * tf.reduce_mean(tf.square(install_deg / args.install_prior_scale_deg))
            )
            loss, pos_l, head_l, res_l = trajectory_loss(
                pred_e, pred_n, pred_h, ge, gn, gh, pos_v, head_v, residual,
                head_weight=args.head_weight,
                residual_l2_weight=args.residual_l2_weight,
                base_l2=base_l2,
            )
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, pos_l, head_l, res_l

    @tf.function
    def val_step(imu, v, ge, gn, gh, pos_v, head_v):
        pred_e, pred_n, pred_h, _, residual, _ = predict_trajectory(
            model, imu, v, ge, gn, gh, mu, std,
            use_residual=True,
            training=False,
        )
        install_deg = model.install_rpy * RAD2DEG
        base_l2 = (
            args.bias_l2_weight * tf.reduce_mean(tf.square(model.bias_base))
            + args.install_l2_weight
            * tf.reduce_mean(tf.square(install_deg / args.install_prior_scale_deg))
        )
        return trajectory_loss(
            pred_e, pred_n, pred_h, ge, gn, gh, pos_v, head_v, residual,
            head_weight=args.head_weight,
            residual_l2_weight=args.residual_l2_weight,
            base_l2=base_l2,
        )

    print("\n[Info] Training...")
    best_val = float("inf")
    patience = 0
    history = []
    for epoch in range(args.epochs):
        sums = np.zeros(4, dtype=np.float64)
        n_train = 0
        for batch in train_ds:
            vals = train_step(*batch)
            bs = int(tf.shape(batch[0])[0].numpy())
            sums += np.array([v.numpy() for v in vals], dtype=np.float64) * bs
            n_train += bs
        train_avg = sums / max(n_train, 1)

        sums = np.zeros(4, dtype=np.float64)
        n_val = 0
        for batch in val_ds:
            vals = val_step(*batch)
            bs = int(tf.shape(batch[0])[0].numpy())
            sums += np.array([v.numpy() for v in vals], dtype=np.float64) * bs
            n_val += bs
        val_avg = sums / max(n_val, 1)

        install_deg = model.install_rpy.numpy() * RAD2DEG
        bias_base = model.bias_base.numpy()
        row = {
            "epoch": epoch + 1,
            "train_loss": float(train_avg[0]),
            "train_pos_m": float(train_avg[1]),
            "train_head_rad": float(train_avg[2]),
            "train_res_l2": float(train_avg[3]),
            "val_loss": float(val_avg[0]),
            "val_pos_m": float(val_avg[1]),
            "val_head_rad": float(val_avg[2]),
            "val_res_l2": float(val_avg[3]),
            "install_deg": install_deg.astype(float).tolist(),
            "bias_base": bias_base.astype(float).tolist(),
        }
        history.append(row)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:3d}/{args.epochs}  "
                f"train loss={train_avg[0]:.2f} pos={train_avg[1]:.2f}m head={train_avg[2] * RAD2DEG:.2f}°  |  "
                f"val loss={val_avg[0]:.2f} pos={val_avg[1]:.2f}m head={val_avg[2] * RAD2DEG:.2f}°  |  "
                f"install rpy=[{install_deg[0]:+.2f},{install_deg[1]:+.2f},{install_deg[2]:+.2f}]°"
            )

        if val_avg[0] < best_val:
            best_val = float(val_avg[0])
            patience = 0
            model.save_weights(str(weights_path))
        else:
            patience += 1
            if patience >= args.patience:
                print(f"[Info] Early stopping at epoch {epoch + 1}")
                break

    model.load_weights(str(weights_path))

    print("\n[Info] Collecting metrics...")
    train_metrics_res = collect_metrics(model, train_ds, mu, std, use_residual=True)
    val_metrics_res = collect_metrics(model, val_ds, mu, std, use_residual=True)
    train_metrics_fixed = collect_metrics(model, train_ds, mu, std, use_residual=False)
    val_metrics_fixed = collect_metrics(model, val_ds, mu, std, use_residual=False)

    bias_base = model.bias_base.numpy()
    install_deg = model.install_rpy.numpy() * RAD2DEG
    install_eval = install_gt_eval(seqs_val, install_deg)

    print("\n[Result] Best model:")
    print(f"  Best val loss: {best_val:.2f}")
    print(f"  Install RPY [deg]: roll={install_deg[0]:.4f} pitch={install_deg[1]:.4f} yaw={install_deg[2]:.4f}")
    print(f"  Acc bias [g]:      ba_x={bias_base[0]:.6f} ba_y={bias_base[1]:.6f} ba_z={bias_base[2]:.6f}")
    print(f"  Gyro bias [deg/s]: bg_x={bias_base[3]:.6f} bg_y={bias_base[4]:.6f} bg_z={bias_base[5]:.6f}")
    print(f"  Val residual RMSE: {val_metrics_res['position_m']['rmse']:.2f} m")
    print(f"  Val fixed-only RMSE: {val_metrics_fixed['position_m']['rmse']:.2f} m")
    if install_eval:
        print(f"  CMCC pitch mean: {install_eval['cmcc_pitch_mean_deg']:.4f}°, error={install_eval['pitch_error_deg']:.4f}°")
        print(f"  CMCC yaw mean:   {install_eval['cmcc_yaw_mean_deg']:.4f}°, error={install_eval['yaw_error_deg']:.4f}°")

    result = {
        "model": "bias_angle_v3",
        "weights": str(weights_path),
        "best_val_loss": best_val,
        "acc_bias_g": {"x": float(bias_base[0]), "y": float(bias_base[1]), "z": float(bias_base[2])},
        "gyro_bias_degs": {"x": float(bias_base[3]), "y": float(bias_base[4]), "z": float(bias_base[5])},
        "install_angle_deg": {
            "roll": float(install_deg[0]),
            "pitch": float(install_deg[1]),
            "yaw": float(install_deg[2]),
        },
        "install_angle_rad": {
            "roll": float(model.install_rpy.numpy()[0]),
            "pitch": float(model.install_rpy.numpy()[1]),
            "yaw": float(model.install_rpy.numpy()[2]),
        },
        "train_metrics_with_residual": train_metrics_res,
        "val_metrics_with_residual": val_metrics_res,
        "train_metrics_fixed_only": train_metrics_fixed,
        "val_metrics_fixed_only": val_metrics_fixed,
        "install_eval_cmcc": install_eval,
        "config": vars(args),
        "history": history,
    }
    result_path = MODEL_DIR / f"{args.model_name}_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {result_path}")


if __name__ == "__main__":
    main()
