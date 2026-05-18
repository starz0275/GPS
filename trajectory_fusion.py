"""
trajectory_fusion.py — EKF + TCN 融合轨迹预测

推理策略（企业车载导航常见做法）：
  - GNSS 有效：位置/航向跟 GNSS（轨迹贴在绿线上）
  - GNSS 丢失：BiasNet 航向 + TCN 逐步位移残差，从上一可靠点外推
  - 恢复 GNSS：下一帧位置重新锚定到 GNSS
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional

from ekf_navigator import EKFNavigatorNP, load_norm_stats

EPS = 1e-8
IMU_KEYS = ['AccX_g', 'AccY_g', 'AccZ_g', 'GyroX_degs', 'GyroY_degs', 'GyroZ_degs']
FEAT_KEYS = IMU_KEYS + ['VehicleSpeed_ms', 'Base_dx', 'Base_dy']


def _norm(x, stats, key):
    s = stats.get(key, {'mean': 0.0, 'std': 1.0})
    return (x - s['mean']) / (s['std'] + EPS)


def _fill_missing_stats(stats: dict, v_ms, base_dx, base_dy):
    out = dict(stats)
    for key, arr in [
        ('VehicleSpeed_ms', v_ms),
        ('Base_dx', base_dx),
        ('Base_dy', base_dy),
    ]:
        if key not in out:
            out[key] = {'mean': float(np.mean(arr)), 'std': float(np.std(arr) + EPS)}
    return out


def build_feature_window(imu_raw, v_ms, base_dx, base_dy, k, window_size, stats):
    i0 = k - window_size + 1
    feat = np.zeros((window_size, len(FEAT_KEYS)), dtype=np.float32)
    for j, t in enumerate(range(i0, k + 1)):
        for c, key in enumerate(IMU_KEYS):
            feat[j, c] = _norm(float(imu_raw[t, c]), stats, key)
        feat[j, 6] = _norm(float(v_ms[t]), stats, 'VehicleSpeed_ms')
        feat[j, 7] = _norm(float(base_dx[t]), stats, 'Base_dx')
        feat[j, 8] = _norm(float(base_dy[t]), stats, 'Base_dy')
    return feat


def compute_hybrid_heading(gyro_z, gps_theta, gps_valid_nav, net_bias, dt):
    """
    GPS 有效 → 采用 GNSS 航向；丢失 → 陀螺减零偏积分。
    与 data_preprocessing_v2 中 compute_heading_with_gps_anchor 逻辑一致。
    """
    T = len(gyro_z)
    h = np.zeros(T, np.float32)
    ok = gps_valid_nav & np.isfinite(gps_theta)
    if not ok.any():
        return h
    first = int(np.where(ok)[0][0])
    h[first] = float(gps_theta[first])
    cur = h[first]
    for k in range(first + 1, T):
        if gps_valid_nav[k] and np.isfinite(gps_theta[k]):
            cur = float(gps_theta[k])
        else:
            g = float(gyro_z[k]) if np.isfinite(gyro_z[k]) else 0.0
            b = float(net_bias[k]) if np.isfinite(net_bias[k]) else 0.0
            cur = cur + (g - b) * dt
        h[k] = cur
    h[:first] = h[first]
    return h


def run_pure_dr(gyro_z, v_ms, gps_theta, gps_valid, dt):
    T = len(v_ms)
    h = np.zeros(T, np.float32)
    ok = gps_valid & np.isfinite(gps_theta)
    first = int(np.where(ok)[0][0]) if ok.any() else 0
    h[first] = float(gps_theta[first])
    dx = np.zeros(T, np.float32)
    dy = np.zeros(T, np.float32)
    cur = h[first]
    for k in range(first + 1, T):
        cur = cur + float(gyro_z[k]) * dt
        if gps_valid[k] and np.isfinite(gps_theta[k]):
            cur = float(gps_theta[k])
        h[k] = cur
        dx[k] = float(v_ms[k]) * np.cos(cur) * dt
        dy[k] = float(v_ms[k]) * np.sin(cur) * dt
    h[:first] = h[first]
    return np.cumsum(dx), np.cumsum(dy), h


def base_from_heading(headings, v_ms, dt):
    return (v_ms * np.cos(headings) * dt).astype(np.float32), \
           (v_ms * np.sin(headings) * dt).astype(np.float32)


def _anchor_to_first_gnss(pred_x, pred_y, gnss_x, gnss_y, gps_valid):
    """将积分轨迹平移到与首个有效 GNSS 位置一致（同一 ENU 坐标系）。"""
    x, y = pred_x.copy(), pred_y.copy()
    ok = np.where(gps_valid & np.isfinite(gnss_x) & np.isfinite(gnss_y))[0]
    if len(ok) == 0:
        return x, y
    k0 = int(ok[0])
    x += float(gnss_x[k0]) - x[k0]
    y += float(gnss_y[k0]) - y[k0]
    return x, y


def integrate_with_gnss_segments(pred_dx, pred_dy, gnss_x, gnss_y, gps_valid):
    """
    分段轨迹：有 GNSS 用实测位置；无 GNSS 用模型步长积分。
    保证有信号段与绿线重合，无信号段才体现预测能力。
    """
    T = len(pred_dx)
    x = np.full(T, np.nan, np.float32)
    y = np.full(T, np.nan, np.float32)
    ok = np.where(gps_valid & np.isfinite(gnss_x) & np.isfinite(gnss_y))[0]
    if len(ok) == 0:
        return np.cumsum(pred_dx).astype(np.float32), np.cumsum(pred_dy).astype(np.float32)

    k0 = int(ok[0])
    x[k0] = float(gnss_x[k0])
    y[k0] = float(gnss_y[k0])

    for k in range(k0 + 1, T):
        if gps_valid[k] and np.isfinite(gnss_x[k]) and np.isfinite(gnss_y[k]):
            x[k] = float(gnss_x[k])
            y[k] = float(gnss_y[k])
        else:
            x[k] = x[k - 1] + float(pred_dx[k])
            y[k] = y[k - 1] + float(pred_dy[k])

    for k in range(k0 - 1, -1, -1):
        if gps_valid[k] and np.isfinite(gnss_x[k]) and np.isfinite(gnss_y[k]):
            x[k] = float(gnss_x[k])
            y[k] = float(gnss_y[k])
        elif k + 1 < T and np.isfinite(x[k + 1]):
            x[k] = x[k + 1] - float(pred_dx[k + 1])
            y[k] = y[k + 1] - float(pred_dy[k + 1])

    bad = ~np.isfinite(x)
    if bad.any():
        x[bad] = np.interp(np.where(bad)[0], np.where(~bad)[0], x[~bad])
        y[bad] = np.interp(np.where(bad)[0], np.where(~bad)[0], y[~bad])
    return x, y


def gnss_relock_position(pred_x, pred_y, gnss_x, gnss_y, gps_valid):
    """GPS 由丢失恢复时，用当前 GNSS 位置平移后续轨迹（实车常见做法）。"""
    x = pred_x.copy()
    y = pred_y.copy()
    T = len(x)
    in_outage = False
    for k in range(T):
        if not gps_valid[k]:
            in_outage = True
            continue
        if in_outage and np.isfinite(gnss_x[k]) and np.isfinite(gnss_y[k]):
            x[k:] += float(gnss_x[k]) - x[k]
            y[k:] += float(gnss_y[k]) - y[k]
            in_outage = False
    return x, y


class FusedTrajectoryPredictor:
    """混合航向 + TCN 残差 → 轨迹。"""

    RESIDUAL_CLIP = 2.0   # 单步残差限幅 (m)，抑制 TCN 野值

    def __init__(self,
                 biasnet_weights: Path | str,
                 tcn_model_path: Path | str,
                 norm_stats_path: Path | str,
                 window_size: int = 30,
                 dt: float = 0.1):
        self.window_size = window_size
        self.dt = dt
        self.norm_stats = load_norm_stats(str(norm_stats_path))
        self.ekf_nav = EKFNavigatorNP(
            str(biasnet_weights), self.norm_stats, window_size=window_size)
        self.tcn = self._load_tcn(tcn_model_path)

    @staticmethod
    def _load_tcn(path):
        import keras
        import tensorflow as tf
        keras.config.enable_unsafe_deserialization()
        p = Path(path)
        if not p.exists():
            return None
        return tf.keras.models.load_model(p, safe_mode=False, compile=False)

    def predict(self, seq: dict, gps_valid_nav: Optional[np.ndarray] = None,
                use_gnss_position: bool = True) -> dict:
        imu = seq['imu_raw']
        v_ms = seq['v_ms']
        gyro_z = seq['gyro_z_rad']
        gps_th = seq['gps_theta']
        gps_v = gps_valid_nav if gps_valid_nav is not None else seq['gps_valid']
        dt = self.dt
        T = len(v_ms)
        W = self.window_size

        dr_x, dr_y, _ = run_pure_dr(gyro_z, v_ms, gps_th, gps_v, dt)

        ekf_x, ekf_y, ekf_h, net_bias = self.ekf_nav.run(
            imu, v_ms, gyro_z, gps_th, gps_v, dt)
        base_dx, base_dy = base_from_heading(ekf_h, v_ms, dt)

        stats = _fill_missing_stats(self.norm_stats, v_ms, base_dx, base_dy)
        pred_dx = base_dx.copy()
        pred_dy = base_dy.copy()

        # 仅在 GPS 丢失段叠加 TCN 残差（有 GNSS 时保持 EKF 底盘，贴近真值）
        if self.tcn is not None and T >= W:
            xs, idxs = [], []
            for k in range(W - 1, T):
                if gps_v[k]:
                    continue
                xs.append(build_feature_window(
                    imu, v_ms, base_dx, base_dy, k, W, stats))
                idxs.append(k)
            if xs:
                res = self.tcn.predict(np.stack(xs), batch_size=64, verbose=0)
                res = np.clip(res, -self.RESIDUAL_CLIP, self.RESIDUAL_CLIP)
                for j, k in enumerate(idxs):
                    pred_dx[k] = base_dx[k] + float(res[j, 0])
                    pred_dy[k] = base_dy[k] + float(res[j, 1])

        gx = seq.get('enu_x_truth')
        gy = seq.get('enu_y_truth')

        # 纯积分轨迹（仅评估无 GNSS 段漂移，不用于有信号段展示）
        pure_x = np.cumsum(pred_dx).astype(np.float32)
        pure_y = np.cumsum(pred_dy).astype(np.float32)
        if gx is not None and gy is not None:
            pure_x, pure_y = _anchor_to_first_gnss(pure_x, pure_y, gx, gy, gps_v)

        if use_gnss_position and gx is not None and gy is not None:
            hybrid_x, hybrid_y = integrate_with_gnss_segments(
                pred_dx, pred_dy, gx, gy, gps_v)
        else:
            hybrid_x, hybrid_y = pure_x, pure_y

        return {
            'dr_x': dr_x, 'dr_y': dr_y,
            'ekf_x': ekf_x, 'ekf_y': ekf_y, 'ekf_h': ekf_h,
            'fused_x': hybrid_x, 'fused_y': hybrid_y,
            'pure_x': pure_x, 'pure_y': pure_y,
            'hybrid_h': ekf_h,
            'base_dx': base_dx, 'base_dy': base_dy,
        }
