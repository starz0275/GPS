"""
trajectory_data.py — 路测 / 标定数据加载（轨迹预测 / 融合验证共用）
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter

DEG2RAD = np.pi / 180.0
EARTH_A = 6378137.0
TARGET_DT = 0.1
VEH_SPD_FACTOR = 260.63
MIN_SPEED_MS = 0.5

DATA_CSV = Path(__file__).parent / "260316_Data" / "260316_Data.csv"
DATA_DIR_CALIB = Path(__file__).parent / "标定实车数据"

# 默认验证集：Data02（与 data_preprocessing_v2.CALIB_VAL_ID 一致）
VAL_DATASET_ID = "Data02"
# 260316 保留段（USE_260316 时）
VAL_T_START = 620.0


def latlon_to_enu(lat, lon, ref_lat, ref_lon):
    dlat = (lat - ref_lat) * DEG2RAD
    dlon = (lon - ref_lon) * DEG2RAD
    east = EARTH_A * dlon * np.cos(ref_lat * DEG2RAD)
    north = EARTH_A * dlat
    return east, north


def clean_gps_outliers(t, lat, lon, max_kmh=150.0):
    lat = lat.copy()
    lon = lon.copy()
    valid = np.ones(len(t), dtype=bool)
    for i in range(1, len(t)):
        dt_ = t[i] - t[i - 1]
        if dt_ <= 0:
            valid[i] = False
            continue
        dlat = (lat[i] - lat[i - 1]) * DEG2RAD * EARTH_A
        dlon = (lon[i] - lon[i - 1]) * DEG2RAD * EARTH_A * np.cos(lat[i - 1] * DEG2RAD)
        if np.sqrt(dlat ** 2 + dlon ** 2) / dt_ * 3.6 > max_kmh:
            valid[i] = False
    if valid.sum() >= 2:
        lat[~valid] = np.interp(t[~valid], t[valid], lat[valid])
        lon[~valid] = np.interp(t[~valid], t[valid], lon[valid])
    return lat, lon


def gps_valid_on_grid(t_raw, gps_ok_raw, tg, max_gap=0.15):
    """下采样网格上的 GPS 有效掩码：仅当附近有原始有效 GPS 采样时为 True。"""
    t_valid = t_raw[gps_ok_raw]
    valid = np.zeros(len(tg), dtype=bool)
    if len(t_valid) == 0:
        return valid
    for i, t in enumerate(tg):
        j = np.searchsorted(t_valid, t)
        cands = []
        if j > 0:
            cands.append(t_valid[j - 1])
        if j < len(t_valid):
            cands.append(t_valid[min(j, len(t_valid) - 1)])
        valid[i] = min(abs(t - tv) for tv in cands) <= max_gap
    return valid


def load_segment(t_start=VAL_T_START, t_end=None):
    """
    加载 260316 一段路测数据（10 Hz 网格）。

    返回 dict，含 imu_raw (T,6)、v_ms、gps_theta、gps_valid、enu 真值等。
    gps_valid 使用原始 LatitudeRaw>1，**不会**把插值空洞标为有效。
    """
    df = pd.read_csv(DATA_CSV)
    df = df[df['Time'] >= t_start].copy()
    if t_end is not None:
        df = df[df['Time'] < t_end].copy()
    df = df.reset_index(drop=True)
    if len(df) < 100:
        raise RuntimeError(f"数据不足: {len(df)} 行 (t>={t_start})")

    t_raw = df['Time'].values.astype(float)
    t_s, t_e = t_raw[0], t_raw[-1]
    tg = np.arange(t_s, t_e, TARGET_DT)

    def interp_sensor(src_v, fill=0.0):
        v = src_v.astype(float)
        mask = ~np.isnan(v) if np.isnan(v).any() else np.ones(len(v), bool)
        if mask.sum() < 2:
            return np.full(len(tg), fill, np.float32)
        return interp1d(t_raw[mask], v[mask], bounds_error=False,
                        fill_value=fill)(tg).astype(np.float32)

    imu_cols = ['AccXRaw', 'AccYRaw', 'AccZRaw', 'GyroXRaw', 'GyroYRaw', 'GyroZRaw']
    imu_raw = np.stack([interp_sensor(df[c].values) for c in imu_cols], axis=1)
    gyro_z_rad = imu_raw[:, 5] * DEG2RAD
    v_ms = interp_sensor(df['VehSpdRaw'].values / VEH_SPD_FACTOR / 3.6)

    lat_raw = df['LatitudeRaw'].values.astype(float).copy()
    lon_raw = df['LongitudeRaw'].values.astype(float).copy()
    head_raw = df['HeadingRaw'].values.astype(float).copy() if 'HeadingRaw' in df.columns else np.zeros(len(df))
    gps_ok_raw = lat_raw > 1

    lat_v = lat_raw.copy()
    lon_v = lon_raw.copy()
    lat_v[~gps_ok_raw] = np.nan
    lon_v[~gps_ok_raw] = np.nan
    head_raw[head_raw < 1] = np.nan

    vidx = np.where(gps_ok_raw)[0]
    if len(vidx) > 2:
        lv, lnv = clean_gps_outliers(t_raw[vidx], lat_v[vidx], lon_v[vidx])[:2]
        lat_v[vidx] = lv
        lon_v[vidx] = lnv

    gps_valid = gps_valid_on_grid(t_raw, gps_ok_raw, tg)

    lat_g = np.full(len(tg), np.nan, np.float32)
    lon_g = np.full(len(tg), np.nan, np.float32)
    if gps_valid.any():
        lat_g[gps_valid] = interp1d(t_raw[gps_ok_raw], lat_v[gps_ok_raw],
                                    bounds_error=False, fill_value=np.nan)(tg[gps_valid])
        lon_g[gps_valid] = interp1d(t_raw[gps_ok_raw], lon_v[gps_ok_raw],
                                    bounds_error=False, fill_value=np.nan)(tg[gps_valid])

    if gps_valid.sum() < 5:
        raise RuntimeError("GPS 有效点过少")

    ref_lat = float(np.nanmean(lat_g[gps_valid]))
    ref_lon = float(np.nanmean(lon_g[gps_valid]))
    enu_x = np.full(len(tg), np.nan, np.float32)
    enu_y = np.full(len(tg), np.nan, np.float32)
    ex, ey = latlon_to_enu(lat_g[gps_valid], lon_g[gps_valid], ref_lat, ref_lon)
    enu_x[gps_valid] = ex
    enu_y[gps_valid] = ey

    # 航向：GPS 位移 + HeadingRaw
    enu_x_fill = np.interp(np.arange(len(tg)), np.where(gps_valid)[0], ex)
    enu_y_fill = np.interp(np.arange(len(tg)), np.where(gps_valid)[0], ey)
    gps_theta = np.arctan2(np.gradient(enu_y_fill, tg), np.gradient(enu_x_fill, tg))
    gps_theta = median_filter(gps_theta.astype(np.float32), size=11)

    head_g = np.full(len(tg), np.nan, np.float32)
    if np.any(~np.isnan(head_raw)):
        head_g[gps_valid] = interp1d(
            t_raw[gps_ok_raw & ~np.isnan(head_raw)],
            head_raw[gps_ok_raw & ~np.isnan(head_raw)],
            bounds_error=False, fill_value=np.nan)(tg[gps_valid])
        has_h = ~np.isnan(head_g) & gps_valid
        if has_h.sum() > 10:
            gps_theta[has_h] = (90.0 - head_g[has_h]) * DEG2RAD

    gps_head_valid = gps_valid & (v_ms > MIN_SPEED_MS)

    return {
        'Time_s': tg,
        'imu_raw': imu_raw,
        'gyro_z_rad': gyro_z_rad,
        'v_ms': v_ms,
        'gps_theta': gps_theta.astype(np.float32),
        'gps_valid': gps_head_valid,
        'enu_x_truth': enu_x,
        'enu_y_truth': enu_y,
        'ref_lat': ref_lat,
        'ref_lon': ref_lon,
    }


def load_calibration_segment(dataset_id=VAL_DATASET_ID):
    """
    加载标定实车整段（Data01 / Data02），格式与 load_segment 相同。
    验证融合轨迹时默认使用 Data02。
    """
    from data_preprocessing_v2 import (
        load_calibration_dataset, clean_label_outliers, df_to_trajectory_seq,
        DATA_DIR_CALIB,
    )
    dfs = load_calibration_dataset(DATA_DIR_CALIB, [dataset_id])
    if not dfs:
        raise RuntimeError(f"未找到标定数据 {dataset_id}")
    df = clean_label_outliers(dfs[0])
    seq = df_to_trajectory_seq(df)
    seq['dataset_id'] = dataset_id
    return seq


def simulate_gps_loss(gps_valid, tg, loss_start_s=15.0, loss_duration_s=60.0):
    """模拟隧道：屏蔽一段 GPS。"""
    gps_v = gps_valid.copy()
    t0 = tg[0]
    i0 = int(np.searchsorted(tg - t0, loss_start_s))
    i1 = int(np.searchsorted(tg - t0, loss_start_s + loss_duration_s))
    i1 = min(i1, len(tg) - 1)
    gps_v[i0:i1] = False
    return gps_v, i0, i1
