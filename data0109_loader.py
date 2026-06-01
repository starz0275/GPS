"""
Data0109 数据加载 —— 嵌套目录 + CMCC 零偏真值
==============================================
每段数据位于 Data0109数据及零偏/<segment_name>/ 下，含 IMU/GNSS/VehicleSpeed/CMCC_result。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter

from data_preprocessing_v2 import (
    TARGET_DT,
    MIN_SPEED_MS,
    read_calibration_tab,
)

DATA0109_DIR = Path(__file__).parent / "Data0109数据及零偏"

DATA0109_TRAIN_SEGMENTS = [
    "Data01_0109_1圈8字+2圈跑道",
    "Data02_0109_2圈8字+2圈跑道",
    "Data03_0109_1圈跑道+2圈8字",
    "Data04_0109_2圈跑道+2圈8字",
]
DATA0109_VAL_SEGMENT = "Data05_0109_4圈跑道"
DATA0109_ALL_SEGMENTS = DATA0109_TRAIN_SEGMENTS + [DATA0109_VAL_SEGMENT]

CMCC_BIAS_COLS = [
    "acc_bias_x", "acc_bias_y", "acc_bias_z",
    "gyro_bias_x", "gyro_bias_y", "gyro_bias_z",
]

CMCC_ATTITUDE_COLS = ["CMCC_pitch", "CMCC_roll", "CMCC_yaw"]
CMCC_INSTALL_COLS = ["rbv_pitch", "rbv_yaw"]  # IMU安装角真值

DEG2RAD = np.pi / 180.0
EARTH_A = 6378137.0
GPS_MAX_KMH = 150.0
CMCC_CALIB_MIN_TYPE = 3
VALID_LAT_DEG = 1.0
# cmcc_ok 后至少等待该时长再纳入训练（跳过静止/CMCC 收敛初期）
CMCC_SETTLE_S = 60.0
# 旧策略：在 settle 后的帧里再取时间后段比例；None=不裁
CMCC_STABLE_TAIL_FRAC = None


def _glob_one(folder: Path, pattern: str) -> Path:
    hits = sorted(folder.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"{folder} 下未找到 {pattern}")
    return hits[0]


def resolve_data0109_paths(segment_name: str) -> Tuple[Path, Path, Path, Path]:
    """返回 (imu, vehicle_speed, gnss, cmcc_result) 路径。"""
    seg_dir = DATA0109_DIR / segment_name
    if not seg_dir.is_dir():
        raise FileNotFoundError(f"段目录不存在: {seg_dir}")
    imu_f = _glob_one(seg_dir, "*_IMU.txt")
    spd_f = _glob_one(seg_dir, "*_VehicleSpeed.txt")
    gps_f = _glob_one(seg_dir, "*_GNSS.txt")
    cmcc_f = _glob_one(seg_dir, "*_CMCC_result.txt")
    return imu_f, spd_f, gps_f, cmcc_f


def clean_gps_outliers(t, lat, lon, max_kmh=GPS_MAX_KMH):
    lat = lat.copy()
    lon = lon.copy()
    valid = np.ones(len(t), dtype=bool)
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if dt <= 0:
            valid[i] = False
            continue
        dlat = (lat[i] - lat[i - 1]) * DEG2RAD * EARTH_A
        dlon = (lon[i] - lon[i - 1]) * DEG2RAD * EARTH_A * np.cos(lat[i - 1] * DEG2RAD)
        spd = np.sqrt(dlat ** 2 + dlon ** 2) / dt * 3.6
        if spd > max_kmh:
            valid[i] = False
    if valid.sum() >= 2:
        lat[~valid] = np.interp(t[~valid], t[valid], lat[valid])
        lon[~valid] = np.interp(t[~valid], t[valid], lon[valid])
    return lat, lon, valid


def wgs84_to_enu_simple(lat_arr, lon_arr, ref_lat, ref_lon):
    dlat = (lat_arr - ref_lat) * DEG2RAD
    dlon = (lon_arr - ref_lon) * DEG2RAD
    east = EARTH_A * dlon * np.cos(ref_lat * DEG2RAD)
    north = EARTH_A * dlat
    return east, north


def cmcc_ok_mask(calibration_type: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """稳定标定段：calibration_type>=3 且纬度有效。"""
    return (calibration_type >= CMCC_CALIB_MIN_TYPE) & (lat_deg > VALID_LAT_DEG)


def cmcc_settle_mask(
    cmcc_ok: np.ndarray,
    settle_s: float = CMCC_SETTLE_S,
    target_dt: float = TARGET_DT,
) -> np.ndarray:
    """
    cmcc_ok 为真后，每段连续区间内再等待 settle_s 秒才标记为 True。
    用于排除起步静止、CMCC 零偏尚未阶跃到稳态的时段。
    """
    stable = np.zeros_like(cmcc_ok, dtype=bool)
    if settle_s <= 0:
        return cmcc_ok.copy()
    n_settle = max(1, int(round(settle_s / target_dt)))
    T = len(cmcc_ok)
    i = 0
    while i < T:
        if not cmcc_ok[i]:
            i += 1
            continue
        j = i
        while j < T and cmcc_ok[j]:
            j += 1
        start = i + n_settle
        if start < j:
            stable[start:j] = True
        i = j
    return stable


def cmcc_stable_mask(
    cmcc_ok: np.ndarray,
    settle_s: float = CMCC_SETTLE_S,
    target_dt: float = TARGET_DT,
    tail_frac: float | None = CMCC_STABLE_TAIL_FRAC,
) -> np.ndarray:
    """
    训练/评估用稳定掩码：先 cmcc_settle_mask，再可选 tail_frac 裁后段。
    """
    stable = cmcc_settle_mask(cmcc_ok, settle_s=settle_s, target_dt=target_dt)
    if tail_frac is None or tail_frac >= 1.0:
        return stable
    idx = np.where(stable)[0]
    if len(idx) < 10:
        return stable
    cut_pos = int(len(idx) * (1.0 - tail_frac))
    cut_idx = idx[cut_pos]
    out = np.zeros_like(stable, dtype=bool)
    out[cut_idx:] = stable[cut_idx:]
    return out


def load_data0109_seq(
    segment_name: str,
    target_dt: float = TARGET_DT,
) -> Optional[dict]:
    """
    加载单段 Data0109，10 Hz 对齐。

    返回 dict（与 load_calibration_seq 兼容），额外字段：
      cmcc_bias_6d (T,6), cmcc_ok (T,), cmcc_stable (T,)（ok+settle_s）, calibration_type (T,)
    """
    try:
        imu_f, spd_f, gps_f, cmcc_f = resolve_data0109_paths(segment_name)
    except FileNotFoundError as e:
        print(f"  [{segment_name}] {e}")
        return None

    try:
        imu = read_calibration_tab(imu_f)
        spd = read_calibration_tab(spd_f)
        gps = read_calibration_tab(gps_f)
        cmcc = read_calibration_tab(cmcc_f)

        t_s = max(
            imu["Time_s"].min(),
            spd["Time_s"].min(),
            gps["Time_s"].min(),
            cmcc["Time_s"].min(),
        )
        t_e = min(
            imu["Time_s"].max(),
            spd["Time_s"].max(),
            gps["Time_s"].max(),
            cmcc["Time_s"].max(),
        )
        tg = np.arange(t_s, t_e, target_dt)

        def interp1(src_t, src_v):
            return interp1d(
                src_t, src_v, bounds_error=False, fill_value="extrapolate"
            )(tg)

        imu_cols = [
            "AccX_g", "AccY_g", "AccZ_g",
            "GyroX_degs", "GyroY_degs", "GyroZ_degs",
        ]
        imu_mat = np.stack(
            [interp1(imu["Time_s"].values, imu[c].values) for c in imu_cols],
            axis=1,
        ).astype(np.float32)
        v_kmh = interp1(spd["Time_s"].values, spd["VehicleSpeed_kmh"].values)
        v_ms = (v_kmh / 3.6).astype(np.float32)
        imu_mat = np.concatenate([imu_mat, v_ms.reshape(-1, 1)], axis=1)

        lat_raw = interp1(gps["Time_s"].values, gps["Latitude_deg"].values)
        lon_raw = interp1(gps["Time_s"].values, gps["Longitude_deg"].values)
        lat_c, lon_c, gps_ok = clean_gps_outliers(tg, lat_raw, lon_raw)

        ref_lat, ref_lon = lat_c[gps_ok][0], lon_c[gps_ok][0]
        enu_x, enu_y = wgs84_to_enu_simple(lat_c, lon_c, ref_lat, ref_lon)

        dx = np.gradient(enu_x, tg)
        dy = np.gradient(enu_y, tg)
        speed_gps = np.sqrt(dx ** 2 + dy ** 2)
        gps_head_valid = gps_ok & (v_ms > MIN_SPEED_MS) & (speed_gps > 0.1)
        gps_theta = np.arctan2(dy, dx)
        gps_theta_smooth = median_filter(gps_theta, size=11)

        cal_type = interp1(
            cmcc["Time_s"].values,
            cmcc["calibration_type"].values.astype(np.float32),
        ).astype(np.int32)
        lat_cmcc = interp1(cmcc["Time_s"].values, cmcc["Latitude_deg"].values)
        cmcc_bias = np.stack(
            [interp1(cmcc["Time_s"].values, cmcc[c].values) for c in CMCC_BIAS_COLS],
            axis=1,
        ).astype(np.float32)
        cmcc_attitude_deg = np.stack(
            [interp1(cmcc["Time_s"].values, cmcc[c].values) for c in CMCC_ATTITUDE_COLS],
            axis=1,
        ).astype(np.float32)  # (T, 3): pitch, roll, yaw [deg]
        cmcc_install_deg = np.stack(
            [interp1(cmcc["Time_s"].values, cmcc[c].values) for c in CMCC_INSTALL_COLS],
            axis=1,
        ).astype(np.float32)  # (T, 2): rbv_pitch, rbv_yaw [deg]
        ok = cmcc_ok_mask(cal_type, lat_cmcc)
        stable = cmcc_stable_mask(ok)

        short_id = segment_name.split("_")[0]
        seq = {
            "id": short_id,
            "segment": segment_name,
            "Time_s": tg,
            "imu": imu_mat,
            "gyro_z_rad": (imu_mat[:, 5] * DEG2RAD).astype(np.float32),
            "v_ms": v_ms,
            "gps_theta": gps_theta_smooth.astype(np.float32),
            "gps_valid": gps_head_valid,
            "enu_x": enu_x.astype(np.float32),
            "enu_y": enu_y.astype(np.float32),
            "cmcc_bias_6d": cmcc_bias,
            "cmcc_attitude_deg": cmcc_attitude_deg,
            "cmcc_install_deg": cmcc_install_deg,  # (T, 2): rbv_pitch, rbv_yaw [deg]
            "cmcc_ok": ok,
            "cmcc_stable": stable,
            "calibration_type": cal_type,
        }
        print(
            f"  [{segment_name}] {len(tg)} 帧  "
            f"cmcc_ok={ok.mean():.1%}  cmcc_stable(settle={CMCC_SETTLE_S:.0f}s)={stable.mean():.1%}  "
            f"GPS航向有效={gps_head_valid.mean():.1%}"
        )
        return seq
    except Exception as e:
        print(f"  [{segment_name}] 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_data0109_segments(segment_names: List[str]) -> List[dict]:
    seqs = []
    for name in segment_names:
        seq = load_data0109_seq(name)
        if seq is not None:
            seqs.append(seq)
    return seqs
