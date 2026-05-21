"""
train_ekf.py — BiasNet 陀螺零偏预测网络训练脚本
================================================

训练逻辑
--------
1. 从标定实车数据训练段加载数据
2. 计算每时刻的"真实"陀螺 Z 零偏：
     b_k = ω_z_measured(rad/s) − ω_heading_GPS(rad/s)
   其中 ω_heading_GPS = Δθ_GPS / dt（GPS 有效段推算）
3. 对 b_k 做中值平滑（去除 GPS 航向噪声）得到标签 b_smooth
4. 构造 IMU 窗口 (W, 6) → 预测 b_smooth（监督回归）
5. 保存最优权重到 trained_models/biasnet_weights

依赖数据文件（由 data_preprocessing_v2.py 生成）
  preprocessed_data/normalization_stats.json
  以及原始数据目录（脚本内自行读取，无需 aligned_data.csv）
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import (ModelCheckpoint, ReduceLROnPlateau,
                                        EarlyStopping)
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
import json
import warnings
warnings.filterwarnings('ignore')

from ekf_navigator import BiasNet
from config import DEFAULT_EKF_CONFIG

# ============================================================================
# 配置
# ============================================================================

from data_preprocessing_v2 import (
    DATA_DIR_CALIB,
    CALIB_TRAIN_IDS,
    CALIB_VAL_ID,
    resolve_calibration_paths,
)

NORM_JSON       = Path(__file__).parent / "preprocessed_data" / "normalization_stats.json"
MODEL_DIR       = Path(__file__).parent / "trained_models"
MODEL_DIR.mkdir(exist_ok=True)
WEIGHTS_PATH    = MODEL_DIR / "biasnet_weights.weights.h5"

WINDOW_SIZE     = 30          # 帧 (3 s @ 10 Hz)
TARGET_DT       = 0.1         # s
DEG2RAD         = np.pi / 180.0
RAD2DEG         = 180.0 / np.pi
EARTH_A         = 6378137.0
EARTH_E2        = 0.00669437999014132
MIN_SPEED_MS    = 0.5
GPS_MAX_KMH     = 150.0
VEH_SPD_FACTOR  = 260.63

BATCH_SIZE      = 256
EPOCHS          = 80
LR              = 3e-4
PATIENCE        = 15
WINDOW_S_INTEG  = 3.0         # 积分法零偏窗口（秒），替代原 BIAS_SMOOTH_W



# ============================================================================
# 坐标工具
# ============================================================================

def wgs84_to_enu_simple(lat_arr, lon_arr, ref_lat, ref_lon):
    """球面近似，仅用于训练数据（精度足够）"""
    dlat = (lat_arr - ref_lat) * DEG2RAD
    dlon = (lon_arr - ref_lon) * DEG2RAD
    R = EARTH_A
    east  = R * dlon * np.cos(ref_lat * DEG2RAD)
    north = R * dlat
    return east, north


def clean_gps_outliers(t, lat, lon, max_kmh=GPS_MAX_KMH):
    lat = lat.copy(); lon = lon.copy()
    valid = np.ones(len(t), dtype=bool)
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if dt <= 0:
            valid[i] = False; continue
        dlat = (lat[i] - lat[i-1]) * DEG2RAD * EARTH_A
        dlon = (lon[i] - lon[i-1]) * DEG2RAD * EARTH_A * np.cos(lat[i-1]*DEG2RAD)
        spd = np.sqrt(dlat**2 + dlon**2) / dt * 3.6
        if spd > max_kmh:
            valid[i] = False
    if valid.sum() >= 2:
        lat[~valid] = np.interp(t[~valid], t[valid], lat[valid])
        lon[~valid] = np.interp(t[~valid], t[valid], lon[valid])
    return lat, lon, valid


# ============================================================================
# 数据加载：标定实车数据
# ============================================================================

def load_calibration_seq(data_dir, dataset_ids):
    """
    加载标定实车各段，返回 list of dict (每段一个 dict)
    dict 键：Time_s, imu(T,6), gyro_z_rad(T,), v_ms(T,), gps_theta(T,), gps_valid(T,)
    """
    seqs = []
    data_dir = Path(data_dir)
    for ds_id in dataset_ids:
        imu_f, spd_f, gps_f = resolve_calibration_paths(data_dir, ds_id)
        if not imu_f.exists():
            continue
        try:
            def rtab(p):
                df = pd.read_csv(p, sep='\t', skipinitialspace=True, encoding='utf-8-sig')
                df.columns = [c.replace('ï»¿', '').strip() for c in df.columns]
                return df

            imu = rtab(imu_f); spd = rtab(spd_f); gps = rtab(gps_f)
            t_s = max(imu['Time_s'].min(), spd['Time_s'].min(), gps['Time_s'].min())
            t_e = min(imu['Time_s'].max(), spd['Time_s'].max(), gps['Time_s'].max())
            tg = np.arange(t_s, t_e, TARGET_DT)

            def interp1(src_t, src_v):
                return interp1d(src_t, src_v, bounds_error=False,
                                fill_value='extrapolate')(tg)

            imu_cols = ['AccX_g','AccY_g','AccZ_g','GyroX_degs','GyroY_degs','GyroZ_degs']
            imu_mat = np.stack([interp1(imu['Time_s'].values, imu[c].values)
                                for c in imu_cols], axis=1).astype(np.float32)
            v_kmh = interp1(spd['Time_s'].values, spd['VehicleSpeed_kmh'].values)
            v_ms  = v_kmh / 3.6
            # 将轮速拼入 imu 矩阵（第7通道），让 BiasNet 能区分转弯 vs 零偏
            imu_mat = np.concatenate([
                imu_mat, v_ms.astype(np.float32).reshape(-1, 1)], axis=1)  # (T, 7)

            lat_raw = interp1(gps['Time_s'].values, gps['Latitude_deg'].values)
            lon_raw = interp1(gps['Time_s'].values, gps['Longitude_deg'].values)
            lat_c, lon_c, gps_ok = clean_gps_outliers(tg, lat_raw, lon_raw)

            # ENU（球面近似）
            ref_lat, ref_lon = lat_c[gps_ok][0], lon_c[gps_ok][0]
            enu_x, enu_y = wgs84_to_enu_simple(lat_c, lon_c, ref_lat, ref_lon)

            # GPS 航向 = atan2(Δy, Δx)，仅速度 > MIN_SPEED_MS 且 GPS 有效时可信
            dx = np.gradient(enu_x, tg)
            dy = np.gradient(enu_y, tg)
            speed_gps = np.sqrt(dx**2 + dy**2)
            gps_head_valid = gps_ok & (v_ms > MIN_SPEED_MS) & (speed_gps > 0.1)
            gps_theta = np.arctan2(dy, dx)           # ENU 弧度

            # 中值平滑 GPS 航向（去除噪声）
            gps_theta_smooth = median_filter(gps_theta, size=11)

            seqs.append({
                'id'           : ds_id,
                'Time_s'       : tg,
                'imu'          : imu_mat,            # (T, 6)
                'gyro_z_rad'   : imu_mat[:, 5] * DEG2RAD,
                'v_ms'         : v_ms.astype(np.float32),
                'gps_theta'    : gps_theta_smooth.astype(np.float32),
                'gps_valid'    : gps_head_valid,
                'enu_x'        : enu_x.astype(np.float32),
                'enu_y'        : enu_y.astype(np.float32),
            })
            print(f"  [{ds_id}] {len(tg)} 帧  GPS有效率={gps_head_valid.mean():.1%}")
        except Exception as e:
            print(f"  [{ds_id}] 加载失败: {e}")
            import traceback; traceback.print_exc()
    return seqs


# ============================================================================
# 数据加载：260316 训练段（前 70%）
# ============================================================================

def load_real_seq(csv_path, t_max=None):
    df = pd.read_csv(csv_path)
    if t_max is not None:
        df = df[df['Time'] < t_max].copy().reset_index(drop=True)
    if len(df) < 100:
        return []

    t_raw = df['Time'].values
    t_s, t_e = t_raw[0], t_raw[-1]
    tg = np.arange(t_s, t_e, TARGET_DT)

    def interp1(src_v, fill=0.0):
        mask = ~np.isnan(src_v.astype(float)) if np.isnan(src_v.astype(float)).any() \
               else np.ones(len(src_v), bool)
        if mask.sum() < 2:
            return np.full(len(tg), fill, dtype=np.float32)
        return interp1d(t_raw[mask], src_v[mask], bounds_error=False,
                        fill_value=fill)(tg).astype(np.float32)

    imu_cols_raw = ['AccXRaw','AccYRaw','AccZRaw','GyroXRaw','GyroYRaw','GyroZRaw']
    imu_mat = np.stack([interp1(df[c].values) for c in imu_cols_raw], axis=1)
    v_ms = interp1(df['VehSpdRaw'].values / VEH_SPD_FACTOR / 3.6)

    lat_raw = df['LatitudeRaw'].values.copy().astype(float)
    lon_raw = df['LongitudeRaw'].values.copy().astype(float)
    gps_ok_raw = (lat_raw > 1)
    lat_raw[~gps_ok_raw] = np.nan
    lon_raw[~gps_ok_raw] = np.nan

    valid_idx = np.where(gps_ok_raw)[0]
    if len(valid_idx) > 2:
        lv, lnv, ok2 = clean_gps_outliers(
            t_raw[valid_idx], lat_raw[valid_idx], lon_raw[valid_idx])
        lat_raw[valid_idx] = lv; lon_raw[valid_idx] = lnv

    lat_g = interp1(lat_raw, fill=np.nan)
    lon_g = interp1(lon_raw, fill=np.nan)
    gps_valid_g = ~np.isnan(lat_g)

    if gps_valid_g.sum() < 10:
        return []

    ref_lat = lat_g[gps_valid_g][0]; ref_lon = lon_g[gps_valid_g][0]
    enu_x_v, enu_y_v = wgs84_to_enu_simple(
        lat_g[gps_valid_g], lon_g[gps_valid_g], ref_lat, ref_lon)
    enu_x = np.zeros(len(tg), np.float32)
    enu_y = np.zeros(len(tg), np.float32)
    enu_x[gps_valid_g] = enu_x_v; enu_y[gps_valid_g] = enu_y_v

    dx = np.gradient(enu_x, tg)
    dy = np.gradient(enu_y, tg)
    speed_gps = np.sqrt(dx**2 + dy**2)
    head_valid = gps_valid_g & (v_ms > MIN_SPEED_MS) & (speed_gps > 0.1)
    gps_theta  = np.arctan2(dy, dx)
    gps_theta_smooth = median_filter(gps_theta, size=11).astype(np.float32)

    if 'HeadingRaw' in df.columns:
        head_raw_v = df['HeadingRaw'].values.copy().astype(float)
        head_raw_v[head_raw_v < 1] = np.nan
        head_g = interp1(head_raw_v, fill=np.nan)
        has_raw = ~np.isnan(head_g) & head_valid
        if has_raw.sum() > 10:
            enu_head = (90.0 - head_g) * DEG2RAD
            gps_theta_smooth[has_raw] = enu_head[has_raw].astype(np.float32)

    print(f"  [260316 train] {len(tg)} 帧  GPS航向有效率={head_valid.mean():.1%}")

    return [{
        'id'         : '260316_train',
        'Time_s'     : tg,
        'imu'        : imu_mat,
        'gyro_z_rad' : imu_mat[:, 5] * DEG2RAD,
        'v_ms'       : v_ms,
        'gps_theta'  : gps_theta_smooth,
        'gps_valid'  : head_valid,
        'enu_x'      : enu_x,
        'enu_y'      : enu_y,
    }]


# ============================================================================
# 计算零偏标签
# ============================================================================

def compute_bias_labels(seq, window_s=3.0):
    """
    用积分法计算零偏，配合速度 + 直行硬约束剔除噪声标签。

    约束条件（三个掩码取 &）:
      1. gps_valid  — GPS 信号有效（来自原始配置）
      2. is_moving  — 轮速 > MIN_SPEED_MS（静止时 GPS 航向无规律跳变）
      3. is_straight — |陀螺 Z| < 8 °/s（放宽阈值以包含转弯数据，避免 BiasNet 未见转弯）

    仅当窗口两端同时满足上述约束时，才用 3 秒积分窗计算零偏：
      bias = (Σ(gyro_z)*dt - Δθ_gps) / (N*dt)

    无效段通过线性插值填补，保证输出序列完整。
    返回 (T,) [rad/s] 及可信掩码 label_ok。
    """
    T = len(seq['Time_s'])
    dt = float(np.median(np.diff(seq['Time_s'])))
    N = max(1, int(window_s / dt))

    gyro_z = seq['gyro_z_rad']                     # (T,) [rad/s]
    gps_th = seq['gps_theta']                      # (T,) [rad]
    gps_ok = seq['gps_valid']                      # (T,) bool — GPS 信号有效
    v_ms   = seq['v_ms']                           # (T,) [m/s]

    # ---- 三合一严格掩码 ----
    is_moving   = v_ms >= MIN_SPEED_MS                              # > 0.5 m/s
    is_straight = np.abs(gyro_z) < 8.0 * DEG2RAD                    # |ω| < 8 °/s（放宽→包含转弯）
    strict_ok   = gps_ok & is_moving & is_straight                   # 严格有效帧

    # ---- 滑动窗口积分计算零偏 ----
    bias_labels = np.zeros(T, dtype=np.float32)
    gyro_cum = np.cumsum(gyro_z) * dt
    for i in range(N, T):
        if strict_ok[i] and strict_ok[i - N]:        # 窗口两端均满足严格条件
            dth_gps = gps_th[i] - gps_th[i - N]
            dth_gps = np.arctan2(np.sin(dth_gps), np.cos(dth_gps))  # wrap
            dth_gyro = gyro_cum[i] - gyro_cum[i - N]
            bias_labels[i] = (dth_gyro - dth_gps) / (N * dt)

    # ---- 有效标签掩码（与 bias_labels 对齐：前 N 帧无标签）----
    label_ok = np.zeros(T, dtype=bool)
    if N < T:
        label_ok[N:] = strict_ok[N:] & strict_ok[:-N]

    # ---- 后处理 ----
    n_valid = label_ok.sum()
    if n_valid < 2:
        # 有效点太少无法做 median_filter + interp，直接返回 0
        return bias_labels.astype(np.float32), label_ok

    # 中值平滑（仅有效段，size 自适应防越界）
    smooth_size = min(11, max(3, n_valid // 20))
    bias_labels[label_ok] = median_filter(bias_labels[label_ok], size=smooth_size)

    # 线性插值填补无效段（保证输出连续）
    idx = np.arange(T)
    bias_labels[:] = np.interp(idx, idx[label_ok], bias_labels[label_ok])

    return bias_labels.astype(np.float32), label_ok


# ============================================================================
# 归一化
# ============================================================================

def load_or_compute_norm(seqs, norm_json_path):
    """
    优先从已有 normalization_stats.json 加载，否则从数据计算并保存。
    """
    keys = ['AccX_g','AccY_g','AccZ_g','GyroX_degs','GyroY_degs','GyroZ_degs','VehicleSpeed_ms']
    if norm_json_path.exists():
        with open(norm_json_path) as f:
            raw = json.load(f)
        # 兼容两种格式
        stats = raw.get('stats', raw)
        print(f"  [Norm] 加载已有统计量：{norm_json_path.name}")
        mu  = np.array([stats[k]['mean'] for k in keys], np.float32)
        std = np.array([stats[k]['std']  for k in keys], np.float32)
    else:
        all_imu = np.concatenate([s['imu'] for s in seqs], axis=0)
        mu  = all_imu.mean(0).astype(np.float32)
        std = all_imu.std(0).astype(np.float32) + 1e-8
        stats = {k: {'mean': float(mu[i]), 'std': float(std[i])}
                 for i, k in enumerate(keys)}
        norm_json_path.parent.mkdir(exist_ok=True)
        with open(norm_json_path, 'w') as f:
            json.dump({'stats': stats}, f, indent=2)
        print(f"  [Norm] 重新计算并保存：{norm_json_path.name}")

    return mu, std, stats


def normalize_imu(imu, mu, std):
    return (imu - mu) / (std + 1e-8)


# ============================================================================
# 构造训练样本
# ============================================================================

def build_samples(seqs, mu, std, window_size=WINDOW_SIZE, tunnel_aug=True):
    """
    返回 X (N, W, 6) 和 Y (N, 1) [rad/s]

    当 tunnel_aug=True 时，对每段数据模拟一段 GPS 丢失，生成额外的训练窗口，
    让 BiasNet 学会在 GPS 丢失场景下的零偏预测。
    """
    X_list, Y_list = [], []

    def _windows_from_seq(seq, label=''):
        imu_norm = normalize_imu(seq['imu'], mu, std)      # (T, 7)
        bias, _ = compute_bias_labels(seq)                   # (T,) [rad/s]
        T = len(imu_norm)
        if T < window_size:
            return
        for i in range(T - window_size + 1):
            X_list.append(imu_norm[i: i + window_size])
            Y_list.append(bias[i + window_size - 1])

    for seq in seqs:
        _windows_from_seq(seq)

        # ---- 隧道增强：模拟一段 GPS 丢失 ----
        if tunnel_aug:
            import copy
            seq_aug = copy.deepcopy(seq)
            T = len(seq_aug['Time_s'])
            # 在中间段找一段运动平稳的区域模拟掉 GPS
            motion = np.where(seq_aug['v_ms'] >= MIN_SPEED_MS)[0]
            if len(motion) < 100:
                continue
            mid = len(motion) // 2
            center = motion[mid]
            half = 40  # 模拟 4 秒掉 GPS（前后留缓冲）
            out_start = max(0, center - half)
            out_end = min(T, center + half)
            if out_end - out_start < 30:
                continue
            # 把这一段标记为 GPS 无效 → compute_bias_labels 会插值填补偏标签
            seq_aug['gps_valid'][out_start:out_end] = False
            _windows_from_seq(seq_aug, label='tunnel')

    if not X_list:
        return np.empty((0, window_size, 7), dtype=np.float32), \
               np.empty((0, 1), dtype=np.float32)

    X = np.stack(X_list, axis=0).astype(np.float32)
    Y = np.array(Y_list, np.float32).reshape(-1, 1)
    return X, Y


# ============================================================================
# 主训练流程
# ============================================================================

def main():
    print("=" * 60)
    print("BiasNet 训练：陀螺 Z 轴零偏预测")
    print("=" * 60)

    # 1. 加载数据：训练集训练 BiasNet，Data05 作 fit 验证
    print(f"\n[1] 加载数据 ({', '.join(CALIB_TRAIN_IDS)} 训练 / "
          f"{CALIB_VAL_ID} 验证) ...")
    seqs_tr = load_calibration_seq(DATA_DIR_CALIB, CALIB_TRAIN_IDS)
    seqs_val = load_calibration_seq(DATA_DIR_CALIB, [CALIB_VAL_ID])
    if not seqs_tr:
        raise RuntimeError(
            f"未找到训练集 {CALIB_TRAIN_IDS}，请先运行 data_preprocessing_v2.py")

    # 2. 归一化统计量
    print("\n[2] 准备归一化参数 ...")
    mu, std, stats = load_or_compute_norm(seqs_tr, NORM_JSON)

    # 3. 构造训练样本
    print("\n[3] 构造训练样本 ...")
    X_tr, Y_tr = build_samples(seqs_tr, mu, std, tunnel_aug=True)
    print(f"  训练样本 + 隧道增强 ({len(CALIB_TRAIN_IDS)} 段): {len(X_tr)}")
    if seqs_val:
        X_val, Y_val = build_samples(seqs_val, mu, std, tunnel_aug=False)
        print(f"  {CALIB_VAL_ID} 验证样本: {len(X_val)}")
    else:
        idx = np.random.permutation(len(X_tr))
        n_val = max(1, int(0.15 * len(X_tr)))
        X_val, Y_val = X_tr[idx[:n_val]], Y_tr[idx[:n_val]]
        X_tr, Y_tr = X_tr[idx[n_val:]], Y_tr[idx[n_val:]]
    print(f"  零偏标签  均值={Y_tr.mean()*RAD2DEG:.4f} deg/s  "
          f"std={Y_tr.std()*RAD2DEG:.4f} deg/s")

    # 4. 构建模型
    print("\n[4] 构建 BiasNet ...")
    model = BiasNet(window_size=WINDOW_SIZE)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(optimizer=optimizer, loss='huber',
                  metrics=['mae'])
    # 预跑一次以初始化权重
    model(X_tr[:1])
    model.summary()

    # 5. 训练
    print("\n[5] 开始训练 ...")
    callbacks = [
        ModelCheckpoint(str(WEIGHTS_PATH), save_weights_only=True,
                        monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8,
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

    # 6. 保存信息
    final_val_mae = min(history.history['val_mae'])
    print(f"\n最优验证 MAE: {final_val_mae*RAD2DEG:.5f} deg/s")

    info = {
        'window_size': WINDOW_SIZE,
        'target_dt'  : TARGET_DT,
        'norm_stats' : stats,
        'train_samples': int(len(X_tr)),
        'val_samples'  : int(len(X_val)),
        'best_val_mae_rads': float(final_val_mae),
        'best_val_mae_degs': float(final_val_mae * RAD2DEG),
    }
    with open(MODEL_DIR / 'biasnet_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    # 7. 快速检验：在验证集上预测
    print("\n[6] 验证集快速检验 ...")
    Y_pred = model(X_val, training=False).numpy().flatten()
    Y_true = Y_val.flatten()
    residual = (Y_pred - Y_true) * RAD2DEG
    print(f"  预测误差  mean={residual.mean():.5f} deg/s  "
          f"std={residual.std():.5f} deg/s  "
          f"p95={np.percentile(np.abs(residual), 95):.5f} deg/s")
    print(f"\n权重已保存到：{WEIGHTS_PATH}")
    print("下一步：运行 validate_ekf.py 查看轨迹效果")

    # 8. 保存训练日志
    save_training_log(info, history, residual, DEFAULT_EKF_CONFIG)


def save_training_log(info, history, residual, config_params):
    """生成带时间戳的训练日志。"""
    from datetime import datetime
    log_dir = Path(__file__).parent / "training_logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = {
        'timestamp': timestamp,
        'config': {
            'q_yaw': config_params.q_yaw,
            'q_vel': config_params.q_vel,
            'q_bg': config_params.q_bg,
            'q_pos': config_params.q_pos,
            'r_gps_xy': config_params.r_gps_xy,
            'r_wheel': config_params.r_wheel,
            'r_nhc': config_params.r_nhc,
            'nhc_yaw_rate_thresh': config_params.nhc_yaw_rate_thresh,
            'nhc_r_scale_turn': config_params.nhc_r_scale_turn,
            'biasnet_max_deg': config_params.biasnet_max_deg,
            'freeze_yaw_below_ms': config_params.freeze_yaw_below_ms,
        },
        'training': {
            'train_samples': info['train_samples'],
            'val_samples': info['val_samples'],
            'epochs_completed': len(history.history['loss']),
            'best_epoch': int(np.argmin(history.history['val_loss'])) + 1,
            'best_val_loss': float(min(history.history['val_loss'])),
            'best_val_mae_rads': info['best_val_mae_rads'],
            'best_val_mae_degs': info['best_val_mae_degs'],
        },
        'prediction': {
            'residual_mean_degs': float(residual.mean()),
            'residual_std_degs': float(residual.std()),
            'residual_p95_degs': float(np.percentile(np.abs(residual), 95)),
        },
    }
    log_path = log_dir / f"train_{timestamp}.json"
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\n[日志] 已保存到 {log_path}")

    # ---- 追加汇总到 SUMMARY_TRAIN.md ----
    summary_path = log_dir / "SUMMARY_TRAIN.md"
    if not summary_path.exists():
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("# 训练记录汇总\n\n")
            f.write("| 时间 | 样本数 | Epochs | Best val MAE(°/s) |"
                    " 残差mean(°/s) | 残差std(°/s) | 残差p95(°/s) |\n")
            f.write("|------|--------|--------|--------------------|"
                    "------------------|-----------------|----------------|\n")

    tr = log['training']
    pr = log['prediction']
    dt_str = datetime.now().strftime("%m-%d %H:%M")
    line = (
        f"| {dt_str} "
        f"| {tr['train_samples']} "
        f"| {tr['epochs_completed']}/{tr.get('best_epoch', '—')} "
        f"| {tr['best_val_mae_degs']:.4f} "
        f"| {pr['residual_mean_degs']:+.4f} "
        f"| {pr['residual_std_degs']:.4f} "
        f"| {pr['residual_p95_degs']:.4f} |\n"
    )
    with open(summary_path, 'a', encoding='utf-8') as f:
        f.write(line)
    print(f"[汇总] 已追加到 {summary_path}")
    return log_path


if __name__ == '__main__':
    np.random.seed(42)
    tf.random.set_seed(42)
    main()
