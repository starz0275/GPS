"""
数据预处理 V2 —— 企业级 GPS 失联定位
======================================
改进点：
  1. 同时处理两份数据源（标定实车 + 260316 真实路跑）
  2. GPS 跳点自动清洗（速度阈值过滤）
  3. GPS 航向角锚定陀螺积分（大幅减少航向漂移）
  4. 非完整约束：侧向位移 = 0（仅前向位移用车速，横向不累积误差）
  5. 仅用 GPS 有效段生成标签，无效段跳过
  6. Huber 损失更鲁棒的标签（清洗残余大跳点）
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 配置
# ============================================================================

DATA_DIR_CALIB    = Path(__file__).parent / "标定实车数据"
DATA_CSV_REAL     = Path(__file__).parent / "260316_Data" / "260316_Data.csv"
OUTPUT_DIR        = Path(__file__).parent / "preprocessed_data"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_FREQ       = 10          # Hz (下采样目标频率)
TARGET_DT         = 1.0 / TARGET_FREQ
WINDOW_SIZE       = 30          # 时间步（3 秒上下文）
WINDOW_STRIDE     = 1
VEH_SPD_RAW_FACTOR = 260.63    # VehSpdRaw / factor = km/h
GPS_MAX_SPEED_KMH  = 150.0     # GPS 跳点速度阈值
MIN_SPEED_MS       = 0.5       # 低于此速度不计算 GPS 航向

# 260316 按时间切分：前段训练，后段验证（验证段与 validate_trajectory 一致）
TRAIN_SPLIT_T     = 490.0      # 训练时间上限（秒）
REAL_TEST_T_START = 620.0      # 验证集起始（秒），held-out

EARTH_A    = 6378137.0
EARTH_E2   = 0.00669437999014132
DEG2RAD    = np.pi / 180.0
RAD2DEG    = 180.0 / np.pi


# ============================================================================
# 工具：WGS84 → ENU
# ============================================================================

def wgs84_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    lat_r  = lat  * DEG2RAD;  lon_r  = lon  * DEG2RAD
    rlat_r = ref_lat * DEG2RAD; rlon_r = ref_lon * DEG2RAD

    N_ref = EARTH_A / np.sqrt(1 - EARTH_E2 * np.sin(rlat_r)**2)
    X0 = (N_ref + ref_alt) * np.cos(rlat_r) * np.cos(rlon_r)
    Y0 = (N_ref + ref_alt) * np.cos(rlat_r) * np.sin(rlon_r)
    Z0 = (N_ref * (1 - EARTH_E2) + ref_alt) * np.sin(rlat_r)

    R  = EARTH_A
    x  = (R + alt)  * np.cos(lat_r) * np.cos(lon_r)
    y  = (R + alt)  * np.cos(lat_r) * np.sin(lon_r)
    z  = (R * (1 - EARTH_E2) + alt) * np.sin(lat_r)

    dx = x - X0;  dy = y - Y0;  dz = z - Z0

    sl = np.sin(rlat_r); cl = np.cos(rlat_r)
    so = np.sin(rlon_r); co = np.cos(rlon_r)

    east  = -so * dx + co * dy
    north = -sl*co * dx - sl*so * dy + cl * dz
    return east, north


def batch_wgs84_to_enu(lat_arr, lon_arr, alt_arr, ref_lat, ref_lon, ref_alt):
    enu_x = np.zeros(len(lat_arr))
    enu_y = np.zeros(len(lat_arr))
    for i in range(len(lat_arr)):
        enu_x[i], enu_y[i] = wgs84_to_enu(
            lat_arr[i], lon_arr[i], alt_arr[i],
            ref_lat, ref_lon, ref_alt)
    return enu_x, enu_y


# ============================================================================
# GPS 跳点清洗
# ============================================================================

def clean_gps_outliers(t, lat, lon, alt, max_speed_kmh=GPS_MAX_SPEED_KMH):
    """
    基于隐含速度检测 GPS 跳点，并用线性插值替换。
    返回 (lat_clean, lon_clean, alt_clean, mask_valid)
    """
    lat = lat.copy(); lon = lon.copy(); alt = alt.copy()
    n = len(t)
    valid = np.ones(n, dtype=bool)

    # 逐帧计算隐含速度
    for i in range(1, n):
        dt = t[i] - t[i-1]
        if dt <= 0:
            valid[i] = False
            continue
        dlat = (lat[i] - lat[i-1]) * DEG2RAD * EARTH_A
        dlon = (lon[i] - lon[i-1]) * DEG2RAD * EARTH_A * np.cos(lat[i-1] * DEG2RAD)
        dist = np.sqrt(dlat**2 + dlon**2)
        speed_kmh = dist / dt * 3.6
        if speed_kmh > max_speed_kmh:
            valid[i] = False

    # 插值修补跳点
    idx = np.arange(n)
    if valid.sum() >= 2:
        lat[~valid] = np.interp(t[~valid], t[valid], lat[valid])
        lon[~valid] = np.interp(t[~valid], t[valid], lon[valid])
        alt[~valid] = np.interp(t[~valid], t[valid], alt[valid])

    outlier_cnt = (~valid).sum()
    if outlier_cnt > 0:
        print(f"    GPS 跳点清洗：修复 {outlier_cnt} 个异常点")
    return lat, lon, alt, valid


# ============================================================================
# 航向角（非完整约束核心）
# ============================================================================

def compute_heading_with_gps_anchor(
        t_arr, gyro_z_degs, v_ms,
        gps_heading_deg=None, gps_valid=None):
    """
    融合 GPS 航向与陀螺积分：
    - GPS 有效时：用 GPS 航向锚定（消除陀螺积累误差）
    - GPS 无效时：从上一个 GPS 航向开始积分陀螺

    参数
    ----
    gps_heading_deg : GPS 航向（正北顺时针度，0 = 无效/未知）
    gps_valid       : GPS 有效布尔数组（True = GPS 此帧可信）

    返回
    ----
    heading_rad : ENU 下的航向角（从东轴逆时针为正）
    """
    n = len(t_arr)
    heading = np.zeros(n)

    has_anchor = (gps_heading_deg is not None and
                  gps_valid is not None and
                  gps_valid.any())

    if has_anchor:
        # 找第一个 GPS 有效且速度够的帧，初始化航向
        first_ok = -1
        for i in range(n):
            if gps_valid[i] and v_ms[i] > MIN_SPEED_MS:
                # GPS 航向（北顺时针°）→ ENU 弧度
                heading[i] = (90.0 - gps_heading_deg[i]) * DEG2RAD
                first_ok = i
                break
        if first_ok == -1:
            first_ok = 0  # 找不到 GPS 就从 0 开始

        for i in range(first_ok + 1, n):
            dt = t_arr[i] - t_arr[i-1]
            if dt <= 0:
                heading[i] = heading[i-1]
                continue
            # 陀螺积分候选
            gyro_heading = heading[i-1] + gyro_z_degs[i] * DEG2RAD * dt

            if gps_valid[i] and v_ms[i] > MIN_SPEED_MS:
                # GPS 有效：直接用 GPS 航向（完全锚定，不漂移）
                heading[i] = (90.0 - gps_heading_deg[i]) * DEG2RAD
            else:
                heading[i] = gyro_heading
    else:
        # 无 GPS 航向参考：纯陀螺积分（从 0 开始）
        for i in range(1, n):
            dt = t_arr[i] - t_arr[i-1]
            heading[i] = heading[i-1] + gyro_z_degs[i] * DEG2RAD * (dt if dt > 0 else TARGET_DT)

    return heading


# ============================================================================
# 航位推算（非完整约束版）
# ============================================================================

def dead_reckoning_nhc(t_arr, v_ms, heading_rad):
    """
    非完整约束（NHC）航位推算：
    - 前向位移 = 车速 × dt（轮速计，可信）
    - 侧向位移 = 0（车辆不横向滑移）
    这与 ai-imu-dr 中 v_body_lateral ≈ 0 的伪量测等价。
    """
    n = len(t_arr)
    dt_arr = np.diff(t_arr, prepend=t_arr[0])
    dt_arr[0] = TARGET_DT  # 首帧用默认 dt

    base_dx = v_ms * np.cos(heading_rad) * dt_arr
    base_dy = v_ms * np.sin(heading_rad) * dt_arr
    return base_dx, base_dy


# ============================================================================
# 加载标定实车数据（Data01 / Data02）
# ============================================================================

def load_calibration_dataset(data_dir):
    datasets = []
    for ds_id in ["Data01", "Data02"]:
        imu_f   = data_dir / f"{ds_id}_IMU.txt"
        spd_f   = data_dir / f"{ds_id}_VehicleSpeed.txt"
        gps_f   = data_dir / f"{ds_id}_GNSS.txt"
        if not imu_f.exists():
            print(f"  [SKIP] {ds_id} 不存在")
            continue
        try:
            def read_tab(p):
                df = pd.read_csv(p, sep='\t', skipinitialspace=True, encoding='utf-8-sig')
                df.columns = [c.replace('ï»¿','').strip() for c in df.columns]
                return df

            imu  = read_tab(imu_f)
            spd  = read_tab(spd_f)
            gps  = read_tab(gps_f)

            # 统一时间范围
            t_s = max(imu['Time_s'].min(), spd['Time_s'].min(), gps['Time_s'].min())
            t_e = min(imu['Time_s'].max(), spd['Time_s'].max(), gps['Time_s'].max())
            t_grid = np.arange(t_s, t_e, TARGET_DT)

            def interp_col(src_t, src_v, tg):
                return interp1d(src_t, src_v, kind='linear',
                                bounds_error=False, fill_value='extrapolate')(tg)

            rec = {'Time_s': t_grid}
            for col in ['AccX_g','AccY_g','AccZ_g','GyroX_degs','GyroY_degs','GyroZ_degs']:
                rec[col] = interp_col(imu['Time_s'].values, imu[col].values, t_grid)
            rec['VehicleSpeed_kmh'] = interp_col(
                spd['Time_s'].values, spd['VehicleSpeed_kmh'].values, t_grid)

            lat_raw = interp_col(gps['Time_s'].values, gps['Latitude_deg'].values,  t_grid)
            lon_raw = interp_col(gps['Time_s'].values, gps['Longitude_deg'].values, t_grid)
            alt_raw = interp_col(gps['Time_s'].values, gps['Height_m'].values,      t_grid)

            # GPS 跳点清洗
            lat_c, lon_c, alt_c, gps_ok = clean_gps_outliers(t_grid, lat_raw, lon_raw, alt_raw)
            rec['Latitude_deg']  = lat_c
            rec['Longitude_deg'] = lon_c
            rec['Height_m']      = alt_c

            # 标定数据没有 HeadingRaw，从 GPS 位移推算航向（作为锚定）
            enu_x, enu_y = batch_wgs84_to_enu(
                lat_c, lon_c, alt_c, lat_c[0], lon_c[0], alt_c[0])
            rec['ENU_x'] = enu_x
            rec['ENU_y'] = enu_y

            # 从 GPS ENU 位移推导航向（速度 > 1m/s 时才可信）
            v_ms = rec['VehicleSpeed_kmh'] / 3.6
            gps_dx_raw = np.diff(enu_x, prepend=enu_x[0])
            gps_dy_raw = np.diff(enu_y, prepend=enu_y[0])
            gps_head_raw = np.arctan2(gps_dy_raw, gps_dx_raw)  # ENU 弧度

            # 平滑 GPS 航向
            gps_head_smooth = median_filter(gps_head_raw, size=9)

            # GPS 航向有效条件：速度够 + GPS 有效 + 位移够大
            gps_disp = np.sqrt(gps_dx_raw**2 + gps_dy_raw**2)
            gps_head_valid = gps_ok & (v_ms > MIN_SPEED_MS) & (gps_disp > 0.05)

            # 把 ENU 航向（弧度）转为"北顺时针度"以统一接口
            # enu_rad → north_cw_deg: (90 - enu_deg)
            gps_head_deg_nc = 90.0 - gps_head_smooth * RAD2DEG

            heading = compute_heading_with_gps_anchor(
                t_grid, rec['GyroZ_degs'], v_ms,
                gps_heading_deg=gps_head_deg_nc,
                gps_valid=gps_head_valid)

            rec['VehicleSpeed_ms'] = v_ms
            rec['Heading_rad']     = heading
            rec['GPS_heading_valid'] = gps_head_valid.astype(int)

            # NHC 航位推算
            base_dx, base_dy = dead_reckoning_nhc(t_grid, v_ms, heading)
            rec['Base_dx'] = base_dx
            rec['Base_dy'] = base_dy

            # 真实位移
            true_dx = np.diff(enu_x, prepend=enu_x[0])
            true_dy = np.diff(enu_y, prepend=enu_y[0])
            rec['True_dx'] = true_dx
            rec['True_dy'] = true_dy

            # 残差标签（在 GPS 有效处才有意义）
            rec['Target_dx']   = true_dx - base_dx
            rec['Target_dy']   = true_dy - base_dy
            rec['GPS_valid']   = gps_ok.astype(int)

            df_out = pd.DataFrame(rec)
            datasets.append(df_out)
            print(f"  [{ds_id}] OK: {len(df_out)} 行, "
                  f"速度 {v_ms.mean()*3.6:.1f} km/h avg, "
                  f"GPS有效 {gps_ok.mean()*100:.0f}%")
        except Exception as e:
            print(f"  [{ds_id}] FAIL: {e}")
            import traceback; traceback.print_exc()

    return datasets


# ============================================================================
# 加载 260316 真实数据
# ============================================================================

def load_real_dataset(csv_path, t_max=None, t_min=None):
    """
    t_max: 只使用 Time < t_max 的数据（训练时用）
    t_min: 只使用 Time >= t_min 的数据（测试时用）
    """
    print(f"  读取 {csv_path.name} ...")
    df = pd.read_csv(csv_path)
    if t_max is not None:
        df = df[df['Time'] < t_max].copy().reset_index(drop=True)
        print(f"  [训练] 截断到 t<{t_max}s，剩余 {len(df)} 行")
    if t_min is not None:
        df = df[df['Time'] >= t_min].copy().reset_index(drop=True)
        print(f"  [测试] 取 t>={t_min}s，共 {len(df)} 行")

    # ---- 单位换算 ----
    df['AccX_g']         = df['AccXRaw']
    df['AccY_g']         = df['AccYRaw']
    df['AccZ_g']         = df['AccZRaw']
    df['GyroX_degs']     = df['GyroXRaw']
    df['GyroY_degs']     = df['GyroYRaw']
    df['GyroZ_degs']     = df['GyroZRaw']
    df['VehicleSpeed_kmh'] = df['VehSpdRaw'] / VEH_SPD_RAW_FACTOR
    df['Time_s']         = df['Time']

    # ---- GPS 有效性 ----
    gps_valid_raw = (df['LatitudeRaw'] > 1).values

    # ---- GPS 跳点清洗（仅在有效段内检测） ----
    lat_raw = df['LatitudeRaw'].values.copy()
    lon_raw = df['LongitudeRaw'].values.copy()
    alt_raw = df['AltitudeRaw'].values.copy()
    t_raw   = df['Time_s'].values

    # 先把无效段 GPS 设为 NaN 以防跳点检测误判
    lat_raw[~gps_valid_raw] = np.nan
    lon_raw[~gps_valid_raw] = np.nan
    alt_raw[~gps_valid_raw] = np.nan

    # 仅在有效段做跳点清洗
    valid_idx = np.where(gps_valid_raw)[0]
    if len(valid_idx) > 2:
        lat_v  = df['LatitudeRaw'].values[valid_idx]
        lon_v  = df['LongitudeRaw'].values[valid_idx]
        alt_v  = df['AltitudeRaw'].values[valid_idx]
        t_v    = t_raw[valid_idx]
        lat_v, lon_v, alt_v, _ = clean_gps_outliers(t_v, lat_v, lon_v, alt_v)
        lat_raw[valid_idx] = lat_v
        lon_raw[valid_idx] = lon_v
        alt_raw[valid_idx] = alt_v

    # ---- 下采样到 10 Hz ----
    t_s = t_raw[0]
    t_e = t_raw[-1]
    t_grid = np.arange(t_s, t_e, TARGET_DT)
    n_grid = len(t_grid)

    def interp_raw(src_v, fill=0.0):
        mask = ~np.isnan(src_v) if np.isnan(src_v).any() else np.ones(len(src_v), bool)
        if mask.sum() < 2:
            return np.full(n_grid, fill)
        return interp1d(t_raw[mask], src_v[mask], kind='linear',
                        bounds_error=False, fill_value=fill)(t_grid)

    rec = {'Time_s': t_grid}
    for col in ['AccX_g','AccY_g','AccZ_g','GyroX_degs','GyroY_degs','GyroZ_degs']:
        rec[col] = interp_raw(df[col].values)
    rec['VehicleSpeed_kmh'] = interp_raw(df['VehicleSpeed_kmh'].values)

    # GPS 插值（仅有效点参与）
    lat_grid = interp_raw(lat_raw, fill=np.nan)
    lon_grid = interp_raw(lon_raw, fill=np.nan)
    alt_grid = interp_raw(alt_raw, fill=0.0)
    head_raw = df['HeadingRaw'].values.copy()
    head_raw[head_raw < 1] = np.nan  # 0 = 无效
    head_grid = interp_raw(head_raw, fill=np.nan)

    gps_valid_grid = (~np.isnan(lat_grid))

    # ENU 坐标（仅有效 GPS）
    enu_x = np.zeros(n_grid)
    enu_y = np.zeros(n_grid)
    if gps_valid_grid.any():
        first_v = np.argmax(gps_valid_grid)
        ref_lat = lat_grid[gps_valid_grid][0]
        ref_lon = lon_grid[gps_valid_grid][0]
        ref_alt = alt_grid[gps_valid_grid][0]

        v_idx = np.where(gps_valid_grid)[0]
        ex, ey = batch_wgs84_to_enu(
            lat_grid[v_idx], lon_grid[v_idx], alt_grid[v_idx],
            ref_lat, ref_lon, ref_alt)
        enu_x[v_idx] = ex
        enu_y[v_idx] = ey
        # 填充 GPS 无效段的 ENU（线性外推，用于连续性，但不生成标签）
        enu_x = np.interp(t_grid, t_grid[v_idx], ex)
        enu_y = np.interp(t_grid, t_grid[v_idx], ey)

    rec['ENU_x'] = enu_x
    rec['ENU_y'] = enu_y

    # ---- 航向计算（HeadingRaw 锚定）----
    v_ms_grid = rec['VehicleSpeed_kmh'] / 3.6

    # GPS 航向有效：GPS valid + heading 非 NaN
    head_valid_grid = gps_valid_grid & (~np.isnan(head_grid))
    head_grid_fill = np.where(np.isnan(head_grid), 0.0, head_grid)

    heading = compute_heading_with_gps_anchor(
        t_grid, rec['GyroZ_degs'], v_ms_grid,
        gps_heading_deg=head_grid_fill,
        gps_valid=head_valid_grid)

    rec['VehicleSpeed_ms']   = v_ms_grid
    rec['Heading_rad']       = heading
    rec['GPS_heading_valid'] = head_valid_grid.astype(int)

    # NHC 航位推算
    base_dx, base_dy = dead_reckoning_nhc(t_grid, v_ms_grid, heading)
    rec['Base_dx'] = base_dx
    rec['Base_dy'] = base_dy

    # 真实位移（仅 GPS 有效段有意义）
    true_dx = np.diff(enu_x, prepend=enu_x[0])
    true_dy = np.diff(enu_y, prepend=enu_y[0])
    rec['True_dx'] = true_dx
    rec['True_dy'] = true_dy

    rec['Target_dx'] = true_dx - base_dx
    rec['Target_dy'] = true_dy - base_dy
    rec['GPS_valid']  = gps_valid_grid.astype(int)

    df_out = pd.DataFrame(rec)
    pct = gps_valid_grid.mean() * 100
    print(f"  [260316] OK: {len(df_out)} 行, "
          f"速度 {v_ms_grid.mean()*3.6:.1f} km/h avg, "
          f"GPS有效 {pct:.0f}%")
    return [df_out]


# ============================================================================
# 残差大跳点二次清洗（针对标签）
# ============================================================================

def clean_label_outliers(df, sigma=4.0):
    """对残差标签做 ±4σ 截断（剩余 GPS 错误）"""
    for col in ['Target_dx', 'Target_dy']:
        mu  = df[col].mean()
        std = df[col].std()
        mask = np.abs(df[col] - mu) > sigma * std
        df.loc[mask, col] = np.clip(df.loc[mask, col], mu - sigma * std, mu + sigma * std)
        if mask.sum() > 0:
            print(f"    标签截断 {col}: {mask.sum()} 个点（±{sigma}σ）")
    return df


# ============================================================================
# 特征归一化 & 滑动窗口
# ============================================================================

FEATURE_COLS = [
    'AccX_g', 'AccY_g', 'AccZ_g',
    'GyroX_degs', 'GyroY_degs', 'GyroZ_degs',
    'VehicleSpeed_ms',
    'Base_dx', 'Base_dy'
]


def compute_norm_stats(df_list):
    """在所有数据集上联合计算归一化统计量"""
    combined = pd.concat(df_list, ignore_index=True)
    stats = {}
    for col in FEATURE_COLS:
        stats[col] = {
            'mean': float(combined[col].mean()),
            'std':  float(combined[col].std() + 1e-6)
        }
    return stats


def normalize_df(df, stats):
    for col in FEATURE_COLS:
        df[f'{col}_norm'] = (df[col] - stats[col]['mean']) / stats[col]['std']
    return df


def create_windows(df, window_size, stride):
    """
    滑动窗口，只在 GPS 全程有效的窗口内生成样本。
    标签取窗口最后一帧的残差。
    """
    feat_cols = [f'{c}_norm' for c in FEATURE_COLS]
    X_list, Y_list, ts_list = [], [], []

    gps_ok = df['GPS_valid'].values
    feat   = df[feat_cols].values
    tdx    = df['Target_dx'].values
    tdy    = df['Target_dy'].values
    times  = df['Time_s'].values

    n = len(df)
    for i in range(0, n - window_size, stride):
        if gps_ok[i:i+window_size].all():
            X_list.append(feat[i:i+window_size])
            target_idx = i + window_size - 1
            Y_list.append([tdx[target_idx], tdy[target_idx]])
            ts_list.append(times[target_idx])

    if len(X_list) == 0:
        return np.empty((0, window_size, len(feat_cols))), \
               np.empty((0, 2)), np.empty((0,))

    return (np.array(X_list, dtype=np.float32),
            np.array(Y_list,  dtype=np.float32),
            np.array(ts_list, dtype=np.float32))


# 隧道增强：模拟 GPS 丢失段，让 TCN 学习 outage 下的残差
TUNNEL_AUG_DURATION_S = 50.0
TUNNEL_AUG_STRIDE     = 15     # 每隔多少帧取一个隧道窗口起点
TUNNEL_AUG_MAX_PER_DF = 200    # 每段数据最多增强窗口数


def _recompute_base_for_outage(df, out_start, out_end):
    """
    在 [out_start, out_end) 模拟 GPS 丢失，按推理逻辑重算 Base 与 Target。
    返回更新后的 Target_dx/dy 及 Base_dx/dy（仅用于构造特征/标签）。
    """
    n = len(df)
    gps_sim = df['GPS_valid'].values.astype(bool).copy()
    gps_sim[out_start:out_end] = False

    head_deg = 90.0 - df['Heading_rad'].values * RAD2DEG
    v_ms = df['VehicleSpeed_ms'].values
    t_arr = df['Time_s'].values

    heading = compute_heading_with_gps_anchor(
        t_arr, df['GyroZ_degs'].values, v_ms,
        gps_heading_deg=head_deg,
        gps_valid=gps_sim)
    base_dx, base_dy = dead_reckoning_nhc(t_arr, v_ms, heading)
    true_dx = df['True_dx'].values
    true_dy = df['True_dy'].values
    return base_dx, base_dy, true_dx - base_dx, true_dy - base_dy


def create_windows_with_tunnel(df, window_size, stride, norm_stats):
    """标准 GPS 有效窗口 + 模拟隧道窗口。"""
    X, Y, ts = create_windows(df, window_size, stride)
    X_list = list(X) if len(X) else []
    Y_list = list(Y) if len(Y) else []
    ts_list = list(ts) if len(ts) else []

    n = len(df)
    tunnel_frames = int(TUNNEL_AUG_DURATION_S / TARGET_DT)
    gps_ok = df['GPS_valid'].values.astype(bool)
    times = df['Time_s'].values
    feat_cols = [f'{c}_norm' for c in FEATURE_COLS]

    added = 0
    i = window_size
    while i < n - window_size and added < TUNNEL_AUG_MAX_PER_DF:
        out_start = i + 5
        out_end = min(out_start + tunnel_frames, n - 5)
        if out_end - out_start < tunnel_frames // 2:
            i += TUNNEL_AUG_STRIDE
            continue
        # 隧道前后需有真实 GPS（标签可靠）
        if not gps_ok[max(0, out_start - 10):out_start].any():
            i += TUNNEL_AUG_STRIDE
            continue
        if not gps_ok[out_end:min(out_end + 10, n)].any():
            i += TUNNEL_AUG_STRIDE
            continue

        bdx, bdy, tdx, tdy = _recompute_base_for_outage(df, out_start, out_end)
        win_end = out_start + window_size - 1
        if win_end >= out_end or win_end >= n:
            i += TUNNEL_AUG_STRIDE
            continue

        feat_win = np.zeros((window_size, len(FEATURE_COLS)), dtype=np.float32)
        for j, t in enumerate(range(i, i + window_size)):
            for c, col in enumerate(FEATURE_COLS):
                if col == 'Base_dx':
                    val = float(bdx[t])
                elif col == 'Base_dy':
                    val = float(bdy[t])
                else:
                    feat_win[j, c] = float(df[f'{col}_norm'].iloc[t])
                    continue
                mu, std = norm_stats[col]['mean'], norm_stats[col]['std']
                feat_win[j, c] = (val - mu) / (std + 1e-6)

        tgt_i = i + window_size - 1
        X_list.append(feat_win)
        Y_list.append([tdx[tgt_i], tdy[tgt_i]])
        ts_list.append(times[tgt_i])
        added += 1
        i += TUNNEL_AUG_STRIDE

    if not X_list:
        return X, Y, ts
    return (np.array(X_list, dtype=np.float32),
            np.array(Y_list, dtype=np.float32),
            np.array(ts_list, dtype=np.float32))


# ============================================================================
# 主流程
# ============================================================================

def main():
    print("=" * 70)
    print("数据预处理 V2 —— 企业级 GPS 失联定位")
    print("=" * 70)

    all_dfs = []

    # 1. 260316 训练段（标定数据 IMU 统计差异太大，不混用）
    print("\n[1/3] 加载 260316 真实路跑数据（训练段 t<{:.0f}s）...".format(TRAIN_SPLIT_T))
    if DATA_CSV_REAL.exists():
        real_dfs = load_real_dataset(DATA_CSV_REAL, t_max=TRAIN_SPLIT_T)
        all_dfs.extend(real_dfs)
    else:
        raise RuntimeError(f"{DATA_CSV_REAL} 不存在")

    if not all_dfs:
        raise RuntimeError("没有加载任何数据！")

    # 2. 标签大跳点二次清洗
    print("\n[2/3] 标签大跳点清洗...")
    all_dfs = [clean_label_outliers(df) for df in all_dfs]

    # 3. 归一化
    print("\n[3/3] 计算归一化统计量（260316 训练段）...")
    norm_stats = compute_norm_stats(all_dfs)

    # 5. 归一化 + 滑动窗口
    X_parts, Y_parts, ts_parts = [], [], []
    for df in all_dfs:
        df = normalize_df(df, norm_stats)
        X_, Y_, ts_ = create_windows_with_tunnel(
            df, WINDOW_SIZE, WINDOW_STRIDE, norm_stats)
        if len(X_) > 0:
            X_parts.append(X_)
            Y_parts.append(Y_)
            ts_parts.append(ts_)
        print(f"     => {len(X_)} 个训练窗口")

    X_all = np.concatenate(X_parts, axis=0)
    Y_all = np.concatenate(Y_parts, axis=0)
    ts_all = np.concatenate(ts_parts, axis=0)

    # 6. 保存
    np.save(OUTPUT_DIR / "X_train.npy", X_all)
    np.save(OUTPUT_DIR / "Y_train.npy", Y_all)
    np.save(OUTPUT_DIR / "timestamps.npy", ts_all)

    import json
    with open(OUTPUT_DIR / "normalization_stats.json", 'w') as f:
        json.dump({
            'stats':        norm_stats,
            'feature_names': [f'{c}_norm' for c in FEATURE_COLS],
            'window_size':  WINDOW_SIZE,
            'target_freq':  TARGET_FREQ,
        }, f, indent=2)

    # 保存合并的对齐数据（可选，用于分析）
    pd.concat(all_dfs, ignore_index=True).to_csv(
        OUTPUT_DIR / "aligned_data.csv", index=False)

    print()
    print("=" * 70)
    print(f"预处理完成！")
    print(f"  X_train: {X_all.shape}  (样本, 时间步={WINDOW_SIZE}, 特征=9)")
    print(f"  Y_train: {Y_all.shape}")
    print(f"  dx 标签: μ={Y_all[:,0].mean():.4f}  σ={Y_all[:,0].std():.4f}"
          f"  range=[{Y_all[:,0].min():.3f}, {Y_all[:,0].max():.3f}]")
    print(f"  dy 标签: μ={Y_all[:,1].mean():.4f}  σ={Y_all[:,1].std():.4f}"
          f"  range=[{Y_all[:,1].min():.3f}, {Y_all[:,1].max():.3f}]")
    print("=" * 70)

    # ---- 生成验证集（260316 后 30%，训练时从未见过） ----
    print("\n[验证集] 处理 260316 验证段 (t >= {:.0f}s)...".format(REAL_TEST_T_START))
    if DATA_CSV_REAL.exists():
        test_dfs = load_real_dataset(DATA_CSV_REAL, t_min=REAL_TEST_T_START)
        test_dfs = [clean_label_outliers(df) for df in test_dfs]
        test_dfs_normed = [normalize_df(df.copy(), norm_stats) for df in test_dfs]
        X_test_parts, Y_test_parts, ts_test_parts = [], [], []
        for df in test_dfs_normed:
            Xt, Yt, tst = create_windows(df, WINDOW_SIZE, WINDOW_STRIDE)
            if len(Xt) > 0:
                X_test_parts.append(Xt)
                Y_test_parts.append(Yt)
                ts_test_parts.append(tst)
        if X_test_parts:
            X_test = np.concatenate(X_test_parts)
            Y_test = np.concatenate(Y_test_parts)
            ts_test = np.concatenate(ts_test_parts)
            np.save(OUTPUT_DIR / "X_test.npy",  X_test)
            np.save(OUTPUT_DIR / "Y_test.npy",  Y_test)
            np.save(OUTPUT_DIR / "ts_test.npy", ts_test)
            # 保存测试集的对齐 CSV（含 ENU 坐标，用于轨迹可视化）
            pd.concat(test_dfs, ignore_index=True).to_csv(
                OUTPUT_DIR / "test_aligned.csv", index=False)
            print(f"  X_test: {X_test.shape}")
            print(f"  Y_test: {Y_test.shape}")
            print(f"  test_aligned.csv 已保存（含 ENU 坐标）")

    return X_all, Y_all, ts_all


if __name__ == "__main__":
    main()
