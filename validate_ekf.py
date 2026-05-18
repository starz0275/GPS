"""
validate_ekf.py — EKF 导航器验证脚本
=====================================

使用 260316_Data.csv 的保留测试段（t >= 620 s）评估：
  1. 纯死推算（无任何校正）
  2. BiasNet + EKF 校正

评估指标：
  - 相对位置误差（RPE）：每 30 s 的漂移
  - 累积位置误差随时间曲线
  - 轨迹对比图（GPS 真值 vs 纯 DR vs EKF）
  - 航向对比图（GPS 真值 vs 陀螺积分 vs EKF）
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

# 配置中文字体
font_paths = [
    'C:/Windows/Fonts/msyh.ttc',    # 微软雅黑
    'C:/Windows/Fonts/simhei.ttf',  # 黑体
    'C:/Windows/Fonts/simsun.ttc',  # 宋体
]
for fp in font_paths:
    if Path(fp).exists():
        fm.fontManager.addfont(fp)
        prop = fm.FontProperties(fname=fp)
        font_name = prop.get_name()
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = [font_name] + matplotlib.rcParams['font.sans-serif']
        break

matplotlib.rcParams['axes.unicode_minus'] = False
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
import json, warnings
warnings.filterwarnings('ignore')

from ekf_navigator import EKFNavigatorNP, load_norm_stats

# ============================================================================
# 配置
# ============================================================================

DATA_CSV       = Path(__file__).parent / "260316_Data" / "260316_Data.csv"
MODEL_DIR      = Path(__file__).parent / "trained_models"
WEIGHTS_PATH   = MODEL_DIR / "biasnet_weights.weights.h5"
NORM_JSON      = Path(__file__).parent / "preprocessed_data" / "normalization_stats.json"
OUTPUT_DIR     = MODEL_DIR

REAL_TEST_T_START = 620.0     # 与 trajectory_data.VAL_T_START 一致
TARGET_DT         = 0.1       # s
DEG2RAD           = np.pi / 180.0
RAD2DEG           = 180.0 / np.pi
EARTH_A           = 6378137.0
MIN_SPEED_MS      = 0.5
VEH_SPD_FACTOR    = 260.63
WINDOW_SIZE       = 30

# ============================================================================
# 坐标工具
# ============================================================================

def latlon_to_enu(lat, lon, ref_lat, ref_lon):
    dlat = (lat - ref_lat) * DEG2RAD
    dlon = (lon - ref_lon) * DEG2RAD
    east  = EARTH_A * dlon * np.cos(ref_lat * DEG2RAD)
    north = EARTH_A * dlat
    return east, north


def clean_gps_outliers(t, lat, lon, max_kmh=150.0):
    lat = lat.copy(); lon = lon.copy()
    valid = np.ones(len(t), dtype=bool)
    for i in range(1, len(t)):
        dt_ = t[i] - t[i-1]
        if dt_ <= 0: valid[i] = False; continue
        dlat = (lat[i] - lat[i-1]) * DEG2RAD * EARTH_A
        dlon = (lon[i] - lon[i-1]) * DEG2RAD * EARTH_A * np.cos(lat[i-1]*DEG2RAD)
        if np.sqrt(dlat**2 + dlon**2) / dt_ * 3.6 > max_kmh:
            valid[i] = False
    if valid.sum() >= 2:
        lat[~valid] = np.interp(t[~valid], t[valid], lat[valid])
        lon[~valid] = np.interp(t[~valid], t[valid], lon[valid])
    return lat, lon, valid


# ============================================================================
# 加载测试段数据
# ============================================================================

def load_test_segment():
    print(f"[数据] 加载测试段 (t >= {REAL_TEST_T_START} s) ...")
    df = pd.read_csv(DATA_CSV)
    df = df[df['Time'] >= REAL_TEST_T_START].copy().reset_index(drop=True)
    if len(df) < 100:
        raise RuntimeError(f"测试段数据不足（{len(df)} 行），请检查 REAL_TEST_T_START 设置")

    t_raw = df['Time'].values
    t_s, t_e = t_raw[0], t_raw[-1]
    tg = np.arange(t_s, t_e, TARGET_DT)
    print(f"  时间范围: {t_s:.1f}–{t_e:.1f} s  ({len(tg)} 帧，约 {(t_e-t_s)/60:.1f} 分钟)")

    def interp1(src_v, fill=0.0):
        mask = ~np.isnan(src_v.astype(float)) if np.isnan(src_v.astype(float)).any() \
               else np.ones(len(src_v), bool)
        if mask.sum() < 2:
            return np.full(len(tg), fill, np.float32)
        return interp1d(t_raw[mask], src_v[mask], bounds_error=False,
                        fill_value=fill)(tg).astype(np.float32)

    # IMU（6 通道）
    imu_cols = ['AccXRaw','AccYRaw','AccZRaw','GyroXRaw','GyroYRaw','GyroZRaw']
    imu_raw = np.stack([interp1(df[c].values) for c in imu_cols], axis=1)  # (T,6)
    gyro_z_rad = imu_raw[:, 5] * DEG2RAD

    v_ms = interp1(df['VehSpdRaw'].values / VEH_SPD_FACTOR / 3.6)   # km/h → m/s

    # GPS
    lat_raw  = df['LatitudeRaw'].values.copy().astype(float)
    lon_raw  = df['LongitudeRaw'].values.copy().astype(float)
    head_raw = df['HeadingRaw'].values.copy().astype(float) \
               if 'HeadingRaw' in df.columns else np.zeros(len(df))

    gps_ok_raw = (lat_raw > 1)
    lat_raw[~gps_ok_raw] = np.nan
    lon_raw[~gps_ok_raw] = np.nan
    head_raw[head_raw < 1] = np.nan

    # 跳点清洗（仅有效段）
    vidx = np.where(gps_ok_raw)[0]
    if len(vidx) > 2:
        lv, lnv, _ = clean_gps_outliers(
            t_raw[vidx], lat_raw[vidx], lon_raw[vidx])
        lat_raw[vidx] = lv; lon_raw[vidx] = lnv

    lat_g  = interp1(lat_raw, fill=np.nan)
    lon_g  = interp1(lon_raw, fill=np.nan)
    head_g = interp1(head_raw, fill=np.nan)
    gps_valid = ~np.isnan(lat_g)

    print(f"  GPS 有效率: {gps_valid.mean():.1%}  "
          f"({gps_valid.sum()} / {len(gps_valid)} 帧)")

    if gps_valid.sum() < 5:
        raise RuntimeError("GPS 有效帧太少，无法建立参考轨迹")

    # ENU（以首个有效 GPS 为原点）
    ref_lat = lat_g[gps_valid][0]; ref_lon = lon_g[gps_valid][0]
    enu_x_v, enu_y_v = latlon_to_enu(
        lat_g[gps_valid], lon_g[gps_valid], ref_lat, ref_lon)

    enu_x_truth = np.full(len(tg), np.nan, np.float32)
    enu_y_truth = np.full(len(tg), np.nan, np.float32)
    enu_x_truth[gps_valid] = enu_x_v
    enu_y_truth[gps_valid] = enu_y_v

    # GPS ENU 推算航向（仅速度足够时）
    enu_x_interp = np.interp(np.arange(len(tg)),
                             np.where(gps_valid)[0], enu_x_v)
    enu_y_interp = np.interp(np.arange(len(tg)),
                             np.where(gps_valid)[0], enu_y_v)
    dx_gps = np.gradient(enu_x_interp, tg)
    dy_gps = np.gradient(enu_y_interp, tg)
    gps_theta_enu = np.arctan2(dy_gps, dx_gps).astype(np.float32)
    gps_theta_smooth = median_filter(gps_theta_enu, size=11)

    # 如果 HeadingRaw 存在，融合（更可靠）
    has_raw_head = ~np.isnan(head_g)
    if has_raw_head.sum() > 10:
        enu_head = (90.0 - head_g) * DEG2RAD
        gps_theta_smooth[has_raw_head & gps_valid] = \
            enu_head[has_raw_head & gps_valid].astype(np.float32)
    gps_head_valid = gps_valid & (v_ms > MIN_SPEED_MS)

    return {
        'Time_s'         : tg,
        'imu_raw'        : imu_raw,
        'gyro_z_rad'     : gyro_z_rad,
        'v_ms'           : v_ms,
        'gps_theta'      : gps_theta_smooth.astype(np.float32),
        'gps_valid'      : gps_head_valid,
        'gps_valid_full' : gps_head_valid.copy(),  # 原始 GPS 有效掩码（未模拟丢失）
        'enu_x_truth'    : enu_x_truth,
        'enu_y_truth'    : enu_y_truth,
        'ref_lat'        : ref_lat,
        'ref_lon'        : ref_lon,
    }


GPS_LOSS_SIM_SECS = 60.0    # 模拟 GPS 丢失时长（秒）


def simulate_gps_loss(seq, loss_duration_s=GPS_LOSS_SIM_SECS):
    """
    在测试段中间屏蔽 GPS 信号（模拟隧道场景），
    返回修改后的 gps_valid 掩码及丢失区间信息。
    """
    tg = seq['Time_s']
    gps_v = seq['gps_valid_full'].copy()
    T = len(tg)

    # 在距离起点 15s 后开始丢失（留一段用于 EKF 初始化）
    loss_start_s = 15.0
    loss_end_s   = loss_start_s + loss_duration_s

    loss_start_idx = np.searchsorted(tg - tg[0], loss_start_s)
    loss_end_idx   = np.searchsorted(tg - tg[0], loss_end_s)
    loss_end_idx   = min(loss_end_idx, T - 1)

    gps_v[loss_start_idx: loss_end_idx] = False

    actual_loss_s = (tg[loss_end_idx] - tg[loss_start_idx])
    print(f"  [模拟 GPS 丢失] 索引 {loss_start_idx}–{loss_end_idx}，"
          f"持续 {actual_loss_s:.1f} s")

    return gps_v, loss_start_idx, loss_end_idx


# ============================================================================
# 纯死推算（基准）
# ============================================================================

def run_pure_dr(seq):
    """
    纯陀螺积分（无零偏校正） + NHC 位置累积
    GPS 有效时锚定航向（仅在实际 GPS 有效处更新，模拟无校正情况下的基准）
    """
    T = len(seq['Time_s'])
    dt = TARGET_DT
    tg = seq['Time_s']
    gyro_z = seq['gyro_z_rad']
    v_ms   = seq['v_ms']
    gps_th = seq['gps_theta']
    gps_v  = seq['gps_valid']

    heading = np.zeros(T, np.float32)
    # 初始化：用第一个 GPS 有效帧的航向
    first_valid = np.argmax(gps_v)
    heading[first_valid] = float(gps_th[first_valid])

    for k in range(first_valid + 1, T):
        dt_k = tg[k] - tg[k-1]
        heading[k] = heading[k-1] + gyro_z[k] * (dt_k if dt_k > 0 else dt)
        if gps_v[k]:
            heading[k] = float(gps_th[k])      # GPS 有效时直接用 GPS 航向

    dx = v_ms * np.cos(heading) * dt
    dy = v_ms * np.sin(heading) * dt
    return np.cumsum(dx), np.cumsum(dy), heading


# ============================================================================
# 计算误差指标
# ============================================================================

def compute_errors(pred_x, pred_y, truth_x, truth_y, gps_valid):
    """
    在 GPS 有效处计算位置误差（m）
    返回 (errors, times_valid)
    """
    idx = np.where(gps_valid)[0]
    if len(idx) == 0:
        return np.array([]), np.array([])
    err = np.sqrt((pred_x[idx] - truth_x[idx])**2 +
                  (pred_y[idx] - truth_y[idx])**2)
    return err, idx


def rpe_30s(pred_x, pred_y, truth_x, truth_y, tg, window_s=30.0):
    """每 30 s 相对位置误差（RPE）"""
    step = max(1, int(window_s / TARGET_DT))
    rpe_list = []
    for i in range(0, len(tg) - step, step):
        j = i + step
        dp_x = (pred_x[j] - pred_x[i]) - (truth_x[j] - truth_x[i])
        dp_y = (pred_y[j] - pred_y[i]) - (truth_y[j] - truth_y[i])
        rpe_list.append(np.sqrt(dp_x**2 + dp_y**2))
    return np.array(rpe_list)


# ============================================================================
# 绘图
# ============================================================================

def plot_results(seq, dr_x, dr_y, dr_h,
                 ekf_x, ekf_y, ekf_h, ekf_bias,
                 loss_start_idx=None, loss_end_idx=None):
    tg      = seq['Time_s']
    truth_x = seq['enu_x_truth']
    truth_y = seq['enu_y_truth']
    gps_v   = seq['gps_valid']
    t_rel   = tg - tg[0]

    # 仅在 GPS 有效处对齐零点（以首个有效帧为原点）
    first = np.argmax(gps_v)
    truth_ox = float(truth_x[first]) if not np.isnan(truth_x[first]) else 0.0
    truth_oy = float(truth_y[first]) if not np.isnan(truth_y[first]) else 0.0

    # 位置预测同样从首帧对齐
    dr_ox  = dr_x[first]; dr_oy  = dr_y[first]
    ekf_ox = ekf_x[first]; ekf_oy = ekf_y[first]

    tx = truth_x - truth_ox; ty = truth_y - truth_oy
    dox = dr_x  - dr_ox;   doy = dr_y  - dr_oy
    ex  = ekf_x - ekf_ox;  ey  = ekf_y - ekf_oy

    # 累积误差（GPS 有效处）
    idx_v = np.where(gps_v)[0]
    if len(idx_v) > 1:
        dr_err  = np.sqrt((dox[idx_v] - tx[idx_v])**2 + (doy[idx_v] - ty[idx_v])**2)
        ekf_err = np.sqrt((ex[idx_v]  - tx[idx_v])**2 + (ey[idx_v]  - ty[idx_v])**2)
    else:
        dr_err  = np.zeros(1); ekf_err = np.zeros(1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('EKF 导航器验证结果（测试集）', fontsize=14, fontweight='bold')

    # ---- (a) 轨迹对比 ----
    ax = axes[0, 0]
    ax.plot(tx[idx_v], ty[idx_v], 'k-', lw=2, label='GPS 真值', zorder=5)
    ax.plot(dox, doy, 'r--', lw=1.2, alpha=0.8, label='纯陀螺 DR')
    ax.plot(ex,  ey,  'b-',  lw=1.5, alpha=0.9, label='BiasNet + EKF')
    # 标记 GPS 丢失区间
    if loss_start_idx is not None and loss_end_idx is not None:
        ax.axvspan(ex[loss_start_idx], ex[loss_end_idx],
                   alpha=0.15, color='orange', label='GPS 丢失区间')
        ax.plot(ex[loss_start_idx], ey[loss_start_idx],
                'go', ms=8, label='丢失起点')
        ax.plot(ex[loss_end_idx], ey[loss_end_idx],
                'g^', ms=8, label='丢失终点')
    ax.set_xlabel('East (m)'); ax.set_ylabel('North (m)')
    ax.set_title('轨迹对比（橙色 = 模拟 GPS 丢失段）')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4); ax.set_aspect('equal')

    # ---- (b) 累积位置误差 ----
    ax = axes[0, 1]
    t_idx = t_rel[idx_v]
    ax.plot(t_idx, dr_err,  'r-', lw=1.5, label=f'纯 DR  终点={dr_err[-1]:.1f} m')
    ax.plot(t_idx, ekf_err, 'b-', lw=1.5, label=f'EKF    终点={ekf_err[-1]:.1f} m')
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('位置误差 (m)')
    ax.set_title('累积位置误差 (GPS 有效帧)')
    ax.legend(); ax.grid(True, alpha=0.4)

    # ---- (c) 航向对比 ----
    ax = axes[1, 0]
    gps_head = np.where(gps_v, seq['gps_theta'] * RAD2DEG, np.nan)
    ax.plot(t_rel, gps_head, 'k-', lw=1.5, alpha=0.7, label='GPS 航向')
    ax.plot(t_rel, dr_h * RAD2DEG,  'r--', lw=1, alpha=0.8, label='纯陀螺积分')
    ax.plot(t_rel, ekf_h * RAD2DEG, 'b-',  lw=1.5, label='BiasNet+EKF')
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('航向 (°)')
    ax.set_title('航向对比')
    ax.legend(); ax.grid(True, alpha=0.4)

    # ---- (d) 网络预测零偏 ----
    ax = axes[1, 1]
    ax.plot(t_rel, ekf_bias * RAD2DEG, 'g-', lw=1.2, label='BiasNet 预测零偏')
    ax.axhline(0, color='k', linestyle='--', lw=0.8)
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('陀螺 Z 零偏 (°/s)')
    ax.set_title('网络预测陀螺零偏')
    ax.legend(); ax.grid(True, alpha=0.4)

    plt.tight_layout()
    out_path = OUTPUT_DIR / 'ekf_validation.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[图像] 已保存到 {out_path}")

    # ---- 打印汇总指标 ----
    rpe_dr  = rpe_30s(dox, doy, tx, ty, t_rel)
    rpe_ekf = rpe_30s(ex,  ey,  tx, ty, t_rel)
    print("\n" + "=" * 55)
    print("验证指标汇总")
    print("=" * 55)
    print(f"  GPS 有效帧数：{len(idx_v)} / {len(tg)} ({gps_v.mean():.1%})")
    print(f"  测试时长：{t_rel[-1]:.0f} s")
    if len(dr_err) > 0:
        print(f"\n  纯 DR  ：终点误差 {dr_err[-1]:.2f} m  "
              f"中值误差 {np.median(dr_err):.2f} m  "
              f"RPE30s 中值 {np.median(rpe_dr):.2f} m")
        print(f"  EKF    ：终点误差 {ekf_err[-1]:.2f} m  "
              f"中值误差 {np.median(ekf_err):.2f} m  "
              f"RPE30s 中值 {np.median(rpe_ekf):.2f} m")
        if dr_err[-1] > 0:
            improvement = (1 - ekf_err[-1] / dr_err[-1]) * 100
            print(f"\n  终点误差改善：{improvement:+.1f}%")
    print("=" * 55)


# ============================================================================
# 主流程
# ============================================================================

def main():
    print("=" * 55)
    print("EKF 导航器验证（测试集：260316_Data t>=620s）")
    print("=" * 55)

    # 1. 加载测试数据
    seq = load_test_segment()

    # 2. 模拟 GPS 丢失掩码
    print("\n[1] 设置模拟 GPS 丢失...")
    gps_v_sim, loss_start, loss_end = simulate_gps_loss(seq)
    seq_sim = dict(seq)
    seq_sim['gps_valid'] = gps_v_sim   # 用模拟后的掩码替换

    # 3. 纯 DR 基准（模拟 GPS 丢失）
    print("\n[2] 运行纯陀螺 DR（模拟 GPS 丢失）...")
    dr_x, dr_y, dr_h = run_pure_dr(seq_sim)

    # 4. BiasNet + EKF（模拟 GPS 丢失）
    print("\n[3] 运行 BiasNet + EKF 导航器（模拟 GPS 丢失）...")
    if not WEIGHTS_PATH.exists():
        raise RuntimeError(
            f"未找到权重文件 {WEIGHTS_PATH}，请先运行 train_ekf.py")

    norm_stats = load_norm_stats(str(NORM_JSON))
    nav = EKFNavigatorNP(
        weights_path=str(WEIGHTS_PATH),
        norm_stats=norm_stats,
        window_size=WINDOW_SIZE,
    )

    ekf_x, ekf_y, ekf_h, ekf_bias = nav.run(
        imu_raw    = seq_sim['imu_raw'],
        v_ms       = seq_sim['v_ms'],
        gyro_z_rad = seq_sim['gyro_z_rad'],
        gps_theta  = seq_sim['gps_theta'],
        gps_valid  = gps_v_sim,
        dt         = TARGET_DT,
    )

    # ---- 诊断：检查零偏预测是否合理 ----
    print("\n[诊断] 零偏预测统计：")
    print(f"  归一化统计量（标定数据）:")
    for k in ['AccX_g','AccY_g','AccZ_g','GyroX_degs','GyroY_degs','GyroZ_degs']:
        print(f"    {k:16s}  mean={norm_stats[k]['mean']:+8.4f}  std={norm_stats[k]['std']:.4f}")
    print(f"  260316 IMU 原始范围:")
    imu_cols = ['AccXRaw','AccYRaw','AccZRaw','GyroXRaw','GyroYRaw','GyroZRaw']
    for i, c in enumerate(imu_cols):
        print(f"    {c:16s}  mean={seq_sim['imu_raw'][:,i].mean():+8.4f}  std={seq_sim['imu_raw'][:,i].std():.4f}")
    print(f"  BiasNet 预测零偏: mean={ekf_bias.mean()*RAD2DEG:.4f} deg/s  "
          f"std={ekf_bias.std()*RAD2DEG:.4f} deg/s  "
          f"min={ekf_bias.min()*RAD2DEG:.4f}  max={ekf_bias.max()*RAD2DEG:.4f}")
    print(f"  陀螺 Z 原始角速度: mean={seq_sim['gyro_z_rad'].mean()*RAD2DEG:.4f} deg/s  "
          f"std={seq_sim['gyro_z_rad'].std()*RAD2DEG:.4f} deg/s")

    # ---- 诊断：GPS 丢失段航向误差 ----
    gps_th_full = seq['gps_theta']  # 完整 GPS 航向真值
    print(f"\n[诊断] GPS 丢失段航向对比（索引 {loss_start}–{loss_end}）：")
    dr_head_err  = (dr_h[loss_start:loss_end] - gps_th_full[loss_start:loss_end]) * RAD2DEG
    ekf_head_err = (ekf_h[loss_start:loss_end] - gps_th_full[loss_start:loss_end]) * RAD2DEG
    print(f"  纯 DR  航向误差: 起点={dr_head_err[0]:.2f}°  终点={dr_head_err[-1]:.2f}°  "
          f"max={np.abs(dr_head_err).max():.2f}°")
    print(f"  EKF    航向误差: 起点={ekf_head_err[0]:.2f}°  终点={ekf_head_err[-1]:.2f}°  "
          f"max={np.abs(ekf_head_err).max():.2f}°")

    # 5. 绘图 + 指标（仅在 GPS 丢失段结束后的第一个 GPS 有效帧评估）
    print("\n[4] 绘制结果 ...")
    # 将 gps_valid 替换为完整的（用于误差评估）
    seq_plot = dict(seq_sim)
    seq_plot['gps_valid'] = seq['gps_valid_full']
    plot_results(seq_plot, dr_x, dr_y, dr_h, ekf_x, ekf_y, ekf_h, ekf_bias,
                 loss_start_idx=loss_start, loss_end_idx=loss_end)


if __name__ == '__main__':
    main()
