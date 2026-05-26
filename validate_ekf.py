"""
validate_ekf.py — EKF 导航器验证脚本
=====================================

用法:
  python validate_ekf.py                                    # 默认 Data05，自动 outage
  python validate_ekf.py --data Data07                       # 指定数据集
  python validate_ekf.py --l1 120 60 --l2 200 60             # 丢1:120s起持续60s, 丢2:200s起持续60s
  python validate_ekf.py --data Data04 --l1 80 40 --l2 0 0  # 一段outage(丢2时长=0则跳过)

评估指标：
  - 分阶段位置误差（GNSS正常→丢1→恢复→丢2→恢复）
  - 轨迹对比图（GPS 真值 vs EKF 预测）
  - 航向对比图（GPS 真值 vs EKF）
"""

import argparse
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

from ekf_navigator import (
    EKFNavigatorNP, load_norm_stats, simulate_gnss_outage, evaluate_trajectory,
    enu_to_body, wrap_angle,
)
from config import DEFAULT_EKF_CONFIG

# ============================================================================
# 配置
# ============================================================================

DATA_CSV       = Path(__file__).parent / "260316_Data" / "260316_Data.csv"
MODEL_DIR      = Path(__file__).parent / "trained_models"
WEIGHTS_PATH       = MODEL_DIR / "biasnet_weights.weights.h5"
COV_WEIGHTS_PATH   = MODEL_DIR / "cov_adapter_weights.weights.h5"
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

def _seq_from_calibration(cseq):
    """将 trajectory_data.load_calibration_segment 输出转为 validate_ekf 格式。"""
    gps_v = cseq['gps_valid']
    return {
        'Time_s': cseq['Time_s'],
        'imu_raw': cseq['imu_raw'],
        'gyro_z_rad': cseq['gyro_z_rad'],
        'v_ms': cseq['v_ms'],
        'gps_theta': cseq['gps_theta'],
        'gps_valid': gps_v,
        'gps_valid_full': gps_v.copy(),
        'enu_x_truth': cseq['enu_x_truth'],
        'enu_y_truth': cseq['enu_y_truth'],
    }


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


GPS_LOSS_SIM_SECS = 60.0    # 模拟 GNSS outage 时长（秒）


def simulate_gps_loss(seq, loss_duration_s=GPS_LOSS_SIM_SECS, loss_start_s=None):
    """隧道场景：屏蔽 GNSS 位置量测（仅保留 wheel + NHC + INS）。"""
    tg = seq['Time_s']
    gps_full = seq.get('gps_valid_full', seq['gps_valid'])
    pos_ok = (
        gps_full
        & np.isfinite(seq['enu_x_truth'])
        & np.isfinite(seq['enu_y_truth'])
    )
    # 自动计算 outage 开始时间：首次运动 + 5s（给 EKF 初始化时间）
    if loss_start_s is None:
        motion_idx = np.where(seq['v_ms'] >= 0.5)[0]
        if len(motion_idx) > 0:
            loss_start_s = float(tg[motion_idx[0]] - tg[0]) + 15.0
            print(f"  [自动] 首次运动在 {float(tg[motion_idx[0]] - tg[0]):.1f}s，"
                  f"outage 从 {loss_start_s:.1f}s 开始")
        else:
            loss_start_s = 20.0

    gps_v, i0, i1 = simulate_gnss_outage(
        pos_ok, tg, loss_start_s, loss_start_s + loss_duration_s)
    i_end = min(i1, len(tg)) - 1
    dur = float(tg[i_end] - tg[i0]) if i_end >= i0 else 0.0
    print(f"  [模拟 GNSS outage] 索引 {i0}–{i1}，持续 {dur:.1f} s")
    return gps_v, i0, i1


# ============================================================================
# 纯死推算（基准）
# ============================================================================

def run_pure_dr(seq, gps_valid_nav=None):
    """
    纯陀螺积分（无 BiasNet）+ 轮速 NHC 位置累积；
    outage 段不锚定 GNSS（模拟无融合 DR）。
    """
    T = len(seq['Time_s'])
    dt = TARGET_DT
    gyro_z = seq['gyro_z_rad']
    v_ms = seq['v_ms']
    gps_th = seq['gps_theta']
    gps_v = gps_valid_nav if gps_valid_nav is not None else seq['gps_valid']
    gx = seq['enu_x_truth']
    gy = seq['enu_y_truth']

    pos_ok = gps_v & np.isfinite(gx) & np.isfinite(gy)
    first = int(np.where(pos_ok)[0][0]) if pos_ok.any() else 0

    heading = np.zeros(T, np.float32)
    heading[first] = float(gps_th[first]) if np.isfinite(gps_th[first]) else 0.0
    px, py = float(gx[first]), float(gy[first])
    xs = np.full(T, np.nan, np.float32)
    ys = np.full(T, np.nan, np.float32)
    xs[first], ys[first] = px, py

    freeze_v = DEFAULT_EKF_CONFIG.freeze_yaw_below_ms
    for k in range(first + 1, T):
        vw = float(v_ms[k]) if np.isfinite(v_ms[k]) else 0.0
        if pos_ok[k]:
            heading[k] = float(gps_th[k])
            px, py = float(gx[k]), float(gy[k])
        else:
            if vw >= freeze_v:
                heading[k] = heading[k - 1] + float(gyro_z[k]) * dt
            else:
                heading[k] = heading[k - 1]
            px += vw * np.cos(heading[k]) * dt
            py += vw * np.sin(heading[k]) * dt
        xs[k], ys[k] = px, py

    xs[:first] = xs[first]
    ys[:first] = ys[first]
    return xs, ys, heading


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

def _align_origin(truth_x, truth_y, pred_x, pred_y, ref_idx):
    ox = float(truth_x[ref_idx])
    oy = float(truth_y[ref_idx])
    return (truth_x - ox, truth_y - oy,
            pred_x - float(pred_x[ref_idx]), pred_y - float(pred_y[ref_idx]))


def plot_results(seq, dr_x, dr_y, dr_h,
                 ekf_x, ekf_y, ekf_h, net_bias, ekf_bg,
                 ekf_vx, ekf_vy,
                 ekf_bg_vec=None, ekf_ba_vec=None,
                 ekf_vel_scale=None,
                 outage_segments=None,  # [(start1,end1), (start2,end2), ...]
                 metrics_dr=None, metrics_ekf=None,
                 tag=''):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_str = f"_{tag}" if tag else ""
    tg = seq['Time_s']
    truth_x = seq['enu_x_truth']
    truth_y = seq['enu_y_truth']
    t_rel = tg - tg[0]

    eval_mask = np.isfinite(truth_x) & np.isfinite(truth_y)
    ref = int(np.where(eval_mask)[0][0])
    tx, ty, drx, dry = _align_origin(truth_x, truth_y, dr_x, dr_y, ref)
    _, _, ex, ey = _align_origin(truth_x, truth_y, ekf_x, ekf_y, ref)

    idx_v = np.where(eval_mask)[0]
    dr_err = np.sqrt((drx[idx_v] - tx[idx_v]) ** 2 + (dry[idx_v] - ty[idx_v]) ** 2)
    ekf_err = np.sqrt((ex[idx_v] - tx[idx_v]) ** 2 + (ey[idx_v] - ty[idx_v]) ** 2)


    # 标记 outage 区域 — 直接从参数传入
    outage_masks = outage_segments if outage_segments else []
    T_plot = len(t_rel)
    # ======================================================================
    # Figure 1：轨迹对比（大地图，重点突出两条预测线）
    # ======================================================================
    fig1, ax1 = plt.subplots(1, 1, figsize=(14, 14))
    fig1.suptitle(f'EKF 轨迹验证 ({timestamp})', fontsize=14, fontweight='bold')

    # GNSS 真值 — 灰色细线作参考
    ax1.plot(tx[idx_v], ty[idx_v], color='gray', lw=1.0, alpha=0.6, label='GNSS 真值', zorder=2)

    # EKF — 蓝色实线，加粗（全段）
    ax1.plot(ex, ey, 'b-', lw=2.2, alpha=0.95, label='EKF 预测轨迹', zorder=4)

    # Outage 段高亮 — 鲜绿粗线 + 起止标记
    for oi, (os_i, oe_i) in enumerate(outage_masks):
        oe_i = min(oe_i, T_plot - 1)
        # EKF outage 段 — 鲜绿粗线
        ax1.plot(ex[os_i:oe_i+1], ey[os_i:oe_i+1], color='#00cc00', lw=6.0, alpha=0.9,
                 label=f'GNSS 丢失段{oi+1} ({tg[os_i]:.0f}–{tg[oe_i]:.0f}s)', zorder=7)
        # 起点 — 大绿圆，终点 — 大红三角
        ax1.scatter([ex[os_i]], [ey[os_i]], c='#00ff00', s=200, marker='o',
                    edgecolors='black', linewidths=1.5, zorder=10)
        ax1.scatter([ex[oe_i]], [ey[oe_i]], c='red', s=250, marker='^',
                    edgecolors='black', linewidths=1.5, zorder=10)
        # 文字标注
        ax1.annotate(f'丢{oi+1} 起点\n{tg[os_i]:.0f}s', (ex[os_i], ey[os_i]),
                     xytext=(15, 20), textcoords='offset points',
                     fontsize=10, color='#00aa00', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='green', lw=2))
        ax1.annotate(f'丢{oi+1} 终点\n{tg[oe_i]:.0f}s', (ex[oe_i], ey[oe_i]),
                     xytext=(15, -25), textcoords='offset points',
                     fontsize=10, color='red', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # 自适应视野
    pad = 25.0
    all_x = np.concatenate([tx[idx_v], ex[idx_v]])
    all_y = np.concatenate([ty[idx_v], ey[idx_v]])
    cx, cy = np.median(all_x), np.median(all_y)
    half = max(np.percentile(np.abs(all_x - cx), 98),
               np.percentile(np.abs(all_y - cy), 98), pad)
    ax1.set_xlim(cx - half, cx + half)
    ax1.set_ylim(cy - half, cy + half)
    ax1.set_xlabel('East (m)', fontsize=12)
    ax1.set_ylabel('North (m)', fontsize=12)
    ax1.set_title('EKF 预测轨迹 — 灰色=GNSS真值  蓝色=EKF  绿色粗线=丢失段 ●起点 ▲终点', fontsize=12)
    ax1.legend(fontsize=10, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    out_path1 = OUTPUT_DIR / f"ekf_trajectory_{timestamp}{tag_str}.png"
    fig1.savefig(out_path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"[轨迹图] {out_path1}")

    # ======================================================================
    # Figure 2：时序诊断（误差 / 航向 / 零偏）
    # ======================================================================
    fig2, axes2 = plt.subplots(3, 1, figsize=(16, 13), sharex=True)
    fig2.suptitle(f'EKF 时序诊断 ({timestamp})', fontsize=14, fontweight='bold')

    # (a) 位置误差
    ax = axes2[0]
    ax.plot(t_rel[idx_v], dr_err, 'r-', lw=1.2, label='DR 位置误差')
    ax.plot(t_rel[idx_v], ekf_err, 'b-', lw=1.5, label='EKF 位置误差')
    for oi, (os_i, oe_i) in enumerate(outage_masks):
        oe_i = min(oe_i, T_plot - 1)
        ax.axvspan(t_rel[os_i], t_rel[oe_i], alpha=0.12, color='lime',
                   label=f'Outage {oi+1}')
    ax.set_ylabel('位置误差 (m)')
    ax.legend(fontsize=9, loc='upper left', ncol=4)
    ax.grid(True, alpha=0.35)
    # 统计标注
    ax.text(0.02, 0.95,
            f"DR  RMSE={np.sqrt(np.mean(dr_err**2)):.1f}m  "
            f"max={np.max(dr_err):.1f}m\n"
            f"EKF RMSE={np.sqrt(np.mean(ekf_err**2)):.1f}m  "
            f"max={np.max(ekf_err):.1f}m",
            transform=ax.transAxes, fontsize=8.5, va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # (b) 航向
    ax = axes2[1]
    moving = seq['v_ms'] >= 0.5
    gps_head = np.where(eval_mask & moving, seq['gps_theta'] * RAD2DEG, np.nan)
    dr_h_plot = np.where(moving, dr_h * RAD2DEG, np.nan)
    ekf_h_plot = np.where(moving, ekf_h * RAD2DEG, np.nan)
    ax.plot(t_rel, gps_head, 'k-', lw=0.8, alpha=0.5, label='GNSS 航向')
    ax.plot(t_rel, dr_h_plot, 'r--', lw=1.0, alpha=0.7, label='DR 航向')
    ax.plot(t_rel, ekf_h_plot, 'b-', lw=1.5, label='EKF 航向')
    for oi, (os_i, oe_i) in enumerate(outage_masks):
        oe_i = min(oe_i, T_plot - 1)
        ax.axvspan(t_rel[os_i], t_rel[oe_i], alpha=0.12, color='lime')
    ax.set_ylabel('航向 (°)')
    ax.legend(fontsize=9, loc='upper left', ncol=3)
    ax.grid(True, alpha=0.35)

    # (c) 零偏
    ax = axes2[2]
    nb_z_rad = net_bias[:, 5] * DEG2RAD if net_bias.ndim == 2 else net_bias * RAD2DEG
    ax.plot(t_rel, nb_z_rad * RAD2DEG, 'b-', lw=1.2, label='BiasNet 预测')
    ax.plot(t_rel, ekf_bg * RAD2DEG, 'r-', lw=0.8, alpha=0.7, label='EKF 残差')
    ax.plot(t_rel, nb_z_rad * RAD2DEG + ekf_bg * RAD2DEG, 'g-', lw=0.6, label='合计零偏')
    ax.axhline(0, color='k', ls='--', lw=0.4)
    for oi, (os_i, oe_i) in enumerate(outage_masks):
        oe_i = min(oe_i, T_plot - 1)
        ax.axvspan(t_rel[os_i], t_rel[oe_i], alpha=0.12, color='lime')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('零偏 (°/s)')
    ax.legend(fontsize=9, loc='upper left', ncol=3)
    ax.grid(True, alpha=0.35)
    ax.text(0.98, 0.97,
            f"BiasNet μ={np.mean(nb_z_rad*RAD2DEG):.3f} σ={np.std(nb_z_rad*RAD2DEG):.3f}°/s\n"
            f"EKF bg  μ={np.mean(ekf_bg*RAD2DEG):.4f} σ={np.std(ekf_bg*RAD2DEG):.4f}°/s",
            transform=ax.transAxes, fontsize=7.5, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    out_path2 = OUTPUT_DIR / f"ekf_diagnosis_{timestamp}{tag_str}.png"
    fig2.savefig(out_path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"[诊断图] {out_path2}")

    print("\n" + "=" * 60)
    print("评估指标（相对 GNSS 真值）")
    print("=" * 60)
    for name, m in [('DR', metrics_dr), ('EKF', metrics_ekf)]:
        if not m:
            continue
        print(f"\n  [{name}]")
        print(f"    RMSE       : {m.get('rmse_m', float('nan')):.2f} m")
        print(f"    中值误差   : {m.get('median_m', float('nan')):.2f} m")
        print(f"    终点误差   : {m.get('final_m', float('nan')):.2f} m")
        if 'outage_max_m' in m:
            print(f"    Outage 最大: {m['outage_max_m']:.2f} m")
            print(f"    Outage 中值: {m.get('outage_median_m', float('nan')):.2f} m")
        if 'heading_rmse_deg' in m:
            print(f"    航向 RMSE  : {m['heading_rmse_deg']:.2f} °")
            if 'heading_rmse_no_outage_deg' in m:
                print(f"    航向 RMSE(GPS有效): {m['heading_rmse_no_outage_deg']:.2f} °")
            if 'heading_rmse_outage_deg' in m:
                print(f"    航向 RMSE(outage):  {m['heading_rmse_outage_deg']:.2f} °")
    print("=" * 60)


def plot_ekf_diagnostics(seq, vel_x, vel_y, headings, net_bias, loss_start, loss_end, tag=''):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_str = f"_{tag}" if tag else ""
    out_name = f"ekf_diagnostics_{timestamp}{tag_str}.png"
    """
    速度诊断图（Figure 4）
      - 4A：Speed Magnitude Comparison — wheel speed vs EKF reconstructed speed
      - 4B：Lateral Velocity Diagnostic — body frame v_lat sanity check
    """
    tg = seq['Time_s']
    t_rel = tg - tg[0]

    # EKF reconstructed speed
    ekf_spd = np.sqrt(vel_x ** 2 + vel_y ** 2)

    # Body-frame velocities
    v_fwd = np.zeros(len(tg), np.float32)
    v_lat = np.zeros(len(tg), np.float32)
    for k in range(len(tg)):
        v_fwd[k], v_lat[k] = enu_to_body(float(vel_x[k]), float(vel_y[k]), float(headings[k]))

    # ======================================================================
    # Figure 4：速度诊断
    # ======================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    fig.suptitle(f'Velocity Diagnostic ({timestamp})', fontsize=13, fontweight='bold')

    # ====== 4A: Speed Magnitude Comparison ======
    ax = axes[0]
    ax.plot(t_rel, seq['v_ms'], 'g-', lw=1.2, alpha=0.8, label='Wheel speed (测量)')
    ax.plot(t_rel, ekf_spd, 'b-', lw=1.2, alpha=0.8, label='EKF speed (√(vx²+vy²))')
    if loss_start is not None:
        ax.axvspan(t_rel[loss_start], t_rel[min(loss_end, len(t_rel) - 1)],
                   alpha=0.1, color='orange', label='GNSS Outage')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('4A：Speed Magnitude Comparison', fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t_rel[0], t_rel[-1])

    # ====== 4B: Lateral Velocity Diagnostic ======
    ax = axes[1]
    ax.plot(t_rel, v_lat, 'r-', lw=1.0)
    ax.axhline(0, color='k', ls='--', lw=0.6)
    # 固定 y 轴范围防止炸裂
    vlat_lim = max(np.percentile(np.abs(v_lat), 99) * 1.5, 0.15)
    ax.set_ylim(-vlat_lim, vlat_lim)
    if loss_start is not None:
        ax.axvspan(t_rel[loss_start], t_rel[min(loss_end, len(t_rel) - 1)],
                   alpha=0.1, color='orange', label='GNSS Outage')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('v_lat (m/s)')
    ax.set_title('4B：Lateral Velocity Diagnostic', fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t_rel[0], t_rel[-1])

    plt.tight_layout()
    out = OUTPUT_DIR / out_name
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[诊断图] {out}")

    # ======================================================================
    # 统计数据输出
    # ======================================================================
    wheel_spd = seq['v_ms']
    print(f"  EKF speed : mean={np.mean(ekf_spd):.3f} m/s, std={np.std(ekf_spd):.3f} m/s")
    print(f"  Wheel spd : mean={np.mean(wheel_spd):.3f} m/s, std={np.std(wheel_spd):.3f} m/s")
    print(f"  v_lat     : mean={np.mean(v_lat):.5f} m/s, std={np.std(v_lat):.3f} m/s, "
          f"max|.|={np.max(np.abs(v_lat)):.3f} m/s")


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='EKF 导航器验证')
    parser.add_argument('--data', type=str, default=None,
                        help='数据集ID (如 Data05, Data07)，默认使用 VAL_DATASET_ID')
    parser.add_argument('--l1', nargs=2, type=float, default=None,
                        metavar=('START_S', 'DUR_S'),
                        help='第一段 outage：起始时间(s) 和 持续时长(s)')
    parser.add_argument('--l2', nargs=2, type=float, default=None,
                        metavar=('START_S', 'DUR_S'),
                        help='第二段 outage：起始时间(s) 和 持续时长(s)，dur=0 则跳过')
    args = parser.parse_args()

    # 1. 加载测试数据
    dataset_id = args.data
    if dataset_id is None:
        try:
            from trajectory_data import VAL_DATASET_ID
            dataset_id = VAL_DATASET_ID
        except Exception:
            dataset_id = 'Data05'
    print("=" * 55)
    print(f"EKF 导航器验证 — 数据集: {dataset_id}")
    print("=" * 55)

    try:
        from trajectory_data import load_calibration_segment
        print(f"[数据] 加载标定验证集 {dataset_id} ...")
        seq = _seq_from_calibration(load_calibration_segment(dataset_id))
    except Exception:
        seq = load_test_segment()

    if 'gps_valid_full' not in seq:
        seq['gps_valid_full'] = seq['gps_valid'].copy()

    # ENU 坐标归零：以首个 GPS 有效帧（非插值）为原点
    gps_ok = seq['gps_valid_full'] if 'gps_valid_full' in seq else seq['gps_valid']
    ok_init = gps_ok & np.isfinite(seq['enu_x_truth']) & np.isfinite(seq['enu_y_truth'])
    if ok_init.any():
        i0 = int(np.where(ok_init)[0][0])
        x0, y0 = float(seq['enu_x_truth'][i0]), float(seq['enu_y_truth'][i0])
        seq['enu_x_truth'] = seq['enu_x_truth'].copy() - x0
        seq['enu_y_truth'] = seq['enu_y_truth'].copy() - y0
        # 首个有效GPS之前的插值填充帧标记为无效（坐标无意义）
        seq['gps_valid_full'][:i0] = False

    # 2. 模拟两段 GNSS outage（真实场景：GNSS正常→丢失1→恢复→丢失2→恢复）
    tg = seq['Time_s']
    T = len(tg)
    # 用全量 GPS 构建初始 gps_valid
    gps_ok_full = seq['gps_valid_full'] if 'gps_valid_full' in seq else seq['gps_valid']
    gps_v_full = gps_ok_full & np.isfinite(seq['enu_x_truth']) & np.isfinite(seq['enu_y_truth'])

    # 两段 outage — 命令行指定 或 自动适配
    total_dur = float(tg[-1] - tg[0])
    if args.l1 is not None:
        loss1_start_s, loss1_dur_s = args.l1[0], args.l1[1]
    else:
        # 自动：确保丢1前EKF已充分收敛
        loss1_start_s = max(100.0, total_dur * 0.45)
        loss1_dur_s   = min(60.0, (total_dur - loss1_start_s) * 0.40)
    if args.l2 is not None:
        loss2_start_s, loss2_dur_s = args.l2[0], args.l2[1]
    else:
        loss2_start_s = loss1_start_s + loss1_dur_s + max(20.0, total_dur * 0.08)
        loss2_start_s = min(loss2_start_s, total_dur - 20.0)
        loss2_dur_s   = min(60.0, total_dur - loss2_start_s - 5.0)

    print(f"\n[1] GNSS outage 配置:")
    gps_v_sim = gps_v_full.copy()
    outage_segs = []
    for name, start_s, dur_s in [("丢失1", loss1_start_s, loss1_dur_s),
                                   ("丢失2", loss2_start_s, loss2_dur_s)]:
        if dur_s <= 0:
            print(f"  {name}: 跳过（时长={dur_s:.0f}s）")
            continue
        gps_v_sim, i0, i1 = simulate_gnss_outage(gps_v_sim, tg, start_s, start_s + dur_s)
        i1 = min(i1, T - 1)
        print(f"  {name}: {tg[i0]:.0f}s–{tg[i1]:.0f}s ({tg[i1]-tg[i0]:.1f}s)"
              f"  索引 {i0}–{i1}")
        outage_segs.append((i0, i1))

    seq_sim = dict(seq)
    seq_sim['gps_valid'] = gps_v_sim

    # ---- 全局轮速比例修正 ----
    gx_full = seq['enu_x_truth']
    gy_full = seq['enu_y_truth']
    vw_raw = seq_sim['v_ms']

    gps_spd = np.zeros(T, dtype=np.float32)
    dt_gps = np.clip(np.diff(tg, prepend=tg[0]), 0.05, 0.5)
    for i in range(1, T):
        if gps_ok_full[i] and gps_ok_full[i - 1]:
            dx = float(gx_full[i] - gx_full[i - 1])
            dy = float(gy_full[i] - gy_full[i - 1])
            gps_spd[i] = np.sqrt(dx * dx + dy * dy) / dt_gps[i]

    mask = gps_ok_full & (vw_raw > 1.0) & (gps_spd > 1.0)
    if mask.sum() > 50:
        scale = float(np.median(gps_spd[mask] / vw_raw[mask]))
        scale = np.clip(scale, 0.85, 1.15)
    else:
        scale = 1.0

    seq_sim['v_ms'] = vw_raw * scale
    print(f"  [轮速修正] GPS/轮速 全局中值 = {scale:.4f}  "
          f"(均值 {seq['v_ms'].mean():.2f} → {seq_sim['v_ms'].mean():.2f} m/s)")

    outage_mask = np.zeros(T, dtype=bool)
    for s, e in outage_segs:
        outage_mask[s:e] = True

    # 3. DR 基准
    print("\n[2] 运行 DR 基准（outage 段无 GNSS 锚定）...")
    dr_x, dr_y, dr_h = run_pure_dr(seq_sim, gps_valid_nav=gps_v_sim)

    # 4. 6-State EKF
    print("\n[3] 运行 BiasNet + 6-State EKF ...")
    if not WEIGHTS_PATH.exists():
        raise RuntimeError(f"未找到权重 {WEIGHTS_PATH}，请先: python train_ekf.py")

    norm_stats = load_norm_stats(str(NORM_JSON))
    nav = EKFNavigatorNP(
        weights_path=str(WEIGHTS_PATH),
        norm_stats=norm_stats,
        window_size=WINDOW_SIZE,
        ekf_config=DEFAULT_EKF_CONFIG,
        cov_weights_path=str(COV_WEIGHTS_PATH) if COV_WEIGHTS_PATH.exists() else None,
    )

    gx = seq['enu_x_truth']
    gy = seq['enu_y_truth']
    (ekf_x, ekf_y, ekf_h, net_bias, ekf_vx, ekf_vy, ekf_bgz,
     ekf_bg_vec, ekf_ba_vec, ekf_vel_scale) = nav.run(
        imu_raw=seq_sim['imu_raw'],
        v_ms=seq_sim['v_ms'],
        gyro_z_rad=seq_sim['gyro_z_rad'],
        gps_enu_x=gx,
        gps_enu_y=gy,
        gps_valid=gps_v_sim,
        dt=TARGET_DT,
        time_s=seq['Time_s'],
        gps_theta=seq_sim['gps_theta'],
    )

    # 向后兼容：net_bias_z 和 ekf_bg 使用 gyro_z 通道
    net_bias_z = net_bias[:, 5] * DEG2RAD            # deg/s → rad/s
    ekf_bg = ekf_bgz  # alias for old code

    eval_mask = (np.isfinite(gx) & np.isfinite(gy)
                 & seq['gps_valid_full'])  # 排除 GPS 无效的插值帧
    # 航向真值：静止时 GPS 航向不可靠（位置位移噪声），仅当运动时参与评估
    truth_yaw = seq['gps_theta'].copy()
    still = seq['v_ms'] < 0.5
    truth_yaw[still] = np.nan
    metrics_dr = evaluate_trajectory(
        dr_x, dr_y, dr_h, gx, gy, truth_yaw, eval_mask, outage_mask)
    metrics_ekf = evaluate_trajectory(
        ekf_x, ekf_y, ekf_h, gx, gy, truth_yaw, eval_mask, outage_mask)

    # ---- 分阶段指标 ----
    def _phase_rmse(px, py, phase_mask):
        idx = np.where(eval_mask & phase_mask)[0]
        if len(idx) == 0:
            return float('nan')
        err = np.sqrt((px[idx] - gx[idx]) ** 2 + (py[idx] - gy[idx]) ** 2)
        return float(np.sqrt(np.mean(err ** 2)))

    # 构建阶段 mask
    phase_masks = []
    prev_end = 0
    for si, (s, e) in enumerate(outage_segs):
        if s > prev_end:
            phase_masks.append((f'GNSS段{si+1}', slice(prev_end, s)))
        phase_masks.append((f'丢失{si+1}', slice(s, e)))
        prev_end = e
    if prev_end < T:
        phase_masks.append((f'GNSS段{len(outage_segs)+1}', slice(prev_end, T)))

    print(f"\n[3.5] 分阶段评估 ({len(outage_segs)}段outage)")
    for name, px, py in [('DR', dr_x, dr_y), ('EKF', ekf_x, ekf_y)]:
        parts = []
        for label, sl in phase_masks:
            pm = np.zeros(T, dtype=bool); pm[sl] = True
            r = _phase_rmse(px, py, pm)
            parts.append(f'{label}={r:.1f}m')
        print(f"  [{name}] {' | '.join(parts)}")
    # ----------------------------------------------------------------

    print("\n[4] 绘制结果 ...")
    plot_results(
        seq, dr_x, dr_y, dr_h, ekf_x, ekf_y, ekf_h, net_bias, ekf_bg,
        ekf_vx, ekf_vy,
        ekf_bg_vec=ekf_bg_vec, ekf_ba_vec=ekf_ba_vec,
        ekf_vel_scale=ekf_vel_scale,
        outage_segments=outage_segs,
        metrics_dr=metrics_dr, metrics_ekf=metrics_ekf,
        tag=dataset_id,
    )
    print(f"\n  轮速比例因子 vel_scale: mean={ekf_vel_scale.mean():.4f}  "
          f"final={ekf_vel_scale[-1]:.4f}  "
          f"(1.0=无修正, <1.0=轮速偏大)")

    save_validation_log(metrics_dr, metrics_ekf, outage_segs, tg, dataset_id)


def save_validation_log(metrics_dr, metrics_ekf, outage_segs, tg, dataset_id=''):
    """生成带时间戳的验证日志，并追加汇总到 SUMMARY.md。"""
    import json
    from datetime import datetime
    from config import DEFAULT_EKF_CONFIG

    log_dir = Path(__file__).parent / "training_logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    outage_info = []
    for i, (s, e) in enumerate(outage_segs):
        dur = float(tg[min(e, len(tg)-1)] - tg[s]) if s < len(tg) else 0
        outage_info.append({'seg': i+1, 'start_idx': int(s), 'end_idx': int(min(e, len(tg)-1)),
                            'start_s': float(tg[s]), 'end_s': float(tg[min(e, len(tg)-1)]),
                            'duration_s': dur})

    log = {
        'timestamp': timestamp,
        'dataset': dataset_id,
        'config': {
            'q_yaw': DEFAULT_EKF_CONFIG.q_yaw, 'q_vel': DEFAULT_EKF_CONFIG.q_vel,
            'q_bg': DEFAULT_EKF_CONFIG.q_bg, 'q_pos': DEFAULT_EKF_CONFIG.q_pos,
            'r_gps_xy': DEFAULT_EKF_CONFIG.r_gps_xy, 'r_wheel': DEFAULT_EKF_CONFIG.r_wheel,
            'r_nhc': DEFAULT_EKF_CONFIG.r_nhc,
            'nhc_r_scale_turn': DEFAULT_EKF_CONFIG.nhc_r_scale_turn,
        },
        'outage_segments': outage_info,
        'dr_metrics': {
            'rmse_m': metrics_dr.get('rmse_m', None),
            'median_m': metrics_dr.get('median_m', None),
            'final_m': metrics_dr.get('final_m', None),
            'max_m': metrics_dr.get('max_m', None),
            'heading_rmse_deg': metrics_dr.get('heading_rmse_deg', None),
        } if metrics_dr else None,
        'ekf_metrics': {
            'rmse_m': metrics_ekf.get('rmse_m', None),
            'median_m': metrics_ekf.get('median_m', None),
            'final_m': metrics_ekf.get('final_m', None),
            'max_m': metrics_ekf.get('max_m', None),
            'heading_rmse_deg': metrics_ekf.get('heading_rmse_deg', None),
            'heading_rmse_no_outage_deg': metrics_ekf.get('heading_rmse_no_outage_deg', None),
            'heading_rmse_outage_deg': metrics_ekf.get('heading_rmse_outage_deg', None),
            'outage_max_m': metrics_ekf.get('outage_max_m', None),
            'outage_median_m': metrics_ekf.get('outage_median_m', None),
            'outage_final_m': metrics_ekf.get('outage_final_m', None),
        } if metrics_ekf else None,
    }

    log_path = log_dir / f"validate_{timestamp}.json"
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\n[日志] 已保存到 {log_path}")

    # ---- 追加汇总到 SUMMARY_VAL.md ----
    summary_path = log_dir / "SUMMARY_VAL.md"
    if not summary_path.exists():
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("# 验证记录汇总\n\n")
            f.write("| 时间 | RMSE(m) | Outage最大(m) | Outage中值(m) |"
                    " 航向RMSE(°) | 航向(GPS有效)(°) | 航向(outage)(°) |"
                    " q_vel | r_nhc | r_gps_hdg |\n")
            f.write("|------|---------|---------------|---------------|"
                    "--------------|--------------------|-------------------|"
                    "-------|-------|------------|\n")

    m = log['ekf_metrics'] or {}
    dt_str = datetime.now().strftime("%m-%d %H:%M")
    c = log['config']
    line = (
        f"| {dt_str} "
        f"| {m.get('rmse_m', '—'):<7.2f} "
        f"| {m.get('outage_max_m', '—'):<13.2f} "
        f"| {m.get('outage_median_m', '—'):<13.2f} "
        f"| {m.get('heading_rmse_deg', '—'):<12.2f} "
        f"| {m.get('heading_rmse_no_outage_deg', '—'):<18.2f} "
        f"| {m.get('heading_rmse_outage_deg', '—'):<17.2f} "
        f"| {c['q_vel']:.2e} | {c['r_nhc']:.2e} | {c['r_gps_xy']:.2e} |\n"
    )
    with open(summary_path, 'a', encoding='utf-8') as f:
        f.write(line)
    print(f"[汇总] 已追加到 {summary_path}")
    return log_path


if __name__ == '__main__':
    main()
