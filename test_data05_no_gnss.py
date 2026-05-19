"""
test_data05_no_gnss.py — Data05 间歇性 GNSS 拒止测试
====================================================

在 Data05 验证集上模拟多段隧道/地下车库场景：
  - 每段缺失 60 s
  - 段间保留约 40 s 开阔区 GNSS（可重新锚定）
  - 首段缺失前留 15 s 用于 EKF 初始化

对比纯 DR 与 BiasNet + 6-State EKF；评价仍相对 GNSS 真值。
"""

import warnings
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')
warnings.filterwarnings('ignore')

for fp in (
    'C:/Windows/Fonts/msyh.ttc',
    'C:/Windows/Fonts/simhei.ttf',
    'C:/Windows/Fonts/simsun.ttc',
):
    if Path(fp).exists():
        prop = fm.FontProperties(fname=fp)
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = [prop.get_name()] + matplotlib.rcParams['font.sans-serif']
        break
matplotlib.rcParams['axes.unicode_minus'] = False

from config import DEFAULT_EKF_CONFIG
from data_preprocessing_v2 import CALIB_VAL_ID
from ekf_navigator import (
    EKFNavigatorNP, evaluate_trajectory, load_norm_stats, simulate_gnss_outage,
)
from trajectory_data import TARGET_DT, load_calibration_segment
from validate_ekf import (
    MODEL_DIR,
    NORM_JSON,
    RAD2DEG,
    WEIGHTS_PATH,
    WINDOW_SIZE,
    run_pure_dr,
)

OUTPUT_PNG = MODEL_DIR / 'data05_no_gnss_trajectory.png'

# 间歇性拒止：60 s 缺失 + 40 s 开阔（相对段起点）
OUTAGE_DURATION_S = 60.0
GNSS_OPEN_GAP_S = 40.0
FIRST_OUTAGE_START_S = 15.0


def position_ok_mask(seq):
    gps_full = seq.get('gps_valid_full', seq['gps_valid'])
    return (
        gps_full
        & np.isfinite(seq['enu_x_truth'])
        & np.isfinite(seq['enu_y_truth'])
    )


def make_intermittent_gnss_mask(
    time_s: np.ndarray,
    pos_ok: np.ndarray,
    outage_duration_s: float = OUTAGE_DURATION_S,
    open_gap_s: float = GNSS_OPEN_GAP_S,
    first_outage_start_s: float = FIRST_OUTAGE_START_S,
):
    """
    周期性 GNSS 拒止：每段 [t, t+outage)，段间间隔 open_gap 秒恢复 GNSS。

    返回
    ----
    gps_nav, outage_mask, segments
    segments: list of dict {i0, i1, t_start, t_end}
    """
    tg = time_s
    gps_nav = pos_ok.copy()
    outage_mask = np.zeros(len(tg), dtype=bool)
    segments = []

    t0 = float(tg[0])
    t_end_rel = float(tg[-1] - t0)
    t_start = first_outage_start_s

    while t_start + outage_duration_s <= t_end_rel + 1e-6:
        t_out_end = t_start + outage_duration_s
        gps_nav, i0, i1 = simulate_gnss_outage(gps_nav, tg, t_start, t_out_end)
        outage_mask[i0:i1] = True
        segments.append({
            'i0': i0, 'i1': i1,
            't_start': t_start, 't_end': t_out_end,
            'dur_s': float(tg[min(i1, len(tg) - 1)] - tg[i0]),
        })
        t_start = t_out_end + open_gap_s

    return gps_nav, outage_mask, segments


def _first_valid_idx(truth_x, truth_y):
    ok = np.isfinite(truth_x) & np.isfinite(truth_y)
    if not ok.any():
        raise RuntimeError('无有效 GNSS 真值帧')
    return int(np.where(ok)[0][0])


def _align_origin(truth_x, truth_y, pred_x, pred_y, ref_idx):
    ox, oy = float(truth_x[ref_idx]), float(truth_y[ref_idx])
    return (
        truth_x - ox, truth_y - oy,
        pred_x - float(pred_x[ref_idx]), pred_y - float(pred_y[ref_idx]),
    )


def plot_intermittent(
    seq, dr_x, dr_y, ekf_x, ekf_y, ekf_h, net_bias,
    segments, metrics_dr, metrics_ekf,
):
    tg = seq['Time_s']
    truth_x = seq['enu_x_truth']
    truth_y = seq['enu_y_truth']
    t_rel = tg - tg[0]

    eval_mask = np.isfinite(truth_x) & np.isfinite(truth_y)
    ref = _first_valid_idx(truth_x, truth_y)
    tx, ty, drx, dry = _align_origin(truth_x, truth_y, dr_x, dr_y, ref)
    _, _, ex, ey = _align_origin(truth_x, truth_y, ekf_x, ekf_y, ref)

    idx_v = np.where(eval_mask)[0]
    dr_err = np.sqrt((drx[idx_v] - tx[idx_v]) ** 2 + (dry[idx_v] - ty[idx_v]) ** 2)
    ekf_err = np.sqrt((ex[idx_v] - tx[idx_v]) ** 2 + (ey[idx_v] - ty[idx_v]) ** 2)

    n_seg = len(segments)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        f'{CALIB_VAL_ID} 间歇 GNSS 拒止 '
        f'({OUTAGE_DURATION_S:.0f}s×{n_seg} 段, 开阔间隔 {GNSS_OPEN_GAP_S:.0f}s)',
        fontsize=13, fontweight='bold',
    )

    ax = axes[0, 0]
    ax.plot(tx[idx_v], ty[idx_v], 'g-', lw=2.0, label='GNSS 真值')
    ax.plot(drx, dry, 'r--', lw=1.0, alpha=0.75, label='纯 DR')
    ax.plot(ex, ey, 'b-', lw=1.4, label='BiasNet + EKF')
    for si, seg in enumerate(segments):
        sl = slice(seg['i0'], seg['i1'])
        ax.plot(ex[sl], ey[sl], color='darkorange', lw=2.2,
                label='拒止段' if si == 0 else None)
        ax.plot(ex[seg['i0']], ey[seg['i0']], 'go', ms=7,
                label='段入口' if si == 0 else None)
        ax.plot(ex[seg['i1'] - 1], ey[seg['i1'] - 1], 'r^', ms=7,
                label='段出口' if si == 0 else None)
    pad = 30.0
    all_x = np.concatenate([tx[idx_v], drx[idx_v], ex[idx_v]])
    all_y = np.concatenate([ty[idx_v], dry[idx_v], ey[idx_v]])
    cx, cy = np.median(all_x), np.median(all_y)
    half = max(
        np.percentile(np.abs(all_x - cx), 98),
        np.percentile(np.abs(all_y - cy), 98),
        pad,
    )
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_title('轨迹（原点 = 首帧 GNSS）')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.35)
    ax.set_aspect('equal', adjustable='box')

    ax = axes[0, 1]
    ax.plot(t_rel[idx_v], dr_err, 'r-', lw=1.2, label='DR')
    ax.plot(t_rel[idx_v], ekf_err, 'b-', lw=1.2, label='EKF')
    for seg in segments:
        ax.axvspan(t_rel[seg['i0']], t_rel[min(seg['i1'], len(t_rel) - 1)],
                   alpha=0.12, color='orange')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('位置误差 (m)')
    ax.set_title('位置误差（橙区 = GNSS 拒止）')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.35)

    ax = axes[1, 0]
    outage_frac = np.mean([seg['dur_s'] for seg in segments]) if segments else 0
    txt = (
        f'拒止段数 {n_seg} | 单段 {OUTAGE_DURATION_S:.0f}s | 间隔 {GNSS_OPEN_GAP_S:.0f}s\n'
        f'DR   RMSE {metrics_dr.get("rmse_m", float("nan")):.1f} m | '
        f'拒止最大 {metrics_dr.get("outage_max_m", float("nan")):.1f} m\n'
        f'EKF  RMSE {metrics_ekf.get("rmse_m", float("nan")):.1f} m | '
        f'拒止最大 {metrics_ekf.get("outage_max_m", float("nan")):.1f} m'
    )
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.plot(t_rel[idx_v], dr_err, 'r-', label='DR')
    ax.plot(t_rel[idx_v], ekf_err, 'b-', label='EKF')
    for seg in segments:
        ax.axvspan(t_rel[seg['i0']], t_rel[min(seg['i1'], len(t_rel) - 1)],
                   alpha=0.12, color='orange')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('误差 (m)')
    ax.set_title('误差时序')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.35)

    ax = axes[1, 1]
    ax.plot(t_rel, net_bias * RAD2DEG, 'g-', lw=0.9, label='BiasNet')
    ax.plot(t_rel, ekf_h * RAD2DEG, 'b-', lw=0.8, alpha=0.7, label='EKF 航向')
    for seg in segments:
        ax.axvspan(t_rel[seg['i0']], t_rel[min(seg['i1'], len(t_rel) - 1)],
                   alpha=0.10, color='orange')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('°/s 或 °')
    ax.set_title('零偏 & 航向')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.35)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n[图像] {OUTPUT_PNG}')


def _print_metrics(name, m, label_outage=True):
    if not m:
        return
    print(f'\n  [{name}]')
    print(f'    全程 RMSE : {m.get("rmse_m", float("nan")):.2f} m')
    print(f'    全程中值   : {m.get("median_m", float("nan")):.2f} m')
    print(f'    终点误差   : {m.get("final_m", float("nan")):.2f} m')
    if label_outage and 'outage_max_m' in m:
        print(f'    拒止段最大 : {m["outage_max_m"]:.2f} m')
        print(f'    拒止段中值 : {m.get("outage_median_m", float("nan")):.2f} m')


def main():
    print('=' * 58)
    print(f'{CALIB_VAL_ID} 间歇性 GNSS 拒止测试')
    print('=' * 58)

    if not WEIGHTS_PATH.exists():
        raise RuntimeError(f'未找到权重 {WEIGHTS_PATH}，请先运行: python train_ekf.py')

    seq = load_calibration_segment(CALIB_VAL_ID)
    if 'gps_valid_full' not in seq:
        seq['gps_valid_full'] = seq['gps_valid'].copy()

    tg = seq['Time_s']
    T = len(tg)
    duration = tg[-1] - tg[0]
    print(f'[数据] {CALIB_VAL_ID}: {T} 帧, 时长 {duration:.1f} s')

    pos_ok = position_ok_mask(seq)
    gps_nav, outage_mask, segments = make_intermittent_gnss_mask(tg, pos_ok)

    print(f'\n[拒止模式] {OUTAGE_DURATION_S:.0f}s 缺失 / {GNSS_OPEN_GAP_S:.0f}s 开阔, '
          f'首段自 {FIRST_OUTAGE_START_S:.0f}s 起')
    if not segments:
        print('  警告: 轨迹过短，未生成完整拒止段')
    for i, seg in enumerate(segments, 1):
        print(f'  段 {i}: t=[{seg["t_start"]:.0f}, {seg["t_end"]:.0f})s, '
              f'索引 {seg["i0"]}–{seg["i1"]}, 实际 {seg["dur_s"]:.1f}s')
    n_out = int(outage_mask.sum())
    print(f'  拒止帧 {n_out}/{T} ({100 * n_out / T:.1f}%), '
          f'GNSS 可用 {int(gps_nav.sum())} 帧')

    seq_nav = dict(seq)
    seq_nav['gps_valid'] = gps_nav

    print('\n[1] 纯 DR ...')
    dr_x, dr_y, dr_h = run_pure_dr(seq_nav, gps_valid_nav=gps_nav)

    print('[2] BiasNet + 6-State EKF ...')
    gx, gy = seq['enu_x_truth'], seq['enu_y_truth']
    nav = EKFNavigatorNP(
        weights_path=str(WEIGHTS_PATH),
        norm_stats=load_norm_stats(str(NORM_JSON)),
        window_size=WINDOW_SIZE,
        ekf_config=DEFAULT_EKF_CONFIG,
    )
    ekf_x, ekf_y, ekf_h, net_bias, _, _, _ = nav.run(
        imu_raw=seq['imu_raw'],
        v_ms=seq['v_ms'],
        gyro_z_rad=seq['gyro_z_rad'],
        gps_enu_x=gx,
        gps_enu_y=gy,
        gps_valid=gps_nav,
        dt=TARGET_DT,
        time_s=tg,
        gps_theta=seq['gps_theta'],
    )

    eval_mask = np.isfinite(gx) & np.isfinite(gy)
    metrics_dr = evaluate_trajectory(
        dr_x, dr_y, dr_h, gx, gy, seq['gps_theta'], eval_mask, outage_mask)
    metrics_ekf = evaluate_trajectory(
        ekf_x, ekf_y, ekf_h, gx, gy, seq['gps_theta'], eval_mask, outage_mask)

    print('\n' + '=' * 58)
    print('评估（相对 GNSS 真值）')
    print('=' * 58)
    _print_metrics('DR', metrics_dr)
    _print_metrics('BiasNet + EKF', metrics_ekf)

    plot_intermittent(
        seq, dr_x, dr_y, ekf_x, ekf_y, ekf_h, net_bias,
        segments, metrics_dr, metrics_ekf)
    print('\n完成。')


if __name__ == '__main__':
    main()
