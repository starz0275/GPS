"""
test_data05_no_gnss.py — Data05 全程无 GPS 推理测试
==================================================

推理阶段完全不使用 GNSS（位置、航向、初始化均不读 GPS）：
  - 起点 (0, 0)，航向 0，零偏由静止段陀螺中值估计
  - BiasNet + EKF 仅依赖 IMU、轮速、NHC

纯 DR 基准同样不用 GPS。评价仍相对 GNSS 真值（仅离线对比，不参与融合）。
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
    EKFNavigatorNP, body_to_enu, evaluate_trajectory, load_norm_stats, wrap_angle,
)
from trajectory_data import TARGET_DT, load_calibration_segment
from validate_ekf import (
    MODEL_DIR,
    NORM_JSON,
    RAD2DEG,
    WEIGHTS_PATH,
    WINDOW_SIZE,
)

OUTPUT_PNG = MODEL_DIR / 'data05_no_gnss_trajectory.png'


class EKFNavigatorNoGPS(EKFNavigatorNP):
    """推理时不使用任何 GNSS 信息做初始化或量测更新。"""

    def _init_state(self, gps_x, gps_y, gps_v, v_ms, gps_theta, gyro_z, net_bias,
                    dt=0.1, time_s=None):
        cfg = self.ekf_config
        min_wheel_ms = cfg.min_speed_wheel_ms

        px0, py0 = 0.0, 0.0
        yaw0 = 0.0
        bg0 = 0.0

        still = (np.abs(v_ms) < 0.1) & np.isfinite(gyro_z)
        still_idx = np.where(still)[0]
        if len(still_idx) >= 10 and np.any(np.diff(still_idx) <= 3):
            bg0 = float(np.median(gyro_z[still_idx]))
            bg0 = np.clip(bg0, -cfg.bg_init_max_bg_rads, cfg.bg_init_max_bg_rads)

        motion = np.where(np.isfinite(v_ms) & (v_ms >= min_wheel_ms))[0]
        k0 = int(motion[0]) if len(motion) > 0 else 0

        vk = float(v_ms[k0]) if np.isfinite(v_ms[k0]) else 0.0
        vx0, vy0 = body_to_enu(vk, 0.0, yaw0)
        return np.array([px0, py0, vx0, vy0, wrap_angle(yaw0), bg0]), k0


def run_pure_dr_no_gps(seq):
    """纯陀螺 + 轮速 DR，起点 (0,0)、航向 0，不用 GPS。"""
    T = len(seq['Time_s'])
    dt = TARGET_DT
    gyro_z = seq['gyro_z_rad']
    v_ms = seq['v_ms']
    min_v = DEFAULT_EKF_CONFIG.min_speed_wheel_ms

    motion = np.where(np.isfinite(v_ms) & (v_ms >= min_v))[0]
    first = int(motion[0]) if len(motion) > 0 else 0

    heading = np.zeros(T, np.float32)
    px, py = 0.0, 0.0
    xs = np.full(T, np.nan, np.float32)
    ys = np.full(T, np.nan, np.float32)
    xs[first], ys[first] = px, py

    freeze_v = DEFAULT_EKF_CONFIG.freeze_yaw_below_ms
    for k in range(first + 1, T):
        vw = float(v_ms[k]) if np.isfinite(v_ms[k]) else 0.0
        if vw >= freeze_v:
            heading[k] = heading[k - 1] + float(gyro_z[k]) * dt
            px += vw * np.cos(heading[k]) * dt
            py += vw * np.sin(heading[k]) * dt
        else:
            heading[k] = heading[k - 1]
        xs[k], ys[k] = px, py

    xs[:first] = xs[first]
    ys[:first] = ys[first]
    heading[:first] = heading[first]
    return xs, ys, heading


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


def plot_no_gps(seq, dr_x, dr_y, ekf_x, ekf_y, ekf_h, net_bias, metrics_dr, metrics_ekf):
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

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        f'{CALIB_VAL_ID} 推理全程无 GPS — BiasNet + EKF vs DR',
        fontsize=13, fontweight='bold',
    )

    ax = axes[0, 0]
    ax.plot(tx[idx_v], ty[idx_v], 'g-', lw=2.0, label='GNSS 真值（仅评价）')
    ax.plot(drx, dry, 'r--', lw=1.0, alpha=0.8, label='纯 DR（无 GPS）')
    ax.plot(ex, ey, 'b-', lw=1.4, label='BiasNet + EKF（无 GPS）')
    ax.plot(ex[ref], ey[ref], 'ko', ms=7, label='推理起点 (0,0)')
    ax.plot(ex[idx_v[-1]], ey[idx_v[-1]], 'r^', ms=8, label='EKF 终点')
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
    ax.set_title('轨迹（与真值首帧原点对齐）')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.35)
    ax.set_aspect('equal', adjustable='box')

    ax = axes[0, 1]
    ax.plot(t_rel[idx_v], dr_err, 'r-', lw=1.2, label='DR')
    ax.plot(t_rel[idx_v], ekf_err, 'b-', lw=1.2, label='EKF')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('位置误差 (m)')
    ax.set_title('相对 GNSS 真值的位置误差')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.35)

    ax = axes[1, 0]
    duration_s = t_rel[idx_v[-1]] - t_rel[idx_v[0]]
    txt = (
        f'时长 {duration_s:.0f} s | 推理无 GPS\n'
        f'DR   RMSE {metrics_dr.get("rmse_m", float("nan")):.1f} m | '
        f'终点 {metrics_dr.get("final_m", float("nan")):.1f} m\n'
        f'EKF  RMSE {metrics_ekf.get("rmse_m", float("nan")):.1f} m | '
        f'终点 {metrics_ekf.get("final_m", float("nan")):.1f} m'
    )
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.plot(t_rel[idx_v], dr_err, 'r-', label='DR')
    ax.plot(t_rel[idx_v], ekf_err, 'b-', label='EKF')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('误差 (m)')
    ax.set_title('全程漂移')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.35)

    ax = axes[1, 1]
    nb_z = net_bias[:, 5] * DEG2RAD if net_bias.ndim == 2 else net_bias * RAD2DEG
    ax.plot(t_rel, nb_z * RAD2DEG, 'g-', lw=0.9, label='BiasNet')
    ax.plot(t_rel, ekf_h * RAD2DEG, 'b-', lw=0.8, alpha=0.7, label='EKF 航向')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('°/s 或 °')
    ax.set_title('零偏 & 航向')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.35)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n[图像] {OUTPUT_PNG}')


def _print_metrics(name, m):
    if not m:
        return
    print(f'\n  [{name}]')
    print(f'    RMSE     : {m.get("rmse_m", float("nan")):.2f} m')
    print(f'    中值误差 : {m.get("median_m", float("nan")):.2f} m')
    print(f'    最大误差 : {m.get("max_m", float("nan")):.2f} m')
    print(f'    终点误差 : {m.get("final_m", float("nan")):.2f} m')


def main():
    print('=' * 58)
    print(f'{CALIB_VAL_ID} 推理全程无 GPS 测试')
    print('=' * 58)

    if not WEIGHTS_PATH.exists():
        raise RuntimeError(f'未找到权重 {WEIGHTS_PATH}，请先运行: python train_ekf.py')

    seq = load_calibration_segment(CALIB_VAL_ID)
    tg = seq['Time_s']
    T = len(tg)
    print(f'[数据] {CALIB_VAL_ID}: {T} 帧, 时长 {tg[-1] - tg[0]:.1f} s')
    print('[导航] gps_valid 全程 False，不传入 GNSS 位置/航向')

    gps_nav = np.zeros(T, dtype=bool)
    dummy_gps = np.zeros(T, dtype=np.float32)

    print('\n[1] 纯 DR（无 GPS）...')
    dr_x, dr_y, dr_h = run_pure_dr_no_gps(seq)

    print('[2] BiasNet + 6-State EKF（无 GPS）...')
    nav = EKFNavigatorNoGPS(
        weights_path=str(WEIGHTS_PATH),
        norm_stats=load_norm_stats(str(NORM_JSON)),
        window_size=WINDOW_SIZE,
        ekf_config=DEFAULT_EKF_CONFIG,
    )
    ekf_x, ekf_y, ekf_h, net_bias, _, _, _, _, _ = nav.run(
        imu_raw=seq['imu_raw'],
        v_ms=seq['v_ms'],
        gyro_z_rad=seq['gyro_z_rad'],
        gps_enu_x=dummy_gps,
        gps_enu_y=dummy_gps,
        gps_valid=gps_nav,
        dt=TARGET_DT,
        time_s=tg,
        gps_theta=None,
    )

    gx, gy = seq['enu_x_truth'], seq['enu_y_truth']
    eval_mask = np.isfinite(gx) & np.isfinite(gy)
    full_nav = np.ones(T, dtype=bool)

    metrics_dr = evaluate_trajectory(
        dr_x, dr_y, dr_h, gx, gy, seq['gps_theta'], eval_mask, full_nav)
    metrics_ekf = evaluate_trajectory(
        ekf_x, ekf_y, ekf_h, gx, gy, seq['gps_theta'], eval_mask, full_nav)

    print('\n' + '=' * 58)
    print('评估（相对 GNSS 真值，首帧对齐）')
    print('=' * 58)
    _print_metrics('DR', metrics_dr)
    _print_metrics('BiasNet + EKF', metrics_ekf)

    plot_no_gps(seq, dr_x, dr_y, ekf_x, ekf_y, ekf_h, net_bias, metrics_dr, metrics_ekf)
    print('\n完成。')


if __name__ == '__main__':
    main()
