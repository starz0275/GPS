"""
validate_trajectory.py — EKF + TCN 融合轨迹验证
输出: trained_models/fused_trajectory.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from trajectory_data import load_segment, simulate_gps_loss, VAL_T_START, TARGET_DT
from trajectory_fusion import FusedTrajectoryPredictor

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "trained_models"
BIASNET_W = MODEL_DIR / "biasnet_weights.weights.h5"
TCN_PATH = MODEL_DIR / "best_model.keras"
NORM_JSON = ROOT / "preprocessed_data" / "normalization_stats.json"
OUT_PNG = MODEL_DIR / "fused_trajectory.png"

GPS_LOSS_START_S = 15.0
GPS_LOSS_DURATION_S = 60.0


def align_origin(truth_x, truth_y, pred_x, pred_y, gps_valid):
    ok = gps_valid & np.isfinite(truth_x) & np.isfinite(truth_y)
    if not ok.any():
        return truth_x, truth_y, pred_x, pred_y
    i0 = int(np.where(ok)[0][0])
    ox, oy = float(truth_x[i0]), float(truth_y[i0])
    return truth_x - ox, truth_y - oy, pred_x - pred_x[i0], pred_y - pred_y[i0]


def interp_truth(tx, ty, gps_valid):
    idx = np.where(gps_valid & np.isfinite(tx))[0]
    if len(idx) < 2:
        return tx, ty
    xi = np.interp(np.arange(len(tx)), idx, tx[idx])
    yi = np.interp(np.arange(len(ty)), idx, ty[idx])
    return xi.astype(np.float32), yi.astype(np.float32)


def rpe_segment(tx, ty, px, py, i0, i1):
    return float(np.sqrt(
        (px[i1] - px[i0] - (tx[i1] - tx[i0])) ** 2 +
        (py[i1] - py[i0] - (ty[i1] - ty[i0])) ** 2))


def main():
    print("=" * 60)
    print("EKF + TCN 融合轨迹验证（无 GPS 段启用 TCN）")
    print("=" * 60)

    if not BIASNET_W.exists():
        raise FileNotFoundError(f"请先运行 train_ekf.py: {BIASNET_W}")
    if not TCN_PATH.exists():
        print(f"[WARN] 缺少 {TCN_PATH}，将只对比 DR / EKF")

    seq = load_segment(t_start=VAL_T_START)
    tg = seq['Time_s']
    print(f"  验证段 t={tg[0]:.0f}–{tg[-1]:.0f}s, {len(tg)} 帧, "
          f"GPS有效率={seq['gps_valid'].mean():.1%}")

    gps_nav, i0, i1 = simulate_gps_loss(
        seq['gps_valid'], tg, GPS_LOSS_START_S, GPS_LOSS_DURATION_S)
    print(f"  模拟无GPS: {GPS_LOSS_DURATION_S:.0f}s (idx {i0}–{i1})")

    pred = FusedTrajectoryPredictor(
        BIASNET_W, TCN_PATH, NORM_JSON, window_size=30, dt=TARGET_DT)
    out = pred.predict(seq, gps_valid_nav=gps_nav, use_gnss_position=True)
    out_raw = pred.predict(seq, gps_valid_nav=gps_nav, use_gnss_position=False)

    eval_mask = seq['gps_valid']
    tx, ty, drx, dry = align_origin(
        seq['enu_x_truth'], seq['enu_y_truth'], out['dr_x'], out['dr_y'], eval_mask)
    _, _, ekx, eky = align_origin(
        seq['enu_x_truth'], seq['enu_y_truth'], out['ekf_x'], out['ekf_y'], eval_mask)
    _, _, fux, fuy = align_origin(
        seq['enu_x_truth'], seq['enu_y_truth'], out['fused_x'], out['fused_y'], eval_mask)
    _, _, fur, fvr = align_origin(
        seq['enu_x_truth'], seq['enu_y_truth'],
        out_raw['fused_x'], out_raw['fused_y'], eval_mask)

    tx_i, ty_i = interp_truth(tx, ty, eval_mask)
    idx = np.where(eval_mask)[0]

    def err_at(k):
        return float(np.sqrt((fux[k] - tx_i[k]) ** 2 + (fuy[k] - ty_i[k]) ** 2))

    dr_rpe = rpe_segment(tx_i, ty_i, drx, dry, i0, i1)
    ek_rpe = rpe_segment(tx_i, ty_i, ekx, eky, i0, i1)
    fu_rpe = rpe_segment(tx_i, ty_i, fux, fuy, i0, i1)
    fu_rpe_raw = rpe_segment(tx_i, ty_i, fur, fvr, i0, i1)
    fu_exit = err_at(min(i1, len(fux) - 1))
    fu_exit_raw = float(np.sqrt(
        (fur[min(i1, len(fur) - 1)] - tx_i[min(i1, len(tx_i) - 1)]) ** 2 +
        (fvr[min(i1, len(fvr) - 1)] - ty_i[min(i1, len(ty_i) - 1)]) ** 2))

    t_rel = tg - tg[0]

    # 绘图：聚焦真值附近
    fig, (ax_traj, ax_err) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f'Fused Trajectory (TCN on outage only, GPS loss {GPS_LOSS_DURATION_S:.0f}s, t>={VAL_T_START:.0f}s)',
        fontsize=13, fontweight='bold')

    ax_traj.plot(tx_i[idx], ty_i[idx], 'g-', lw=2.5, label='GNSS truth', zorder=5)
    ax_traj.plot(drx, dry, 'r--', lw=1.2, alpha=0.8, label=f'Pure DR (RPE {dr_rpe:.1f}m)')
    ax_traj.plot(ekx, eky, color='darkorange', ls='-.', lw=1.6,
                 label=f'EKF only (RPE {ek_rpe:.1f}m)')
    if pred.tcn is not None:
        ax_traj.plot(fux, fuy, 'b-', lw=2,
                     label=f'EKF+TCN hybrid (RPE {fu_rpe:.1f}m)')
    if i0 < len(fux):
        ax_traj.axvspan(fux[i0], fux[min(i1, len(fux) - 1)],
                        alpha=0.15, color='orange', label='No GNSS')
    ax_traj.set_xlabel('East (m)')
    ax_traj.set_ylabel('North (m)')
    ax_traj.legend(fontsize=8)
    ax_traj.grid(True, alpha=0.35)
    ax_traj.set_aspect('equal')
    pad = 30
    xm, xM = np.nanmin(tx_i[idx]) - pad, np.nanmax(tx_i[idx]) + pad
    ym, yM = np.nanmin(ty_i[idx]) - pad, np.nanmax(ty_i[idx]) + pad
    ax_traj.set_xlim(xm, xM)
    ax_traj.set_ylim(ym, yM)

    dr_e = np.sqrt((drx[idx] - tx_i[idx]) ** 2 + (dry[idx] - ty_i[idx]) ** 2)
    ek_e = np.sqrt((ekx[idx] - tx_i[idx]) ** 2 + (eky[idx] - ty_i[idx]) ** 2)
    fu_e = np.sqrt((fux[idx] - tx_i[idx]) ** 2 + (fuy[idx] - ty_i[idx]) ** 2)

    ax_err.plot(t_rel[idx], dr_e, 'r--', lw=1.2, label='Pure DR')
    ax_err.plot(t_rel[idx], ek_e, color='darkorange', ls='-.', lw=1.4, label='EKF only')
    if pred.tcn is not None:
        ax_err.plot(t_rel[idx], fu_e, 'b-', lw=1.6, label='EKF+TCN')
    ax_err.axvspan(t_rel[i0], t_rel[min(i1, len(t_rel) - 1)], alpha=0.12, color='orange')
    ax_err.set_xlabel('Time (s)')
    ax_err.set_ylabel('Error vs GNSS (m)')
    ax_err.legend(fontsize=9)
    ax_err.grid(True, alpha=0.35)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] {OUT_PNG}")

    print("\n" + "=" * 60)
    print("指标（相对 GNSS）")
    print("=" * 60)
    print(f"  隧道段 RPE ({GPS_LOSS_DURATION_S:.0f}s):  DR {dr_rpe:.1f}m  "
          f"EKF {ek_rpe:.1f}m  纯积分 {fu_rpe_raw:.1f}m  "
          f"融合(出隧道锚定GNSS) {fu_rpe:.1f}m")
    print(f"  丢失段结束误差:  纯积分 {fu_exit_raw:.1f}m  "
          f"融合锚定后 {fu_exit:.1f}m")
    if len(fu_e) > 0:
        print(f"  全程(GPS有效帧) 中位误差:  DR {np.median(dr_e):.1f}m  "
              f"EKF {np.median(ek_e):.1f}m  EKF+TCN {np.median(fu_e):.1f}m")
    print("=" * 60)


if __name__ == '__main__':
    main()
