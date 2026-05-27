"""
validate_trajectory.py — 轨迹对比验证（GNSS / DR / TCN / EKF+TCN）

对比四条轨迹：
  - GNSS：真值
  - DR：纯航位推算（陀螺 + 车速，无 BiasNet/TCN）
  - TCN：BiasNet 航向 + 失锁段 TCN 残差，全程纯积分（不贴 GNSS 坐标）
  - EKF+TCN：同上步长，有 GPS 时段位置贴 GNSS

输出:
  trained_models/fused_trajectory.png
  python validate_trajectory.py --plotly

依赖: pip install plotly
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

from trajectory_data import (
    load_calibration_segment, simulate_gps_loss, VAL_DATASET_ID, TARGET_DT,
    CALIB_ALL_IDS,
)
from trajectory_fusion import FusedTrajectoryPredictor

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "trained_models"
BIASNET_W = MODEL_DIR / "biasnet_weights.weights.h5"
TCN_PATH = MODEL_DIR / "best_model.keras"
NORM_JSON = ROOT / "preprocessed_data" / "normalization_stats.json"
OUT_PNG = MODEL_DIR / "fused_trajectory.png"

GPS_LOSS_START_S = 15.0
GPS_LOSS_DURATION_S = 60.0

# 绘图：绿线=GNSS 参考（半透明、底层）；蓝线=融合（顶层，失锁段加粗）
GNSS_LINE_LW_MPL = 3.5
GNSS_LINE_ALPHA = 0.45
FUSED_LINE_LW_MPL = 2.8
FUSED_OUTAGE_LW_MPL = 4.5
GNSS_LINE_LW_PLOTLY = 3.5
GNSS_LINE_OPACITY_PLOTLY = 0.45
FUSED_LINE_LW_PLOTLY = 3.5
FUSED_OUTAGE_LW_PLOTLY = 5.5


def _mask_break_line(x, y, show_mask):
    """仅在 show_mask 为 True 的连续段画线（False 处插 NaN 断线）。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.asarray(show_mask, dtype=bool)
    if m.size == 0:
        return x, y
    xo = x.copy()
    yo = y.copy()
    xo[~m] = np.nan
    yo[~m] = np.nan
    # 段首尾断线，避免跨段直连
    gaps = np.where(np.diff(m.astype(int)) != 0)[0] + 1
    for g in gaps:
        if 0 < g < len(xo):
            xo[g - 1] = yo[g - 1] = np.nan
    return xo, yo


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


def run_fusion_eval(
    dataset_id=VAL_DATASET_ID,
    loss_start_s=GPS_LOSS_START_S,
    loss_duration_s=GPS_LOSS_DURATION_S,
):
    if not BIASNET_W.exists():
        raise FileNotFoundError(f"请先运行 train_ekf.py: {BIASNET_W}")

    seq = load_calibration_segment(dataset_id)
    tg = seq['Time_s']
    gps_nav, i0, i1 = simulate_gps_loss(
        seq['gps_valid'], tg, loss_start_s, loss_duration_s)

    pred = FusedTrajectoryPredictor(
        BIASNET_W, TCN_PATH, NORM_JSON, window_size=30, dt=TARGET_DT)
    # 一次 predict：pure_* = EKF 底盘 + 失锁段 TCN，纯积分；fused_* = 分段贴 GNSS
    out = pred.predict(seq, gps_valid_nav=gps_nav, use_gnss_position=True)

    eval_mask = seq['gps_valid']
    tx, ty, drx, dry = align_origin(
        seq['enu_x_truth'], seq['enu_y_truth'], out['dr_x'], out['dr_y'], eval_mask)
    _, _, tcx, tcy = align_origin(
        seq['enu_x_truth'], seq['enu_y_truth'], out['pure_x'], out['pure_y'], eval_mask)
    _, _, fux, fuy = align_origin(
        seq['enu_x_truth'], seq['enu_y_truth'], out['fused_x'], out['fused_y'], eval_mask)

    tx_i, ty_i = interp_truth(tx, ty, eval_mask)
    idx = np.where(eval_mask)[0]
    t_rel = tg - tg[0]

    dr_e = np.sqrt((drx[idx] - tx_i[idx]) ** 2 + (dry[idx] - ty_i[idx]) ** 2)
    tcn_e = np.sqrt((tcx[idx] - tx_i[idx]) ** 2 + (tcy[idx] - ty_i[idx]) ** 2)
    fu_e = np.sqrt((fux[idx] - tx_i[idx]) ** 2 + (fuy[idx] - ty_i[idx]) ** 2)

    metrics = {
        'dr_rpe': rpe_segment(tx_i, ty_i, drx, dry, i0, i1),
        'tcn_rpe': rpe_segment(tx_i, ty_i, tcx, tcy, i0, i1),
        'fu_rpe': rpe_segment(tx_i, ty_i, fux, fuy, i0, i1),
        'tcn_exit': float(np.sqrt(
            (tcx[min(i1, len(tcx) - 1)] - tx_i[min(i1, len(tx_i) - 1)]) ** 2 +
            (tcy[min(i1, len(tcy) - 1)] - ty_i[min(i1, len(ty_i) - 1)]) ** 2)),
        'fu_exit': float(np.sqrt(
            (fux[min(i1, len(fux) - 1)] - tx_i[min(i1, len(tx_i) - 1)]) ** 2 +
            (fuy[min(i1, len(fuy) - 1)] - ty_i[min(i1, len(ty_i) - 1)]) ** 2)),
    }

    return {
        'dataset_id': dataset_id,
        'pred': pred,
        'tg': tg,
        't_rel': t_rel,
        'gps_nav': gps_nav,
        'i0': i0,
        'i1': i1,
        'idx': idx,
        'tx_i': tx_i, 'ty_i': ty_i,
        'drx': drx, 'dry': dry,
        'tcx': tcx, 'tcy': tcy,
        'fux': fux, 'fuy': fuy,
        'dr_e': dr_e, 'tcn_e': tcn_e, 'fu_e': fu_e,
        'metrics': metrics,
        'loss_duration_s': loss_duration_s,
    }


def save_png_plot(res, out_path=OUT_PNG):
    m = res['metrics']
    idx = res['idx']
    t_rel = res['t_rel']
    i0, i1 = res['i0'], res['i1']

    fig, (ax_traj, ax_err) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f'Trajectory Compare ({res["dataset_id"]}, GPS loss {res["loss_duration_s"]:.0f}s)',
        fontsize=13, fontweight='bold')

    gps_nav = res['gps_nav']
    ax_traj.plot(res['tx_i'][idx], res['ty_i'][idx], color='#2ca02c', lw=GNSS_LINE_LW_MPL,
                 alpha=GNSS_LINE_ALPHA, solid_capstyle='round',
                 label='GNSS', zorder=4)
    ax_traj.plot(res['drx'], res['dry'], 'r--', lw=1.4, alpha=0.85,
                 label=f'DR (RPE {m["dr_rpe"]:.1f}m)', zorder=3)
    if res['pred'].tcn is not None:
        ax_traj.plot(res['tcx'], res['tcy'], color='#7B1FA2', lw=2.2,
                     label=f'TCN (RPE {m["tcn_rpe"]:.1f}m)', zorder=6)
        ax_traj.plot(res['fux'], res['fuy'], color='#1565C0', lw=FUSED_LINE_LW_MPL,
                     label=f'EKF+TCN (RPE {m["fu_rpe"]:.1f}m)', zorder=8)
        ox, oy = _mask_break_line(res['fux'], res['fuy'], ~gps_nav)
        ax_traj.plot(ox, oy, color='#0D47A1', lw=FUSED_OUTAGE_LW_MPL,
                     label='EKF+TCN 失锁段', zorder=9)
    if i0 < len(res['fux']):
        ax_traj.axvspan(res['fux'][i0], res['fux'][min(i1, len(res['fux']) - 1)],
                        alpha=0.15, color='orange', label='No GNSS')
    ax_traj.set_xlabel('East (m)')
    ax_traj.set_ylabel('North (m)')
    ax_traj.legend(fontsize=8)
    ax_traj.grid(True, alpha=0.35)
    ax_traj.set_aspect('equal')

    ax_err.plot(t_rel[idx], res['dr_e'], 'r--', lw=1.2, label='DR')
    if res['pred'].tcn is not None:
        ax_err.plot(t_rel[idx], res['tcn_e'], color='#7B1FA2', lw=1.5, label='TCN')
        ax_err.plot(t_rel[idx], res['fu_e'], 'b-', lw=1.6, label='EKF+TCN')
    ax_err.axvspan(t_rel[i0], t_rel[min(i1, len(t_rel) - 1)], alpha=0.12, color='orange')
    ax_err.set_xlabel('Time (s)')
    ax_err.set_ylabel('Error vs GNSS (m)')
    ax_err.legend(fontsize=9)
    ax_err.grid(True, alpha=0.35)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] {out_path}")


def show_plotly_interactive(res):
    """Plotly 交互图：浏览器中缩放、平移、悬停查看时间/误差。"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    m = res['metrics']
    idx = res['idx']
    t_rel = res['t_rel']
    i0, i1 = res['i0'], res['i1']
    gps_nav = res['gps_nav']

    hover = (
        '时间: %{customdata[0]:.1f} s<br>'
        '东向: %{x:.2f} m<br>'
        '北向: %{y:.2f} m<br>'
        '误差: %{customdata[1]:.2f} m<br>'
        '导航GPS: %{customdata[2]}<extra></extra>'
    )
    tcn_e_full = np.sqrt(
        (res['tcx'] - res['tx_i']) ** 2 + (res['tcy'] - res['ty_i']) ** 2)
    fu_e_full = np.sqrt(
        (res['fux'] - res['tx_i']) ** 2 + (res['fuy'] - res['ty_i']) ** 2)
    cd_tcn = np.column_stack([t_rel, tcn_e_full, gps_nav.astype(int)])
    cd_fu = np.column_stack([t_rel, fu_e_full, gps_nav.astype(int)])

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.58, 0.42],
        subplot_titles=(
            f'轨迹 ({res["dataset_id"]}, 失锁 {res["loss_duration_s"]:.0f}s)',
            '位置误差 vs 时间',
        ),
        horizontal_spacing=0.08,
    )

    fig.add_trace(go.Scatter(
        x=res['tx_i'][idx], y=res['ty_i'][idx],
        mode='lines', name='GNSS',
        line=dict(color='#2ca02c', width=GNSS_LINE_LW_PLOTLY),
        opacity=GNSS_LINE_OPACITY_PLOTLY,
        customdata=np.column_stack([t_rel[idx], np.zeros(len(idx)), gps_nav[idx].astype(int)]),
        hovertemplate=hover,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=res['drx'], y=res['dry'], mode='lines', name=f'DR (RPE {m["dr_rpe"]:.1f}m)',
        line=dict(color='red', dash='dash', width=1.5),
    ), row=1, col=1)

    if res['pred'].tcn is not None:
        fig.add_trace(go.Scatter(
            x=res['tcx'], y=res['tcy'], mode='lines',
            name=f'TCN (RPE {m["tcn_rpe"]:.1f}m)',
            line=dict(color='#7B1FA2', width=2.4),
            customdata=cd_tcn,
            hovertemplate=hover,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=res['fux'], y=res['fuy'], mode='lines',
            name=f'EKF+TCN (RPE {m["fu_rpe"]:.1f}m)',
            line=dict(color='#1565C0', width=FUSED_LINE_LW_PLOTLY),
            customdata=cd_fu,
            hovertemplate=hover,
        ), row=1, col=1)
        ox, oy = _mask_break_line(res['fux'], res['fuy'], ~gps_nav)
        fig.add_trace(go.Scatter(
            x=ox, y=oy, mode='lines', name='EKF+TCN 失锁段',
            line=dict(color='#0D47A1', width=FUSED_OUTAGE_LW_PLOTLY),
            customdata=cd_fu,
            hovertemplate=hover,
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t_rel[idx], y=res['dr_e'], mode='lines', name='DR',
        line=dict(color='red', dash='dash'),
    ), row=1, col=2)
    if res['pred'].tcn is not None:
        fig.add_trace(go.Scatter(
            x=t_rel[idx], y=res['tcn_e'], mode='lines', name='TCN',
            line=dict(color='#7B1FA2', width=1.8),
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=t_rel[idx], y=res['fu_e'], mode='lines', name='EKF+TCN',
            line=dict(color='royalblue', width=2),
        ), row=1, col=2)

    fig.add_vrect(
        x0=t_rel[i0], x1=t_rel[min(i1, len(t_rel) - 1)],
        fillcolor='orange', opacity=0.12, line_width=0,
        row=1, col=2,
    )

    fig.update_xaxes(title_text='East (m)', scaleanchor='y', scaleratio=1, row=1, col=1)
    fig.update_yaxes(title_text='North (m)', row=1, col=1)
    fig.update_xaxes(title_text='时间 (s)', row=1, col=2)
    fig.update_yaxes(title_text='误差 (m)', row=1, col=2)

    fig.update_layout(
        height=650,
        title_text=(
            f'轨迹对比 — {res["dataset_id"]}  '
            f'隧道RPE: DR {m["dr_rpe"]:.1f}m / TCN {m["tcn_rpe"]:.1f}m / '
            f'EKF+TCN {m["fu_rpe"]:.1f}m'
        ),
        legend=dict(orientation='h', yanchor='bottom', y=-0.15, x=0),
        hovermode='closest',
    )

    print('[Plotly] 正在打开交互窗口（可缩放、框选、悬停）…')
    fig.show(renderer='browser')


def print_metrics(res):
    m = res['metrics']
    print("\n" + "=" * 60)
    print("指标（相对 GNSS）")
    print("=" * 60)
    print(f"  隧道段 RPE ({res['loss_duration_s']:.0f}s):  "
          f"DR {m['dr_rpe']:.1f}m  TCN {m['tcn_rpe']:.1f}m  EKF+TCN {m['fu_rpe']:.1f}m")
    print(f"  丢失段结束误差:  TCN {m['tcn_exit']:.1f}m  EKF+TCN {m['fu_exit']:.1f}m")
    if len(res['fu_e']) > 0:
        print(f"  全程(GPS有效帧) 中位误差:  DR {np.median(res['dr_e']):.1f}m  "
              f"TCN {np.median(res['tcn_e']):.1f}m  "
              f"EKF+TCN {np.median(res['fu_e']):.1f}m")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description='EKF+TCN 融合轨迹验证')
    ap.add_argument('--plotly', action='store_true',
                    help='用 Plotly 打开交互图（浏览器）')
    ap.add_argument('--no-png', action='store_true', help='不保存静态 PNG')
    ap.add_argument('--dataset', default=VAL_DATASET_ID, choices=CALIB_ALL_IDS)
    ap.add_argument('--loss-start', type=float, default=GPS_LOSS_START_S)
    ap.add_argument('--loss-duration', type=float, default=GPS_LOSS_DURATION_S)
    args = ap.parse_args()

    print("=" * 60)
    print("EKF + TCN 融合轨迹验证")
    print("=" * 60)

    res = run_fusion_eval(args.dataset, args.loss_start, args.loss_duration)
    tg = res['tg']
    print(f"  验证集 {args.dataset}  t={tg[0]:.0f}–{tg[-1]:.0f}s, {len(tg)} 帧")
    print(f"  模拟无GPS: {args.loss_duration:.0f}s (idx {res['i0']}–{res['i1']})")

    if not args.no_png:
        save_png_plot(res)

    print_metrics(res)

    if args.plotly:
        show_plotly_interactive(res)
    else:
        print("\n提示: 加 --plotly 可打开 Plotly 交互图")


if __name__ == '__main__':
    main()
