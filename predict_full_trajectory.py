"""
predict_full_trajectory.py — 整段 260316 轨迹预测（全帧、不抽点）

融合输出（默认）：
  - 有 GNSS：位置 = 实测（与绿线重合）
  - 无 GNSS：BiasNet+EKF+TCN 外推（体现模型能力）

另输出 pure 轨迹：全程纯积分，仅用于看无信号段漂移。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

from trajectory_data import load_segment, TARGET_DT
from trajectory_fusion import FusedTrajectoryPredictor

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / 'trained_models'
BIASNET_W = MODEL_DIR / 'biasnet_weights.weights.h5'
TCN_PATH = MODEL_DIR / 'best_model.keras'
NORM_JSON = ROOT / 'preprocessed_data' / 'normalization_stats.json'
OUT_PNG = MODEL_DIR / 'full_trajectory_prediction.png'
OUT_CSV = MODEL_DIR / 'full_trajectory_points.csv'
OUT_TXT = MODEL_DIR / 'full_trajectory_metrics.txt'


def align_at_first_gps(truth_x, truth_y, pred_x, pred_y, gps_valid):
    ok = gps_valid & np.isfinite(truth_x) & np.isfinite(truth_y)
    if not ok.any():
        return truth_x, truth_y, pred_x, pred_y
    i0 = int(np.where(ok)[0][0])
    ox, oy = float(truth_x[i0]), float(truth_y[i0])
    return (truth_x - ox, truth_y - oy,
            pred_x - pred_x[i0], pred_y - pred_y[i0])


def interp_truth(tx, ty, gps_valid):
    idx = np.where(gps_valid & np.isfinite(tx))[0]
    if len(idx) < 2:
        return tx, ty
    xi = np.interp(np.arange(len(tx)), idx, tx[idx])
    yi = np.interp(np.arange(len(ty)), idx, ty[idx])
    return xi.astype(np.float32), yi.astype(np.float32)


def point_errors(px, py, tx, ty, mask):
    m = mask & np.isfinite(tx) & np.isfinite(ty) & np.isfinite(px) & np.isfinite(py)
    if not m.any():
        return np.array([]), {}
    e = np.sqrt((px[m] - tx[m]) ** 2 + (py[m] - ty[m]) ** 2)
    return e, {
        'n': int(m.sum()),
        'mean': float(e.mean()),
        'median': float(np.median(e)),
        'rmse': float(np.sqrt((e ** 2).mean())),
        'p95': float(np.percentile(e, 95)),
        'max': float(e.max()),
    }


def format_stats(name, stats):
    if not stats:
        return f'  {name}: 无有效样本\n'
    return (f'  {name} (n={stats["n"]}):\n'
            f'    平均 {stats["mean"]:.2f} m  中位 {stats["median"]:.2f} m  '
            f'RMSE {stats["rmse"]:.2f} m  P95 {stats["p95"]:.2f} m  最大 {stats["max"]:.2f} m\n')


def main():
    print('=' * 60)
    print('整段 260316 轨迹预测（全帧）')
    print('=' * 60)

    if not BIASNET_W.exists():
        raise FileNotFoundError(f'请先训练 EKF/BiasNet: {BIASNET_W}')
    if not TCN_PATH.exists():
        raise FileNotFoundError(f'请先训练 TCN: {TCN_PATH}')

    seq = load_segment(t_start=0.0)
    tg = seq['Time_s']
    gps_v = seq['gps_valid']
    n = len(tg)
    print(f'  时间 {tg[0]:.1f}–{tg[-1]:.1f} s, 共 {n} 帧 ({n * TARGET_DT / 60:.1f} min)')
    print(f'  GNSS 有效率 {gps_v.mean():.1%}')

    predictor = FusedTrajectoryPredictor(
        BIASNET_W, TCN_PATH, NORM_JSON, window_size=30, dt=TARGET_DT)

    print('  融合推理（有 GNSS 用实测位置）...')
    out = predictor.predict(seq, gps_valid_nav=gps_v, use_gnss_position=True)
    print('  纯预测推理（全程积分，仅作无信号段参考）...')
    out_pure = predictor.predict(seq, gps_valid_nav=gps_v, use_gnss_position=False)

    tx0, ty0 = seq['enu_x_truth'], seq['enu_y_truth']
    tx_a, ty_a, fx, fy = align_at_first_gps(tx0, ty0, out['fused_x'], out['fused_y'], gps_v)
    _, _, px, py = align_at_first_gps(tx0, ty0, out_pure['pure_x'], out_pure['pure_y'], gps_v)
    tx_i, ty_i = interp_truth(tx_a, ty_a, gps_v)

    err_fused = np.sqrt((fx - tx_i) ** 2 + (fy - ty_i) ** 2)
    err_pure = np.sqrt((px - tx_i) ** 2 + (py - ty_i) ** 2)

    lines = [
        '260316_Data.csv 整段轨迹（全帧，不抽点）\n',
        f'帧数 {n}, GNSS有效率 {gps_v.mean():.1%}\n\n',
        '=== 融合轨迹：有 GNSS 跟实测，无 GNSS 用模型 ===\n',
        format_stats('GNSS有效帧', point_errors(fx, fy, tx_a, ty_a, gps_v)[1]),
        format_stats('GNSS丢失帧', point_errors(fx, fy, tx_i, ty_i, ~gps_v)[1]),
        format_stats('全程', point_errors(fx, fy, tx_i, ty_i, np.ones(n, bool))[1]),
        '\n=== 纯积分（只看模型在无 GNSS 段能扛多少）===\n',
        format_stats('GNSS丢失帧', point_errors(px, py, tx_i, ty_i, ~gps_v)[1]),
    ]
    hold = tg >= 620.0
    if hold.any():
        lines.append('\n=== 保留段 t>=620 s（融合）===\n')
        lines.append(format_stats(
            '全程', point_errors(fx[hold], fy[hold], tx_i[hold], ty_i[hold],
                                 np.ones(hold.sum(), bool))[1]))

    text = ''.join(lines)
    OUT_TXT.write_text(text, encoding='utf-8')
    print(text)

    # 全帧 CSV
    import pandas as pd
    pd.DataFrame({
        'Time_s': tg,
        'gps_valid': gps_v.astype(int),
        'gnss_east_m': tx_a,
        'gnss_north_m': ty_a,
        'fused_east_m': fx,
        'fused_north_m': fy,
        'pure_east_m': px,
        'pure_north_m': py,
        'err_fused_m': err_fused,
        'err_pure_m': err_pure,
    }).to_csv(OUT_CSV, index=False, float_format='%.4f')
    print(f'[OK] 全帧 CSV: {OUT_CSV} ({n} 行)')

    # 绘图：全部点，不抽稀
    t_rel = tg - tg[0]
    idx_out = np.where(~gps_v)[0]

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])

    ax_map = fig.add_subplot(gs[0, :])
    ax_map.plot(tx_a, ty_a, 'g-', lw=1.2, label=f'GNSS truth ({n} pts)', zorder=3)
    ax_map.plot(fx, fy, 'b-', lw=1.0, alpha=0.95,
                label='Fused (GNSS when valid + model in outage)')
    ax_map.plot(px, py, color='steelblue', lw=0.6, alpha=0.45,
                label='Pure dead-reckoning+TCN (reference)')
    if len(idx_out) > 0:
        ax_map.plot(px[idx_out], py[idx_out], color='darkorange', lw=1.2, alpha=0.85,
                    label=f'Outage only — pure model ({len(idx_out)} pts)')
    ax_map.set_xlabel('East (m)')
    ax_map.set_ylabel('North (m)')
    ax_map.set_title(
        f'260316 Full Trajectory — all {n} frames, no downsampling',
        fontweight='bold')
    ax_map.legend(loc='best', fontsize=9)
    ax_map.set_aspect('equal')
    ax_map.grid(True, alpha=0.3)

    ax_err = fig.add_subplot(gs[1, 0])
    ax_err.plot(t_rel, err_fused, 'b-', lw=0.7, label='Fused error')
    ax_err.plot(t_rel, err_pure, color='steelblue', lw=0.5, alpha=0.6, label='Pure integral error')
    if len(idx_out) > 0:
        ax_err.axvspan(t_rel[idx_out[0]], t_rel[idx_out[-1]], alpha=0.1, color='orange',
                       label='GNSS outage')
    ax_err.set_xlabel('Time (s)')
    ax_err.set_ylabel('Error vs GNSS (m)')
    ax_err.set_title('Per-frame error (all frames)')
    ax_err.legend(fontsize=8)
    ax_err.grid(True, alpha=0.3)

    ax_hist = fig.add_subplot(gs[1, 1])
    if (~gps_v).any():
        ax_hist.hist(err_fused[~gps_v], bins=60, color='darkorange', alpha=0.7,
                     label='Fused @ outage')
        ax_hist.hist(err_pure[~gps_v], bins=60, histtype='step', color='steelblue',
                     lw=1.5, label='Pure @ outage')
    if gps_v.any():
        ax_hist.hist(err_fused[gps_v], bins=40, color='green', alpha=0.5,
                     label='Fused @ GNSS valid')
    ax_hist.set_xlabel('Error (m)')
    ax_hist.set_ylabel('Count')
    ax_hist.set_title('Error distribution (all frames)')
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[OK] 轨迹图: {OUT_PNG}')


if __name__ == '__main__':
    main()
