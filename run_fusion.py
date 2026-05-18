"""
run_fusion.py — TCN + BiasNet/EKF 融合轨迹推理（日常使用入口）

用法示例：
  # Data02 全段，真实 GPS 掩码
  python run_fusion.py --dataset Data02

  # 模拟 15s 后起 60s 无 GPS（与 validate_trajectory 相同）
  python run_fusion.py --dataset Data02 --loss-start 15 --loss-duration 60

  # 导出 CSV
  python run_fusion.py --dataset Data02 --csv outputs/fusion_Data02.csv
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

from trajectory_data import (
    load_calibration_segment, simulate_gps_loss, TARGET_DT)
from trajectory_fusion import FusedTrajectoryPredictor

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / 'trained_models'
BIASNET_W = MODEL_DIR / 'biasnet_weights.weights.h5'
TCN_PATH = MODEL_DIR / 'best_model.keras'
NORM_JSON = ROOT / 'preprocessed_data' / 'normalization_stats.json'


def main():
    ap = argparse.ArgumentParser(description='TCN+EKF 融合轨迹推理')
    ap.add_argument('--dataset', default='Data02', choices=['Data01', 'Data02'],
                    help='标定数据段（需有 标定实车数据/{id}_IMU/GNSS/VehicleSpeed.txt）')
    ap.add_argument('--loss-start', type=float, default=None,
                    help='模拟无 GPS 起始时间(s，相对段首)，不设则用真实 GPS 掩码')
    ap.add_argument('--loss-duration', type=float, default=60.0,
                    help='模拟无 GPS 时长(s)')
    ap.add_argument('--no-relock', action='store_true',
                    help='禁用 GNSS 重捕获（只看纯预测）')
    ap.add_argument('--png', type=str, default=None, help='输出轨迹图路径')
    ap.add_argument('--csv', type=str, default=None, help='输出逐帧 CSV 路径')
    args = ap.parse_args()

    if not BIASNET_W.exists():
        raise FileNotFoundError(f'缺少 BiasNet: {BIASNET_W}\n请先: python train_ekf.py')
    if not TCN_PATH.exists():
        raise FileNotFoundError(f'缺少 TCN: {TCN_PATH}\n请先: python train_tcn_model.py')
    if not NORM_JSON.exists():
        raise FileNotFoundError(f'缺少归一化: {NORM_JSON}\n请先: python data_preprocessing_v2.py')

    print('=' * 60)
    print(f'融合推理  dataset={args.dataset}')
    print('=' * 60)

    seq = load_calibration_segment(args.dataset)
    tg = seq['Time_s']
    gps_nav = seq['gps_valid'].copy()

    if args.loss_start is not None:
        gps_nav, i0, i1 = simulate_gps_loss(
            seq['gps_valid'], tg, args.loss_start, args.loss_duration)
        print(f'  模拟无 GPS: {args.loss_duration:.0f}s (idx {i0}–{i1})')
    else:
        print('  使用真实 GPS 有效掩码（不模拟隧道）')

    print(f'  帧数 {len(tg)}, t={tg[0]:.1f}–{tg[-1]:.1f}s, GPS有效 {gps_nav.mean():.1%}')

    predictor = FusedTrajectoryPredictor(
        BIASNET_W, TCN_PATH, NORM_JSON, window_size=30, dt=TARGET_DT)
    use_pos = not args.no_relock
    out = predictor.predict(seq, gps_valid_nav=gps_nav, use_gnss_position=use_pos)
    out_pure = predictor.predict(
        seq, gps_valid_nav=gps_nav, use_gnss_position=False)

    fx, fy = out['fused_x'], out['fused_y']
    tx, ty = seq['enu_x_truth'], seq['enu_y_truth']
    err = np.sqrt((fx - tx) ** 2 + (fy - ty) ** 2)
    err_pure = np.sqrt(
        (out_pure['fused_x'] - tx) ** 2 + (out_pure['fused_y'] - ty) ** 2)

    on_out = ~gps_nav
    if on_out.any():
        print(f'  无GPS段 误差中位: 融合 {np.median(err[on_out]):.2f} m  '
              f'纯积分 {np.median(err_pure[on_out]):.2f} m')
    print(f'  全程误差中位: {np.median(err):.2f} m')

    if args.csv:
        import pandas as pd
        p = Path(args.csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            'Time_s': tg,
            'gps_valid_nav': gps_nav.astype(int),
            'gnss_east_m': tx,
            'gnss_north_m': ty,
            'fused_east_m': fx,
            'fused_north_m': fy,
            'pure_east_m': out_pure['fused_x'],
            'pure_north_m': out_pure['fused_y'],
            'err_fused_m': err,
        }).to_csv(p, index=False, float_format='%.4f')
        print(f'[OK] CSV: {p}')

    png = Path(args.png) if args.png else MODEL_DIR / f'fusion_{args.dataset}.png'
    png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    ok = np.isfinite(tx) & np.isfinite(ty)
    ax.plot(tx[ok], ty[ok], 'g-', lw=1.5, label='GNSS')
    ax.plot(fx, fy, 'b-', lw=1.2, label='Fused (TCN+EKF)')
    if on_out.any():
        ax.plot(fx[on_out], fy[on_out], 'r.', ms=3, alpha=0.6, label='No GPS segment')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Fusion {args.dataset}  relock={use_pos}')
    plt.tight_layout()
    plt.savefig(png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[OK] PNG: {png}')


if __name__ == '__main__':
    main()
