#!/usr/bin/env python3
"""scripts/visualize_attitude.py — 姿态预测 vs CMCC 真值对比，输出矢量图。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data0109_loader import (
    DATA0109_TRAIN_SEGMENTS,
    DATA0109_VAL_SEGMENT,
    load_data0109_segments,
)
from train_ekf import load_or_compute_norm, normalize_imu
from scripts.train_attitude_from_imu import (
    AttitudeNet,
    quat_to_euler,
    build_windows,
    WINDOW_SIZE,
)

NORM_JSON_0109 = ROOT / "preprocessed_data" / "normalization_stats_data0109.json"
MODEL_DIR = ROOT / "trained_models"
OUT_DIR = ROOT / "trained_models"
OUT_DIR.mkdir(exist_ok=True)


def plot_segment(seq, model, mu, std, out_path: Path):
    """对一段数据逐窗口推理，生成 pred vs truth 时间序列图。"""
    window_size = WINDOW_SIZE
    imu = seq['imu']
    gps_theta = seq['gps_theta']
    att_deg = seq['cmcc_attitude_deg']  # (T, 3): pitch, roll, yaw
    cmcc_stable = seq['cmcc_stable']
    t = seq['Time_s']
    T = len(imu)

    gps_sin = np.sin(gps_theta).astype(np.float32)
    gps_cos = np.cos(gps_theta).astype(np.float32)

    pred_euler = np.full((T, 3), np.nan, dtype=np.float32)
    true_euler = np.full((T, 3), np.nan, dtype=np.float32)

    for i in range(0, T - window_size + 1):
        idx = i + window_size - 1
        if not cmcc_stable[idx]:
            continue
        imu_norm = normalize_imu(imu[i:i + window_size], mu, std)
        gps_feat = np.stack([gps_sin[i:i + window_size],
                             gps_cos[i:i + window_size]], axis=-1)
        x_win = np.concatenate([imu_norm, gps_feat], axis=-1)
        x_batch = x_win[np.newaxis, ...]

        q_pred = model(x_batch, training=False)
        euler_rad = quat_to_euler(q_pred).numpy().flatten()
        pred_euler[idx] = euler_rad

        pitch, roll, yaw = att_deg[idx] * (np.pi / 180.0)
        true_euler[idx] = [roll, pitch, yaw]

    # ── 绘图 ──
    labels = ['Roll', 'Pitch', 'Yaw']
    colors = ['#d62728', '#2ca02c', '#1f77b4']
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        t_rel = (t - t[0]) / 60.0  # 分钟
        p = pred_euler[:, i] * 180.0 / np.pi
        g = true_euler[:, i] * 180.0 / np.pi

        valid = ~np.isnan(p) & ~np.isnan(g)
        ax.plot(t_rel[valid], g[valid], color='gray', alpha=0.7,
                linewidth=0.6, label='CMCC (truth)')
        ax.plot(t_rel[valid], p[valid], color=color, linewidth=1.0,
                alpha=0.9, label='AttitudeNet')

        diff = np.abs(np.arctan2(np.sin((p[valid] - g[valid]) * np.pi / 180.0),
                                 np.cos((p[valid] - g[valid]) * np.pi / 180.0)))
        rmse = np.sqrt(np.mean(diff ** 2))
        ax.set_ylabel(f'{label} [deg]')
        ax.legend(loc='upper right', fontsize=8)
        ax.text(0.99, 0.05, f'RMSE = {rmse:.2f}°',
                transform=ax.transAxes, ha='right', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time [min]')
    seg_name = seq.get('segment', 'unknown')
    fig.suptitle(f'AttitudeNet Prediction vs CMCC Truth — {seg_name}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()

    fig.savefig(out_path, dpi=150, format='svg')
    plt.close(fig)
    print(f'  Saved {out_path.name}')


def plot_error_distribution(all_pred, all_true, out_path: Path):
    """绘制三角度误差分布直方图。"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    labels = ['Roll', 'Pitch', 'Yaw']

    for i, (ax, label) in enumerate(zip(axes, labels)):
        diff = np.abs(np.arctan2(
            np.sin(all_pred[:, i] - all_true[:, i]),
            np.cos(all_pred[:, i] - all_true[:, i])))
        diff_deg = diff * 180.0 / np.pi
        ax.hist(diff_deg, bins=60, density=True, color='steelblue',
                edgecolor='white', alpha=0.85)
        ax.axvline(np.median(diff_deg), color='red', linewidth=1.5,
                   linestyle='--', label=f'median={np.median(diff_deg):.2f}°')
        ax.set_xlabel(f'{label} error [deg]')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Attitude Error Distribution (Validation Set)', fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, format='svg')
    plt.close(fig)
    print(f'  Saved {out_path.name}')


def main():
    parser = argparse.ArgumentParser(description='Visualize attitude predictions')
    parser.add_argument('--segment', type=str, default='val',
                        help='val | train index (0-3) | segment name')
    parser.add_argument('--weights', type=str, default='attitude_from_imu')
    parser.add_argument('--output-dir', type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    print('[Info] Loading model...')
    import tensorflow as tf
    model = AttitudeNet(window_size=WINDOW_SIZE)
    model(tf.zeros((1, WINDOW_SIZE, 9), dtype=tf.float32))
    weights_path = MODEL_DIR / f'{args.weights}.weights.h5'
    model.load_weights(str(weights_path))
    print(f'  Loaded {weights_path}')

    print('[Info] Loading segments...')
    seqs_val = load_data0109_segments([DATA0109_VAL_SEGMENT])
    seqs_val = [s for s in seqs_val if s is not None]
    seqs_tr = load_data0109_segments(DATA0109_TRAIN_SEGMENTS)
    seqs_tr = [s for s in seqs_tr if s is not None]
    all_seqs = seqs_tr + seqs_val

    mu, std, _ = load_or_compute_norm(all_seqs, NORM_JSON_0109)

    # 选择段
    if args.segment == 'val':
        segments = seqs_val
    elif args.segment.isdigit():
        idx = int(args.segment)
        segments = [seqs_tr[idx]] if 0 <= idx < len(seqs_tr) else seqs_val
    else:
        segments = [s for s in all_seqs if s and args.segment in s.get('segment', '')]
        if not segments:
            segments = seqs_val

    # ── 逐段时间序列图 ──
    print('[Info] Generating time-series plots...')
    for seq in segments:
        seg_name = seq['segment'].replace(' ', '_')
        ts_path = out_dir / f'attitude_ts_{seq["id"]}.svg'
        plot_segment(seq, model, mu, std, ts_path)

    # ── 全局误差分布 ──
    print('[Info] Computing global error distribution...')
    all_pred, all_true = [], []
    window_size = WINDOW_SIZE
    for seq in segments:
        imu = seq['imu']
        gps_theta = seq['gps_theta']
        att_deg = seq['cmcc_attitude_deg']
        cmcc_stable = seq['cmcc_stable']
        T = len(imu)
        gps_sin = np.sin(gps_theta).astype(np.float32)
        gps_cos = np.cos(gps_theta).astype(np.float32)

        for i in range(0, T - window_size + 1):
            idx = i + window_size - 1
            if not cmcc_stable[idx]:
                continue
            imu_norm = normalize_imu(imu[i:i + window_size], mu, std)
            gps_feat = np.stack([gps_sin[i:i + window_size],
                                 gps_cos[i:i + window_size]], axis=-1)
            x_win = np.concatenate([imu_norm, gps_feat], axis=-1)
            x_batch = x_win[np.newaxis, ...]

            q_pred = model(x_batch, training=False)
            euler_rad = quat_to_euler(q_pred).numpy().flatten()
            all_pred.append(euler_rad)

            pitch, roll, yaw = att_deg[idx] * (np.pi / 180.0)
            all_true.append([roll, pitch, yaw])

    all_pred = np.array(all_pred, dtype=np.float32)
    all_true = np.array(all_true, dtype=np.float32)
    dist_path = out_dir / 'attitude_error_dist.svg'
    plot_error_distribution(all_pred, all_true, dist_path)

    # ── 汇总 ──
    print('\n[Summary]')
    for i, label in enumerate(['Roll', 'Pitch', 'Yaw']):
        diff = np.abs(np.arctan2(np.sin(all_pred[:, i] - all_true[:, i]),
                                 np.cos(all_pred[:, i] - all_true[:, i])))
        rmse = np.sqrt(np.mean(diff ** 2)) * 180.0 / np.pi
        mae = np.mean(diff) * 180.0 / np.pi
        p95 = np.percentile(diff, 95) * 180.0 / np.pi
        print(f'  {label:5s}  RMSE={rmse:.2f}°  MAE={mae:.2f}°  P95={p95:.2f}°')

    print(f'\n[Info] Output dir: {out_dir}')


if __name__ == '__main__':
    main()
