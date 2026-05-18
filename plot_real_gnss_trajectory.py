"""
plot_real_gnss_trajectory.py — 画 260316 真实 GNSS 轨迹

与 simulate_tunnel.py（tunnel_trajectory.png）使用相同时间段 t≈103–163 s，
对比：
  1) 原始 CSV 经纬度转 ENU（真实测量）
  2) 预处理 aligned_data.csv 里的 ENU
  3) simulate_tunnel 那种从 (0,0) 逐步积分的「合成绿线」

输出: trained_models/real_gnss_trajectory.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

from trajectory_data import load_segment, DATA_CSV, TARGET_DT

# 与 simulate_tunnel.py 一致：窗口起点 1000、长 600 步 → 对齐到时间轴
TUNNEL_WINDOW_START = 1000
TUNNEL_STEPS = 600
WINDOW_SIZE = 30
ALIGNED_CSV = Path(__file__).parent / 'preprocessed_data' / 'aligned_data.csv'
OUT_PNG = Path(__file__).parent / 'trained_models' / 'real_gnss_trajectory.png'


def tunnel_time_range():
    """从 aligned_data 反查 simulate_tunnel 用的时间段。"""
    df = pd.read_csv(ALIGNED_CSV)
    i0 = TUNNEL_WINDOW_START + WINDOW_SIZE
    i1 = i0 + TUNNEL_STEPS - 1
    i1 = min(i1, len(df) - 1)
    t0 = float(df['Time_s'].iloc[i0])
    t1 = float(df['Time_s'].iloc[i1])
    return t0, t1, i0, i1


def simulate_tunnel_synthetic(df_aligned, i0, i1):
    """复现 simulate_tunnel 里的绿色真值轨迹（从原点累加）。"""
    true_x, true_y = [0.0], [0.0]
    cx, cy = 0.0, 0.0
    for ri in range(i0, i1 + 1):
        cx += float(df_aligned['Base_dx'].iloc[ri])
        cy += float(df_aligned['Base_dy'].iloc[ri])
        cx += float(df_aligned['Target_dx'].iloc[ri])
        cy += float(df_aligned['Target_dy'].iloc[ri])
        true_x.append(cx)
        true_y.append(cy)
    return np.array(true_x), np.array(true_y)


def align_xy(x, y):
    """平移到首点为零，便于和合成轨迹对比形状。"""
    m = np.isfinite(x) & np.isfinite(y)
    if not m.any():
        return x, y
    i0 = int(np.where(m)[0][0])
    ox, oy = float(x[i0]), float(y[i0])
    return x - ox, y - oy


def main():
    print('=' * 60)
    print('真实 GNSS 轨迹可视化（对比 tunnel_trajectory 合成路径）')
    print('=' * 60)

    t0, t1, i0, i1 = tunnel_time_range()
    print(f'  simulate_tunnel 对应时间: {t0:.1f} – {t1:.1f} s ({TUNNEL_STEPS} 步)')

    # --- 1. 原始 CSV → ENU（trajectory_data 逻辑）---
    seq = load_segment(t_start=t0, t_end=t1 + TARGET_DT)
    tg = seq['Time_s']
    gx = seq['enu_x_truth']
    gy = seq['enu_y_truth']
    gps_ok = seq['gps_valid'] & np.isfinite(gx) & np.isfinite(gy)
    print(f'  原始网格: {len(tg)} 帧, GNSS有效 {gps_ok.mean():.1%}')

    # --- 2. aligned_data 同时间段 ---
    df = pd.read_csv(ALIGNED_CSV)
    sl = df[(df['Time_s'] >= t0) & (df['Time_s'] <= t1)].copy()
    ax_enu = sl['ENU_x'].values.astype(float)
    ay_enu = sl['ENU_y'].values.astype(float)

    # --- 3. simulate_tunnel 合成路径 ---
    sx, sy = simulate_tunnel_synthetic(df, i0, i1)

    # 对齐到各自首点，只看形状
    rx, ry = align_xy(gx.copy(), gy.copy())
    px, py = align_xy(ax_enu.copy(), ay_enu.copy())

    # 全帧绘图
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(
        f'260316 Real GNSS vs tunnel_trajectory synthetic  (t = {t0:.0f}–{t1:.0f} s)',
        fontsize=14, fontweight='bold')

    # (a) 真实 GNSS：只画有效点，不插值
    ax = axes[0]
    if gps_ok.any():
        ax.plot(gx[gps_ok], gy[gps_ok], 'g.-', ms=2, lw=1.2,
                label=f'Raw GNSS ({gps_ok.sum()} pts)')
    ax.plot(rx[gps_ok], ry[gps_ok], 'k--', lw=0.8, alpha=0.5,
            label='Same, origin at 1st fix')
    ax.set_title('(a) Real GNSS from 260316_Data.csv')
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.35)

    # (b) 预处理 aligned ENU
    ax = axes[1]
    ax.plot(ax_enu, ay_enu, 'c.-', ms=2, lw=1.2,
            label=f'aligned_data ENU ({len(sl)} pts)')
    ax.plot(px, py, 'k--', lw=0.8, alpha=0.5, label='Origin at 1st point')
    ax.set_title('(b) Preprocessed aligned_data.csv')
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.35)

    # (c) simulate_tunnel 合成「8字」来源
    ax = axes[2]
    ax.plot(sx, sy, 'g-', lw=2.5,
            label='tunnel_trajectory green\\n(Base+Target cumsum from 0,0)')
    if gps_ok.any():
        ax.plot(rx[gps_ok], ry[gps_ok], 'm--', lw=1.2, alpha=0.85,
                label='Real GNSS (aligned shape)')
    ax.set_title('(c) Why tunnel_trajectory looks like a loop')
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.legend(fontsize=7)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.35)

    plt.tight_layout()
    OUT_PNG.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'\n[OK] {OUT_PNG}')

    # 形状统计
    def path_length(x, y, mask=None):
        if mask is None:
            mask = np.isfinite(x) & np.isfinite(y)
        idx = np.where(mask)[0]
        if len(idx) < 2:
            return 0.0
        d = 0.0
        for k in range(1, len(idx)):
            i, j = idx[k - 1], idx[k]
            d += np.hypot(x[j] - x[i], y[j] - y[i])
        return d

    print(f'  路径长度: 真实GNSS {path_length(gx, gy, gps_ok):.0f} m  '
          f'aligned {path_length(ax_enu, ay_enu):.0f} m  '
          f'合成绿线 {path_length(sx, sy):.0f} m')


if __name__ == '__main__':
    main()
