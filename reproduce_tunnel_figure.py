"""
复现 tunnel_trajectory.png 左图绘制逻辑（逐步说明用）

绿线不是直接画经纬度，而是：
  从 (0,0) 出发，每 0.1s 累加 (Base_dx + Target_dx, Base_dy + Target_dy)

与 real_gnss_trajectory.png 同一段路（t≈103–163s）对比。
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

W = 30
START_WIN = 1000
LENGTH = 600

ROOT = Path(__file__).parent
OUT = ROOT / 'trained_models' / 'tunnel_trajectory_reproduced.png'


def main():
    df = pd.read_csv(ROOT / 'preprocessed_data' / 'aligned_data.csv')
    X = np.load(ROOT / 'preprocessed_data' / 'X_train.npy')
    model = tf.keras.models.load_model(
        ROOT / 'trained_models/best_model.keras', safe_mode=False, compile=False)

    # === 与 simulate_tunnel.py 完全相同的累加方式 ===
    true_x, true_y = [0.0], [0.0]
    fp32_x, fp32_y = [0.0], [0.0]
    cx = cy = fx = fy = 0.0

    for step, i in enumerate(range(START_WIN, START_WIN + LENGTH)):
        win = X[i : i + 1].astype(np.float32)
        pred = model(win, training=False)
        ri = i + W
        if ri >= len(df):
            ri = len(df) - 1

        base_dx = df['Base_dx'].iloc[ri]
        base_dy = df['Base_dy'].iloc[ri]
        true_dx = df['Target_dx'].iloc[ri]
        true_dy = df['Target_dy'].iloc[ri]

        cx += base_dx + true_dx
        cy += base_dy + true_dy
        true_x.append(cx)
        true_y.append(cy)

        fx += base_dx + float(pred[0, 0])
        fy += base_dy + float(pred[0, 1])
        fp32_x.append(fx)
        fp32_y.append(fy)

    true_x, true_y = np.array(true_x), np.array(true_y)
    fp32_x, fp32_y = np.array(fp32_x), np.array(fp32_y)

    # 同时间段真实 ENU（aligned_data，首点对齐）
    i0, i1 = START_WIN + W, START_WIN + W + LENGTH - 1
    t0, t1 = df['Time_s'].iloc[i0], df['Time_s'].iloc[i1]
    enu_x = df['ENU_x'].iloc[i0 : i1 + 1].values
    enu_y = df['ENU_y'].iloc[i0 : i1 + 1].values
    enu_x -= enu_x[0]
    enu_y -= enu_y[0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f'How tunnel_trajectory.png is drawn  (t={t0:.0f}–{t1:.0f}s, same as simulate_tunnel.py)',
        fontsize=12, fontweight='bold')

    axes[0].plot(enu_x, enu_y, 'g-', lw=2, label='Real ENU (aligned)')
    axes[0].set_title('Real road shape — not figure-8')
    axes[0].axis('equal')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(true_x, true_y, 'g-', lw=3, label='Green in tunnel_trajectory')
    axes[1].set_title('Green = cumsum(Base+Target) from (0,0)')
    axes[1].axis('equal')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(true_x, true_y, 'g-', lw=3, alpha=0.7, label='Green (synthetic)')
    axes[2].plot(fp32_x, fp32_y, color='darkorange', ls='-.', lw=2,
                 label='Orange (FP32 TCN)')
    axes[2].set_title('Same as tunnel_trajectory left panel')
    axes[2].axis('equal')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT, dpi=200, bbox_inches='tight')
    plt.close()

    print(f'[OK] {OUT}')
    print(f'时间 {t0:.1f}–{t1:.1f}s, 绿线点数 {len(true_x)}')
    print(f'绿线 vs 真实ENU 终点偏差 '
          f'{np.hypot(true_x[-1]-enu_x[-1], true_y[-1]-enu_y[-1]):.2f} m')
    print('说明: 磁盘上旧的 tunnel_trajectory.png 还画了 INT8 蓝虚线;')
    print('      当前 INT8 模型输入维度和 TCN 不一致(20 vs 30)，脚本已跑不通。')
    print('      若旧图呈「8」字，多半是当时 INT8 预测发散打圈，不是绿线真值。')


if __name__ == '__main__':
    main()
