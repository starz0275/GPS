"""
模型验证脚本 —— 真实测试集（从未参与训练）
使用 260316_Data.csv t>=620s 段，完整评估模型泛化能力。
绘制：
  1. 预测 vs 真实 残差散点
  2. 误差分布直方图
  3. 测试段 2D 轨迹（GPS 真值 vs 纯 DR vs TCN 修正）
  4. 模拟 GPS 失联时误差随时间变化
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from pathlib import Path

keras.config.enable_unsafe_deserialization()

MODEL_PATH  = Path(__file__).parent / "trained_models" / "best_model.keras"
NORM_PATH   = Path(__file__).parent / "preprocessed_data" / "normalization_stats.json"
X_TEST      = Path(__file__).parent / "preprocessed_data" / "X_test.npy"
Y_TEST      = Path(__file__).parent / "preprocessed_data" / "Y_test.npy"
TS_TEST     = Path(__file__).parent / "preprocessed_data" / "ts_test.npy"
TEST_CSV    = Path(__file__).parent / "preprocessed_data" / "test_aligned.csv"
OUT_DIR     = Path(__file__).parent / "trained_models"

TARGET_FREQ = 10


def main():
    print("=" * 65)
    print("TCN Model Evaluation — Held-out Test Set (t >= 620s)")
    print("=" * 65)

    # ---- 加载 ----
    model  = tf.keras.models.load_model(MODEL_PATH, safe_mode=False, compile=False)
    X_test = np.load(X_TEST)
    Y_test = np.load(Y_TEST)
    ts     = np.load(TS_TEST)

    print(f"Test set: {len(X_test)} windows  shape={X_test.shape}")

    # ---- 推理 ----
    Y_pred = model.predict(X_test, batch_size=64, verbose=0)
    err_dx = Y_pred[:, 0] - Y_test[:, 0]
    err_dy = Y_pred[:, 1] - Y_test[:, 1]

    print(f"dx: MAE={np.abs(err_dx).mean():.4f}m  RMSE={np.sqrt((err_dx**2).mean()):.4f}m  "
          f"p50={np.percentile(np.abs(err_dx),50):.4f}  p90={np.percentile(np.abs(err_dx),90):.4f}")
    print(f"dy: MAE={np.abs(err_dy).mean():.4f}m  RMSE={np.sqrt((err_dy**2).mean()):.4f}m  "
          f"p50={np.percentile(np.abs(err_dy),50):.4f}  p90={np.percentile(np.abs(err_dy),90):.4f}")

    # ---- 加载对齐 CSV 做轨迹重建 ----
    df = pd.read_csv(TEST_CSV)
    df = df[df['GPS_valid'] == 1].reset_index(drop=True)   # 只用 GPS 有效行

    # GPS ENU 真值轨迹（累积真实位移）
    gps_x = np.cumsum(df['True_dx'].values)
    gps_y = np.cumsum(df['True_dy'].values)

    # 纯 DR 轨迹（累积 Base 位移）
    dr_x = np.cumsum(df['Base_dx'].values)
    dr_y = np.cumsum(df['Base_dy'].values)

    # TCN 修正轨迹：Base_dx + 预测残差
    # 找对齐到 df 时间戳的预测（按 ts 对应）
    t_df = df['Time_s'].values
    corr_dx = df['Base_dx'].values.copy()
    corr_dy = df['Base_dy'].values.copy()
    for j, t_j in enumerate(ts):
        idx_in_df = np.searchsorted(t_df, t_j)
        if 0 <= idx_in_df < len(df):
            corr_dx[idx_in_df] += Y_pred[j, 0]
            corr_dy[idx_in_df] += Y_pred[j, 1]
    tcn_x = np.cumsum(corr_dx)
    tcn_y = np.cumsum(corr_dy)

    # 位置误差随时间
    dr_err  = np.sqrt((dr_x  - gps_x)**2 + (dr_y  - gps_y)**2)
    tcn_err = np.sqrt((tcn_x - gps_x)**2 + (tcn_y - gps_y)**2)
    t_rel   = np.arange(len(df)) / TARGET_FREQ

    # ======== 绘图 ========
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "TCN Validation — Held-out Test Set (260316, t≥620s, never seen in training)",
        fontsize=12, fontweight='bold')

    # ---- 左上：dx 散点 ----
    ax = axes[0, 0]
    lim = float(np.percentile(np.abs(Y_test[:, 0]), 95))
    ax.scatter(Y_test[:, 0], Y_pred[:, 0], alpha=0.3, s=6, c='steelblue')
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1.5, label='Perfect')
    r2_dx = 1 - float(np.var(err_dx)) / (float(np.var(Y_test[:, 0])) + 1e-9)
    ax.set_xlim(-lim, lim);  ax.set_ylim(-lim, lim)
    ax.set_xlabel('True residual dx (m)');  ax.set_ylabel('Predicted dx (m)')
    ax.set_title('dx: Predicted vs True  [Test Set]')
    ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.93,
            f'MAE={np.abs(err_dx).mean():.4f}m\nR²={r2_dx:.3f}',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ---- 右上：dy 散点 ----
    ax = axes[0, 1]
    lim = float(np.percentile(np.abs(Y_test[:, 1]), 95))
    ax.scatter(Y_test[:, 1], Y_pred[:, 1], alpha=0.3, s=6, c='darkorange')
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1.5, label='Perfect')
    r2_dy = 1 - float(np.var(err_dy)) / (float(np.var(Y_test[:, 1])) + 1e-9)
    ax.set_xlim(-lim, lim);  ax.set_ylim(-lim, lim)
    ax.set_xlabel('True residual dy (m)');  ax.set_ylabel('Predicted dy (m)')
    ax.set_title('dy: Predicted vs True  [Test Set]')
    ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.93,
            f'MAE={np.abs(err_dy).mean():.4f}m\nR²={r2_dy:.3f}',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ---- 左下：误差直方图 ----
    ax = axes[1, 0]
    bins = np.linspace(-0.6, 0.6, 80)
    ax.hist(err_dx, bins=bins, alpha=0.6, density=True, color='steelblue',
            label=f'dx  MAE={np.abs(err_dx).mean():.4f}m')
    ax.hist(err_dy, bins=bins, alpha=0.6, density=True, color='darkorange',
            label=f'dy  MAE={np.abs(err_dy).mean():.4f}m')
    ax.set_xlabel('Prediction Error (m)');  ax.set_ylabel('Density')
    ax.set_title('Error Distribution — Test Set')
    ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
    ax.text(0.97, 0.93,
            f'p50  dx: {np.percentile(np.abs(err_dx),50):.4f}m\n'
            f'p90  dx: {np.percentile(np.abs(err_dx),90):.4f}m\n'
            f'p99  dx: {np.percentile(np.abs(err_dx),99):.3f}m',
            transform=ax.transAxes, fontsize=8.5,
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ---- 右下：2D 轨迹对比 ----
    ax = axes[1, 1]
    ax.plot(gps_x, gps_y, 'g-',  lw=2,   label='GPS Ground Truth')
    ax.plot(dr_x,  dr_y,  'r--', lw=1.5, label=f'Dead Reckoning (final={dr_err[-1]:.1f}m)')
    ax.plot(tcn_x, tcn_y, 'b-',  lw=1.5, label=f'TCN Corrected  (final={tcn_err[-1]:.1f}m)')
    ax.set_xlabel('East (m)');  ax.set_ylabel('North (m)')
    ax.set_title(f'Test Segment Trajectory  (~{len(df)/TARGET_FREQ:.0f}s @ ~71 km/h)')
    ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', 'datalim')

    # 关键时刻误差
    print()
    print(f"{'Time':>6} {'DR err':>10} {'TCN err':>10} {'Improvement':>12}")
    print("─" * 44)
    for t_s in [5, 10, 20, 30, 60]:
        step = min(t_s * TARGET_FREQ, len(df)-1)
        dr_e  = float(dr_err[step])
        tcn_e = float(tcn_err[step])
        impr  = (1 - tcn_e / max(dr_e, 0.01)) * 100
        print(f"{t_s:>5}s  {dr_e:>9.2f}m  {tcn_e:>9.2f}m  {impr:>10.1f}%")

    plt.tight_layout()
    out = OUT_DIR / "validation_result.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Saved: {out}")


if __name__ == "__main__":
    main()
