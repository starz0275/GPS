"""
模型结果可视化
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import tensorflow as tf
import keras
import json
from pathlib import Path

keras.config.enable_unsafe_deserialization()

MODEL_DIR  = Path(__file__).parent / "trained_models"
DATA_DIR   = Path(__file__).parent / "preprocessed_data"
OUT_DIR    = MODEL_DIR

# ---- 加载 ----
model = tf.keras.models.load_model(
    MODEL_DIR / "best_model.keras", safe_mode=False, compile=False)

X = np.load(DATA_DIR / "X_train.npy")
Y = np.load(DATA_DIR / "Y_train.npy")

np.random.seed(42)
idx   = np.random.permutation(len(X))
split = int(len(X) * 0.8)
X_val = X[idx[split:]]
Y_val = Y[idx[split:]]

Y_pred = model.predict(X_val, verbose=0)
err_dx = Y_pred[:, 0] - Y_val[:, 0]
err_dy = Y_pred[:, 1] - Y_val[:, 1]

with open(MODEL_DIR / "training_info.json") as f:
    info = json.load(f)

# ============================================================
# 图 1：训练曲线（直接展示已保存的 training_history.png）
# ============================================================
# （该文件已在训练结束时自动保存，这里不再重复）

# ============================================================
# 图 2：预测 vs 真实（散点图）
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
lim = 2.0
for ax, (name, yv, yp) in zip(axes, [
        ('dx (m/step)', Y_val[:, 0], Y_pred[:, 0]),
        ('dy (m/step)', Y_val[:, 1], Y_pred[:, 1])]):
    ax.scatter(yv, yp, s=2, alpha=0.3, color='steelblue')
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1.5, label='理想线')
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(f'真实 {name}'); ax.set_ylabel(f'预测 {name}')
    mae = np.abs(yv - yp).mean()
    ax.set_title(f'{name}  MAE={mae:.4f}m')
    ax.legend(); ax.grid(True, alpha=0.3)
plt.suptitle('预测 vs 真实（验证集）', fontsize=14)
plt.tight_layout()
plt.savefig(OUT_DIR / "pred_vs_true.png", dpi=120)
print("[OK] pred_vs_true.png")

# ============================================================
# 图 3：误差分布直方图
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (name, e) in zip(axes, [('dx', err_dx), ('dy', err_dy)]):
    clip = np.clip(e, -1.5, 1.5)
    ax.hist(clip, bins=80, color='steelblue', edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='r', lw=1.5)
    ax.set_xlabel(f'误差 {name} (m)')
    ax.set_ylabel('频次')
    p50 = np.percentile(np.abs(e), 50)
    p90 = np.percentile(np.abs(e), 90)
    ax.set_title(f'{name}  p50={p50:.4f}m  p90={p90:.4f}m')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.5, 1.5)
plt.suptitle('预测误差分布（验证集，截断至 ±1.5m）', fontsize=14)
plt.tight_layout()
plt.savefig(OUT_DIR / "error_distribution.png", dpi=120)
print("[OK] error_distribution.png")

# ============================================================
# 图 4：模拟 GPS 失联轨迹（用验证集连续片段）
# ============================================================
# 取一段连续的 200 步（20s）验证数据来模拟轨迹
SEG_LEN  = 200   # 步
GPS_LOSS_START = 30   # 第 3s 开始失联
GPS_LOSS_END   = 130  # 第 13s 恢复

seg_start = 0
Xseg = X_val[seg_start:seg_start + SEG_LEN]
Yseg = Y_val[seg_start:seg_start + SEG_LEN]   # 真实残差
Pseg = model.predict(Xseg, verbose=0)          # 预测残差

# 从特征中取归一化的 Base_dx / Base_dy（第 7、8 列，索引 7、8）
with open(DATA_DIR / "normalization_stats.json") as f:
    norm = json.load(f)['stats']

# 反归一化 Base_dx, Base_dy（特征最后两列）
base_dx_norm = Xseg[:, -1, 7]  # 窗口最后一步的 Base_dx_norm
base_dy_norm = Xseg[:, -1, 8]  # 窗口最后一步的 Base_dy_norm
base_dx = base_dx_norm * norm['Base_dx']['std'] + norm['Base_dx']['mean']
base_dy = base_dy_norm * norm['Base_dy']['std'] + norm['Base_dy']['mean']

# 三条轨迹
true_dx_arr  = base_dx + Yseg[:, 0]     # 真实位移 = base + 真实残差
dead_dx_arr  = base_dx                   # 纯航位推算（无修正）
model_dx_arr = base_dx + Pseg[:, 0]     # 模型修正后

true_dy_arr  = base_dy + Yseg[:, 1]
dead_dy_arr  = base_dy
model_dy_arr = base_dy + Pseg[:, 1]

# 积分成轨迹
def integrate(dx, dy):
    return np.cumsum(dx), np.cumsum(dy)

tx, ty   = integrate(true_dx_arr,  true_dy_arr)
dx_dr, dy_dr = integrate(dead_dx_arr,  dead_dy_arr)
mx, my   = integrate(model_dx_arr, model_dy_arr)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.plot(tx,    ty,    'g-',  lw=2,   label='GPS 真实轨迹')
ax.plot(dx_dr, dy_dr, 'r--', lw=1.5, label='纯航位推算（无修正）')
ax.plot(mx,    my,    'b-',  lw=1.5, label='TCN 模型修正后')
# 标记 GPS 失联区域
ax.axvspan(tx[GPS_LOSS_START], tx[min(GPS_LOSS_END, len(tx)-1)],
           alpha=0.12, color='gray', label='GPS 失联区（10s）')
ax.set_xlabel('East (m)'); ax.set_ylabel('North (m)')
ax.set_title('模拟 GPS 失联轨迹对比')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# 右图：误差随时间变化
t_axis = np.arange(SEG_LEN) * 0.1  # 秒
err_dr    = np.sqrt((dx_dr - tx)**2 + (dy_dr - ty)**2)
err_model = np.sqrt((mx    - tx)**2 + (my    - ty)**2)
ax2 = axes[1]
ax2.plot(t_axis, err_dr,    'r--', lw=1.5, label='纯航位推算误差')
ax2.plot(t_axis, err_model, 'b-',  lw=1.5, label='TCN 修正后误差')
ax2.axvspan(GPS_LOSS_START * 0.1, GPS_LOSS_END * 0.1,
            alpha=0.12, color='gray', label='GPS 失联区')
ax2.set_xlabel('时间 (s)'); ax2.set_ylabel('累计位置误差 (m)')
ax2.set_title('位置误差随时间变化')
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.suptitle('GPS 失联场景仿真（验证集片段）', fontsize=14)
plt.tight_layout()
plt.savefig(OUT_DIR / "tunnel_simulation.png", dpi=120)
print("[OK] tunnel_simulation.png")

# ============================================================
# 图 5：误差 CDF（累积分布函数）
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
for name, e in [('dx', err_dx), ('dy', err_dy)]:
    abs_e = np.sort(np.abs(e))
    cdf   = np.arange(1, len(abs_e)+1) / len(abs_e)
    ax.plot(abs_e, cdf, lw=2, label=f'|误差 {name}|')

ax.axhline(0.5,  color='gray', ls=':', lw=1)
ax.axhline(0.9,  color='gray', ls=':', lw=1)
ax.axhline(0.99, color='gray', ls=':', lw=1)
ax.text(0.01, 0.51, 'p50', fontsize=9, color='gray')
ax.text(0.01, 0.91, 'p90', fontsize=9, color='gray')
ax.text(0.01, 0.995,'p99', fontsize=9, color='gray')
ax.set_xlim(0, 1.0)
ax.set_xlabel('绝对误差 (m)')
ax.set_ylabel('累积比例')
ax.set_title('误差累积分布函数（CDF）')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "error_cdf.png", dpi=120)
print("[OK] error_cdf.png")

print()
print("所有图已保存到 trained_models/")
print(f"  - training_history.png  (训练曲线)")
print(f"  - pred_vs_true.png      (预测 vs 真实散点)")
print(f"  - error_distribution.png (误差分布直方图)")
print(f"  - tunnel_simulation.png  (GPS失联轨迹仿真)")
print(f"  - error_cdf.png          (误差CDF)")
