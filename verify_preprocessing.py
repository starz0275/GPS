"""
预处理数据验证脚本
验证和展示预处理后数据的统计信息
"""

import numpy as np
import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "preprocessed_data"

print("="*80)
print("预处理数据验证")
print("="*80)

# 加载数据
X = np.load(OUTPUT_DIR / "X_train.npy")
Y = np.load(OUTPUT_DIR / "Y_train.npy")
timestamps = np.load(OUTPUT_DIR / "timestamps.npy")

with open(OUTPUT_DIR / "normalization_stats.json") as f:
    config = json.load(f)

print("\n[输入数据形状]")
print(f"  X: {X.shape} (Batch=样本数, 20=时间步, 9=特征数)")
print(f"  Y: {Y.shape} (Batch=样本数, 2=dx和dy残差)")
print(f"  timestamps: {timestamps.shape}")

print("\n[特征列表]")
for i, name in enumerate(config['feature_names']):
    print(f"  [{i}] {name}")

print("\n[时间跨度]")
print(f"  开始时刻: {timestamps[0]:.3f} s")
print(f"  结束时刻: {timestamps[-1]:.3f} s")
print(f"  总时长: {timestamps[-1] - timestamps[0]:.3f} s")
print(f"  频率: {config['target_freq']} Hz")
print(f"  每样本包含: {config['window_size']} 时间步 = {config['window_size'] / config['target_freq']:.1f} s")

print("\n[目标标签 Y 统计]")
print(f"  Y[:, 0] (dx残差):")
print(f"    Mean: {Y[:, 0].mean():.6f} m")
print(f"    Std:  {Y[:, 0].std():.6f} m")
print(f"    Min:  {Y[:, 0].min():.6f} m")
print(f"    Max:  {Y[:, 0].max():.6f} m")

print(f"\n  Y[:, 1] (dy残差):")
print(f"    Mean: {Y[:, 1].mean():.6f} m")
print(f"    Std:  {Y[:, 1].std():.6f} m")
print(f"    Min:  {Y[:, 1].min():.6f} m")
print(f"    Max:  {Y[:, 1].max():.6f} m")

print("\n[输入特征 X 统计（已归一化）]")
for i, name in enumerate(config['feature_names']):
    mean_val = X[:, :, i].mean()
    std_val = X[:, :, i].std()
    min_val = X[:, :, i].min()
    max_val = X[:, :, i].max()
    print(f"  {name}:")
    print(f"    Mean: {mean_val:8.4f}, Std: {std_val:8.4f}, Min: {min_val:8.4f}, Max: {max_val:8.4f}")

print("\n[样本示例]")
print("第一个样本的形状:")
print(f"  样本 0 -> X[0].shape: {X[0].shape}, Y[0]: {Y[0]}")

print("\n[样本时间戳]")
print(f"  第一个样本时间戳: {timestamps[0]:.3f} s")
print(f"  最后一个样本时间戳: {timestamps[-1]:.3f} s")

print("\n[归一化配置]")
print("用于推理时逆标准化的统计信息:")
for feature, stats in config['stats'].items():
    print(f"  {feature}:")
    print(f"    mean={stats['mean']:.6f}, std={stats['std']:.6f}")

print("\n" + "="*80)
print("[OK] 验证完成！数据已准备就绪用于模型训练")
print("="*80)
