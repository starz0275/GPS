# GPS 失联定位项目技术总结

> 目标：企业级地面车辆在 GPS 丢失场景（隧道、地下停车场等）下的高精度定位
> 方法演进：TCN 残差学习 → BiasNet + 卡尔曼滤波器（EKF）

---

## 一、数据来源

| 数据集 | 格式 | 时长 | 用途 |
|--------|------|------|------|
| `标定实车数据/Data01_*.txt` | Tab 分隔，3 传感器文件 | ~313 s | 训练 |
| `标定实车数据/Data02_*.txt` | 同上 | ~291 s | 训练 |
| `260316_Data/260316_Data.csv` | 单 CSV | ~701 s | 训练（t < 600 s）+ 测试（t ≥ 620 s） |

**传感器通道（10 Hz）：**
- IMU：AccX/Y/Z (g)、GyroX/Y/Z (deg/s)
- 车速：VehicleSpeed（轮速计，km/h）
- GNSS：纬度、经度、高度、航向角（仅标定数据有效）

---

## 二、代码文件一览

```
GPS/
├── data_preprocessing.py       # V1 预处理（已弃用）
├── data_preprocessing_v2.py    # V2 预处理（主力）
├── build_tcn_model.py          # TCN 模型定义
├── train_tcn_model.py          # TCN 训练脚本
├── validate_model.py           # TCN 验证脚本
│
├── ekf_navigator.py            # BiasNet + EKF 核心（最终方案）
├── train_ekf.py                # BiasNet 训练脚本
├── validate_ekf.py             # EKF 验证脚本
│
├── preprocessed_data/
│   └── normalization_stats.json
└── trained_models/
    ├── best_model.keras         # TCN 权重
    ├── biasnet_weights.weights.h5  # BiasNet 权重
    ├── biasnet_info.json
    └── *.png                    # 验证图像
```

---

## 三、第一阶段：数据预处理 V2（`data_preprocessing_v2.py`）

### 3.1 核心改进（相较 V1）

| 改进项 | 说明 |
|--------|------|
| **GPS 跳点清洗** | 计算相邻 GPS 点隐含速度，超过 150 km/h 的点用线性插值替换 |
| **GPS 航向锚定** | GPS 有效时直接用 GPS 航向覆盖陀螺积分结果，从根本上消除漂移 |
| **非完整约束（NHC）** | 位置累积仅用前向车速（`Δx = v·cos θ·dt`），零侧向位移 |
| **双数据源合并** | 标定数据 + 260316 数据（训练段）拼接为统一训练集 |
| **时序测试集切分** | 260316 数据 t ≥ 620 s 部分完全隔离为测试集，不参与训练 |

### 3.2 关键参数

```
TARGET_FREQ       = 10 Hz
WINDOW_SIZE       = 30 帧（3 秒时序窗口）
GPS_MAX_SPEED_KMH = 150 km/h（跳点阈值）
REAL_TRAIN_T_MAX  = 600 s（260316 训练截止）
REAL_TEST_T_START = 620 s（测试段起始）
```

### 3.3 输出文件

```
preprocessed_data/
├── X_train.npy   (N, 30, 9)  — 输入窗口
├── Y_train.npy   (N, 2)      — 残差标签 (dx, dy)
├── X_test.npy                — 测试集输入
├── Y_test.npy                — 测试集标签
├── ts_test.npy               — 测试时间戳
├── test_aligned.csv          — 测试原始对齐数据
└── normalization_stats.json  — 归一化均值/标准差
```

---

## 四、第二阶段：TCN 残差学习（已验证但存在局限）

### 4.1 模型架构（`build_tcn_model.py`）

```
输入: (30帧, 9特征) — IMU 6通道 + 车速 + Base_dx + Base_dy

初始投影层: Conv1D(128, 3, causal)
                   ↓
深度残差塔（5层，膨胀因果卷积）:
  Block 1: dilation=1   → 感受野 3 帧
  Block 2: dilation=2   → 感受野 7 帧
  Block 3: dilation=4   → 感受野 15 帧
  Block 4: dilation=8   → 感受野 31 帧
  Block 5: dilation=16  → 感受野 63 帧
                   ↓
取最后时间步 → Dense(128) → Dense(64) → Dense(32)
                   ↓
输出: (2,) — 预测位移残差 (Δdx, Δdy)

总参数量: ~300K+
损失函数: Huber Loss (δ = 0.1)
```

### 4.2 训练指标

| 指标 | 训练集 | 验证集 |
|------|--------|--------|
| Huber Loss | 0.00612 | 0.00781 |
| MAE | 0.114 m | 0.101 m |
| RMSE | 0.333 m | 0.296 m |
| dx 相对误差 | — | **29.0%** |
| dy 相对误差 | — | **26.9%** |

训练轮次：20 epochs（含早停）

### 4.3 TCN 局限性（为什么转向 EKF）

TCN 本质上预测**逐帧位移残差**，无法直接修正**累积航向误差**。
实测发现：

- 陀螺 Z 轴零偏导致每秒约 0.1–0.5°航向漂移
- 60 秒 GPS 丢失后，航向误差累积 6°–30°
- 一个 15°航向误差，在 100 m 行驶后产生 **26 m 横向偏差**
- TCN 预测的位移残差无法补偿这种系统性航向漂移

---

## 五、第三阶段：BiasNet + EKF（最终方案）

参考 **ai-imu-dr**（Martin et al. 2021），针对二维地面车辆简化实现。

### 5.1 核心思路

```
根本问题：陀螺 Z 轴零偏 b_ω 随时间漂移 → 航向累积误差

解决方案（分层）：
  层 1 —— BiasNet（神经网络）：
      输入: IMU 窗口 (30帧 × 6通道)
      输出: 当前陀螺 Z 轴零偏估计 b̂_ω (rad/s)
      训练: 监督回归，标签 = ω_z测量 − ω_GPS推算（真实零偏）

  层 2 —— EKF（卡尔曼滤波器）：
      状态: x = [θ, δb_ω]  (航向 + 残余零偏)
      传播: θ ← θ + (ω_z − b̂_net − δb_ω) × dt
      量测: GPS 有效时 z = θ_GPS，反推精细校正 δb_ω
      效果: GPS 丢失时维持最优状态估计

  层 3 —— NHC 位置累积：
      Δx = v × cos(θ) × dt
      Δy = v × sin(θ) × dt   （零侧向位移假设）
```

### 5.2 BiasNet 架构（`ekf_navigator.py`）

```
输入: (batch, 30, 6) — 归一化 IMU 窗口

Conv1D(32, 5, causal, relu)           # 局部特征
Conv1D(32, 5, dilation=4, causal, relu)  # 1.2 s 感受野
Conv1D(16, 3, causal, relu)
GlobalAveragePooling1D()
Dense(32, relu) → Dropout(0.2)
Dense(1, linear)  ← 输出: b̂_ω (rad/s)

总参数量: 8,273  (~32 KB)
```

### 5.3 EKF 状态方程

**过程模型（时间更新）：**

```
F = [[1, -dt],    Q = [[0,    0   ],
     [0,  1 ]]         [0, q_δb_ω]]

x_pred = F × x_k
P_pred = F × P_k × Fᵀ + Q
```

**量测模型（GPS 有效时更新）：**

```
H = [1, 0]    （只观测航向）
S = P_pred[0,0] + r_θ
K = P_pred[:,0] / S        （卡尔曼增益）
x_upd = x_pred + K × (θ_GPS − θ_pred)
P_upd = (I − K×H) × P_pred
```

### 5.4 训练指标（BiasNet，`train_ekf.py`）

| 指标 | 值 |
|------|----|
| 训练样本数 | 10,167 |
| 验证样本数 | 1,794 |
| 最优验证 MAE | **0.430 deg/s** (0.0075 rad/s) |
| 验证集预测均值误差 | 0.008 deg/s（近乎无偏） |
| 验证集预测标准差 | 0.719 deg/s |
| 验证集 p95 误差 | 1.646 deg/s |
| 训练轮次 | 80 epochs |
| 损失函数 | Huber Loss |
| 优化器 | Adam (lr=3e-4) |

---

## 六、最终验证结果

### 6.1 测试设置

- **测试数据**：260316_Data.csv，t = 620–701 s（81 秒，约 1.4 分钟）
- **模拟场景**：在测试段 t+15s 处屏蔽 GPS 信号持续 **60 秒**，模拟隧道穿越
- **GPS 真值**：完整 GPS ENU 坐标（参考）
- **对比方案**：纯陀螺积分 DR vs BiasNet + EKF

### 6.2 量化指标

| 指标 | 纯陀螺 DR | BiasNet + EKF | 改善 |
|------|-----------|----------------|------|
| 终点位置误差 | **93.52 m** | **18.73 m** | **↓ 80%** |
| 中值位置误差 | 30.73 m | 13.05 m | ↓ 58% |

> 测试段全程 81 s，其中 GPS 丢失持续 60 s（占 74%）。

### 6.3 结果图解读（`trained_models/ekf_validation.png`）

| 子图 | 说明 |
|------|------|
| 左上·轨迹对比 | EKF（蓝）紧贴 GPS 真值（黑），纯 DR（红虚线）在 GPS 丢失区间（橙色）明显偏离 |
| 右上·累积误差 | GPS 丢失后 DR 误差迅速攀升至 93 m，EKF 维持在 ~18 m |
| 左下·航向对比 | 纯陀螺积分在 GPS 丢失后航向漂移严重；EKF 利用 BiasNet 零偏校正，航向基本跟随真值 |
| 右下·预测零偏 | BiasNet 实时估计陀螺 Z 轴零偏，典型值 -0.25 ~ +0.1 deg/s |

---

## 七、运行流程

```bash
# 步骤 1：数据预处理（生成训练/测试 npy 文件）
python data_preprocessing_v2.py

# 步骤 2a：训练 TCN（早期方案，可选）
python train_tcn_model.py

# 步骤 2b：训练 BiasNet（最终方案，约 90 秒）
python train_ekf.py

# 步骤 3：验证 EKF 导航器（约 20 秒，生成 ekf_validation.png）
python validate_ekf.py
```

---

## 八、技术结论与展望

### 8.1 结论

| 问题 | TCN 方案 | BiasNet+EKF 方案 |
|------|---------|-----------------|
| 短期(<5s) 位移预测 | ✓ MAE ~0.1m | 不适用（面向航向） |
| 长期(>30s) 航向维持 | ✗ 无航向修正 | ✓ 零偏补偿，漂移大幅降低 |
| GPS 丢失 60s 终点误差 | ~93 m | **~19 m** |
| 模型大小 | ~300K 参数 | **8,273 参数** |
| 物理可解释性 | 低 | 高（零偏+EKF 有明确物理意义） |

**BiasNet + EKF 是更适合企业真实场景的方案**：轻量（仅 32 KB）、可解释、GPS 丢失鲁棒。

### 8.2 后续改进方向

1. **更长 GPS 丢失测试**：采集实际隧道（>120 s）数据验证
2. **在线自适应零偏**：GPS 重获后更新 EKF 状态，持续在线标定
3. **地图辅助**（Map Matching）：结合高精地图进一步约束累积误差
4. **量化部署**：BiasNet 仅 8K 参数，可直接 INT8 量化部署至 S32K5 Neutron NPU

---

*生成时间：2026-05-18*
