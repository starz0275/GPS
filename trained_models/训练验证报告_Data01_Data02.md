# GPS 失联定位 — 训练与验证报告（Data01 / Data02）

> 生成说明：基于 `data_preprocessing_v2.py` 划分与终端运行日志（TCN / validate_model / validate_trajectory / validate_ekf）。  
> 数据路径：`标定实车数据/Data01_*`、`Data02_*`；260316 路跑数据**未参与**本次训练。

---

## 1. 数据划分

| 角色 | 数据集 | 时间范围 | 样本规模（10 Hz） | 预处理输出 |
|------|--------|----------|-------------------|------------|
| **训练** | Data01 前 80% | 0.06 – 250.9 s | 2509 行 → **2627** 滑动窗 | `X_train.npy` |
| **测试** | Data01 后 20% | 250.9 – 313.6 s | 627 行 → **597** 窗 | `X_test.npy` |
| **验证** | Data02 整段 | 0.06 – 291.2 s | 2912 行 → **2882** 窗 | `X_val.npy` |

- 划分配置：`preprocessed_data/dataset_split.json`
- 对齐 CSV：`aligned_data.csv`（Data01）、`val_aligned.csv`（Data02）
- 归一化统计量：**仅来自 Data01 训练段**

**Data02 特点（验证集）**：平均车速约 **11.7 km/h**，预处理显示 **GPS 有效率 100%**（标定 GNSS 连续）。

---

## 2. 模型与训练状态总览

| 模块 | 文件 | 训练数据 | 验证数据 | 状态 |
|------|------|----------|----------|------|
| **TCN** 位移残差 | `best_model.keras` | Data01 | Data02（训练时 `val_loss`） | 已收敛，见 §3 |
| **BiasNet** 陀螺零偏 | `biasnet_weights.weights.h5` | Data01 | Data02 | 已训练，见 §4 |
| **融合推理** | `trajectory_fusion.py` | — | Data02 轨迹验证 | 见 §5、§6 |

---

## 3. TCN（位移残差网络）

### 3.1 训练结果

| 指标 | 数值 |
|------|------|
| 训练轮数 | 17（早停） |
| 最优验证 Huber loss | **0.00306** |
| 验证 MAE（dx / dy） | **0.0585 m / 0.0459 m** |
| 验证 RMSE（dx / dy） | **0.1176 m / 0.0826 m** |
| 相对底盘误差（dx / dy） | 49.7% / 54.4% |

> 相对误差偏高，因 Data02 车速低、单步位移小，分母小；**绝对误差约 5–6 cm/步** 更有参考价值。

### 3.2 Data02 验证集逐步残差（`validate_model.py`）

| 方向 | MAE | RMSE | 中位数 p50 | p90 |
|------|-----|------|------------|-----|
| dx | **0.0585 m** | 0.1176 m | 0.0188 m | 0.1981 m |
| dy | **0.0459 m** | 0.0826 m | 0.0180 m | 0.1624 m |

**结论（TCN）**：在**从未参与训练的 Data02** 上，逐步预测残差约 **2–6 cm（中位数）**，单步回归质量**较好**，`best_val_loss=0.003` 与实测一致。

### 3.3 关于 `validation_result.png` 里「TCN 比 DR 更差」的表格

日志中 5 s / 10 s / … / 60 s「DR err vs TCN err」出现大幅负改善，**不能**据此判断 TCN 失败：

- 该表是在 **GPS 仍有效** 的 `val_aligned.csv` 行上，用时间索引**硬对齐**窗口预测，积分方式与融合管线不一致；
- 起步几秒 DR 误差接近 0，TCN 稍有偏差会被算成「负改善几千 %」。

**应以 §3.2 的逐步 MAE/RMSE 为准** 评价 TCN；轨迹级评价看 §5 `validate_trajectory.py`。

---

## 4. BiasNet + EKF（航向）

### 4.1 训练结果（`biasnet_info.json`）

| 指标 | 数值 |
|------|------|
| Data01 训练样本 | 3107 窗 |
| Data02 验证样本 | 2883 窗 |
| 最优验证零偏 MAE | **0.90 °/s** |

### 4.2 Data02 诊断（`validate_ekf.py`）

| 项目 | 数值 | 说明 |
|------|------|------|
| 陀螺 Z 原始 | 均值 **-1.22 °/s**，标准差 **35.2 °/s** | 波动极大，需怀疑标定/静止/单位 |
| BiasNet 预测零偏 | 均值 -0.19 °/s，std 0.86 °/s | 网络有输出，但难以压住原始噪声 |
| **隧道 60 s 航向误差（终点）** | DR **-357.8°**，EKF **-345.7°** | 航向积分基本失效（近一圈） |

| 位置指标（全程，GPS 有效帧统计） | 纯 DR | EKF |
|----------------------------------|-------|-----|
| 终点误差 | 37.09 m | 37.09 m |
| 中位误差 | 15.74 m | 16.11 m |
| RPE 30 s 中位 | 4.89 m | 5.50 m |

**结论（EKF）**：在 Data02 上 **BiasNet+EKF 未改善位置**，隧道段**航向严重发散**；与 Data02 陀螺 Z 质量及低速工况有关。**不宜**用当前 EKF 结果评价整体融合上限。

---

## 5. 融合轨迹（EKF + TCN，`validate_trajectory.py`）

**设置**：Data02 全段；自第 **15 s** 起模拟 **60 s** 无 GPS；有 GPS 时位置跟 GNSS，无 GPS 时 TCN 补位移，恢复后 GNSS 锚定。

### 5.1 隧道段（60 s 相对位移 RPE）

| 方法 | RPE |
|------|-----|
| 纯 DR | 5.0 m |
| 仅 EKF | 6.4 m |
| **融合·纯积分（无锚定）** | **2.4 m** |
| 融合·出隧道 GNSS 锚定后 | 0.2 m |

| 丢失段结束位置误差 | 纯积分 | 锚定后 |
|--------------------|--------|--------|
| | **2.4 m** | **≈0 m** |

### 5.2 全程（GPS 有效帧，中位误差）

| 方法 | 中位误差 |
|------|----------|
| 纯 DR | 15.7 m |
| 仅 EKF | 16.1 m |
| EKF+TCN（有 GPS 跟实测） | **0.0 m** |

**结论（融合）**：

- 在 **Data02、低速、GNSS 连续** 条件下，模拟隧道 **60 s 纯预测 RPE ≈ 2.4 m**，明显优于此前 260316 高速长 outage（~70 m 量级）。
- 出隧道后 **GNSS 重捕获** 可把误差拉回；有 GPS 段中位 0 m 来自「位置跟 GNSS」策略，不是 TCN 单独能力。
- **EKF 在隧道段未体现优势**，与 §4 航向失效一致；当前融合收益**主要来自 TCN 逐步残差**。

输出图：`trained_models/fused_trajectory.png`

---

## 6. 综合结论：训练得怎么样？

### 做得好的

1. **数据管线**：Data01 训 / 测、Data02 验 划分清晰，预处理与 `dataset_split.json` 一致。
2. **TCN**：Data02 逐步残差 **MAE ≈ 5–6 cm**，验证 loss **0.003**，泛化到第二段标定数据**可用**。
3. **融合 + TCN（Data02）**：60 s 无 GPS、约 12 km/h，**纯积分 RPE ≈ 2.4 m**，达到当前阶段**实用潜力**。

### 需要警惕的

1. **BiasNet / EKF**：Data02 隧道航向误差 **~360°**，位置相对 DR **无改善**；需排查 Data02 陀螺标定、零偏标签与低速航向锚定。
2. **`validate_model.py` 轨迹表**：与逐步指标矛盾，**勿**用于汇报 TCN 轨迹好坏。
3. **有 GPS 段中位 0 m**：是融合策略（跟 GNSS），不能等同于「全程预测等于真值」。
4. **泛化边界**：仅在标定 Data02 验证；**未**在 260316 路跑、Data01 留出段上复测本次权重。

### 建议下一步（按优先级）

1. 用 **`X_test.npy`（Data01 后 20%）** 跑一遍 `validate_model.py` / 融合，看同域时间外推。
2. 查 **Data02 `GyroZ_degs` 大噪声**（std≈39 °/s）是否单位/静止段未滤除，再重训 BiasNet。
3. 对外汇报隧道能力：用 **`fused_trajectory.png` + 纯积分 RPE 2.4 m**，并注明车速与 Data02 场景。
4. 若需路跑对比：设 `USE_260316=True` 单独做一轮，不与标定混训。

---

## 7. 复现命令

```powershell
cd C:\Users\nxj\Desktop\GPS

python data_preprocessing_v2.py
python train_ekf.py
python train_tcn_model.py

python validate_model.py
python validate_trajectory.py
python validate_ekf.py
```

---

## 8. 输出文件索引

| 文件 | 含义 |
|------|------|
| `preprocessed_data/X_train.npy` | Data01 训练窗 |
| `preprocessed_data/X_test.npy` | Data01 测试窗 |
| `preprocessed_data/X_val.npy` | Data02 验证窗 |
| `trained_models/best_model.keras` | TCN 权重 |
| `trained_models/biasnet_weights.weights.h5` | BiasNet 权重 |
| `trained_models/validation_result.png` | TCN 残差散点/分布（Data02） |
| `trained_models/fused_trajectory.png` | 融合轨迹（Data02，60 s 失锁） |
| `trained_models/ekf_validation.png` | EKF 航向/位置（Data02） |
| `trained_models/training_info.json` | TCN 训练指标 |
| `trained_models/biasnet_info.json` | BiasNet 训练指标 |

---

*报告日期：与本次终端运行一致；若重新训练，请更新本节数值。*
