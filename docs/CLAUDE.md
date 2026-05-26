# CLAUDE.md

此文件为 Claude Code（claude.ai/code）在本仓库中工作时提供指导。

## 项目概述

基于零偏校正惯性导航的 GNSS 拒止环境车辆定位系统。两阶段方案：(1) **BiasNet** — 轻量 CNN，从 IMU 滑窗预测实时陀螺 Z 轴零偏；(2) **6 状态 EKF** — 融合 IMU、轮速、NHC（非完整性约束）和 GNSS 位置，在 GPS 丢失（隧道、地下车库）时实现鲁棒定位。

## 运行流程

```bash
# 1. 数据预处理（可选——预处理数据已提交）
conda activate traj312 && python data_preprocessing_v2.py

# 2. 训练 BiasNet（核心模型，约 90 秒）
python train_ekf.py

# 3. 验证 GNSS 丢失场景（约 20 秒，生成 ekf_validation.png）
python validate_ekf.py

# 4. Data0109 CMCC 零偏评估（对比 CMCC_result 真值）
python evaluate_bias_cmcc.py

# 5. CMCC 三阶段微调（cmcc_ok 后再等 settle_s，默认 60s，才参与训练）
python train_biasnet_cmcc.py
python train_biasnet_cmcc.py --settle-s 50   # 稳定 50s 后再训练
python train_biasnet_cmcc.py --acc-only      # 仅重跑阶段3
python evaluate_bias_cmcc.py
python evaluate_bias_cmcc.py --stable-only
```

验证输出图片位于 `trained_models/ekf_validation.png` 和 `trained_models/ekf_diagnostics.png`。

Data0109 评估图位于 `trained_models/cmcc_bias_compare_*.png`，日志在 `training_logs/cmcc_eval_*.json`。主用权重：`biasnet_weights_cmcc.weights.h5`；推理时自动叠加 `biasnet_cmcc_acc_calib.json` 的 acc 偏移。

## 代码架构

### 核心模块

- **`ekf_navigator.py`** — 推理代码：`BiasNet`（Keras 模型）、`EKF6D`（6 状态 EKF，含 GNSS/轮速/NHC 更新）、`EKFNavigatorNP`（纯 NumPy 端到端推理循环）。`evaluate_trajectory()` 计算 RMSE、中位数、终点误差和 outage 指标。

- **`config.py`** — `EKFConfig` 数据类，包含所有可调过程/量测噪声参数、零偏限幅、速度门限和初始化协方差。

- **`train_ekf.py`** — 训练流程：加载标定数据（Data01–Data04、Data06），用 GPS 航向变化率减去原始陀螺 Z 计算零偏标签，中值滤波平滑，构建滑窗，用 Huber 损失 + Adam 训练 BiasNet。

- **`validate_ekf.py`** — 评估脚本：加载测试段（默认标定验证集 Data05），模拟 GNSS 丢失（90 秒），对比纯 DR 基准 vs BiasNet+EKF，绘制 2×2 对比图（轨迹叠加、误差曲线、航向、速度诊断）。

- **`data_preprocessing_v2.py`** — 数据加载和预处理：GPS 异常点清洗（速度阈值 150 km/h）、WGS84→ENU 转换、传感器插值到 10 Hz 网格、滑窗生成。导出 `CALIB_TRAIN_IDS`、`CALIB_VAL_ID` 和跨文件使用的路径解析。

- **`data0109_loader.py`** — Data0109 加载；`cmcc_ok`（标定有效）→ `cmcc_stable`（`cmcc_ok` 后再等 `CMCC_SETTLE_S=60s`，跳过静止/收敛初期）。

- **`evaluate_bias_cmcc.py`** — CMCC 对比评估；推理仅在 `cmcc_stable` 段输出，段内默认 2s 平滑（`biasnet_postprocess.py`）。

- **`train_biasnet_cmcc.py`** — 三阶段均在 `cmcc_stable`（默认 ok 后 60s）上训练；阶段3 为 acc 末 60s 常值 + tanh 对齐损失。详细说明见 [`docs/训练零偏文档.md`](训练零偏文档.md)。

-  暂时不用：**`trajectory_data.py`** — 可视化/验证脚本的共享数据加载器：`load_segment()` 用于 260316 CSV 数据，`load_calibration_segment()` 用于标定数据。重新导出 `data_preprocessing_v2.py` 中的常量。

### 关键数据流

```
标定数据 (Data01-06) ──► train_ekf.py ──► BiasNet 权重 (.weights.h5)
                                                   │
Data05 (测试) ──► validate_ekf.py ──► EKFNavigatorNP
           │                                                │
           └──► imu_raw + v_ms + gyro_z + gps_enu ────────►│
                                                             │
                         gps_valid ──► simulate_gnss_outage ──►
                                                             │
               输出: enu_x/y, heading, net_bias, vel_x/y, ekf_bg
```

### 状态向量

6 状态 EKF：`x = [px, py, vx, vy, yaw, bg]^T`
- `px, py`：ENU 位置（米）
- `vx, vy`：ENU 速度（米/秒）
- `yaw`：航向（弧度，东为 0，逆时针为正）
- `bg`：残余陀螺 Z 零偏（弧度/秒）

### 关键参数（config.py）

| 参数 | 默认值 | 用途 |
|-----------|---------|------|
| `q_yaw` | 2e-5 | 航向过程噪声 (rad²/步) |
| `q_vel` | 0.05² | ENU 速度随机游走 |
| `q_bg` | 1e-8 | 残余零偏随机游走 |
| `r_gps_xy` | 2.0² | GNSS 位置噪声 (m²) |
| `r_wheel` | 0.12² | 轮速前向量测噪声 (m/s)² |
| `r_nhc` | 0.08² | 横向速度伪量测噪声 |
| `biasnet_max_deg` | 1.0 | BiasNet 输出 tanh 限幅 |

### 量测更新

- **GNSS 位置**：`gps_valid` 为 True 时直接观测 (px, py)
- **轮速前向**：`v_fwd = vx*cos(yaw) + vy*sin(yaw) ≈ v_wheel` — 约束航向/速度
- **NHC**：`v_lat = -vx*sin(yaw) + vy*cos(yaw) ≈ 0` — 转弯时动态放宽 R

## 项目结构

```
GPS/
├── ekf_navigator.py        # BiasNet + EKF6D + EKFNavigatorNP（推理）
├── train_ekf.py            # BiasNet 训练脚本
├── validate_ekf.py         # EKF 验证与绘图
├── config.py               # EKFConfig 数据类（所有可调参数）
├── data_preprocessing_v2.py# 数据加载、GPS 清洗、ENU 转换
├── data0109_loader.py      # Data0109 + CMCC 零偏加载
├── evaluate_bias_cmcc.py   # CMCC 零偏对比评估
├── train_biasnet_cmcc.py   # CMCC 监督微调
├── trajectory_data.py      # 验证用共享段加载器
├── Data0109数据及零偏/       # 0109 实车 + CMCC_result 真值
├── trained_models/
│   ├── biasnet_weights.weights.h5  # 训练好的 BiasNet 权重
│   ├── biasnet_weights_cmcc.weights.h5        # CMCC 三阶段最终权重
│   ├── biasnet_weights_cmcc_s1.weights.h5     # 阶段1
│   ├── biasnet_cmcc_acc_calib.json            # 阶段3 估计的 acc 推理偏移 [g]
│   ├── biasnet_info.json           # 训练元数据
│   ├── biasnet_info_cmcc_stable.json
│   ├── cmcc_bias_compare_*_stable.png
│   ├── ekf_validation.png          # 主验证图
│   └── ekf_diagnostics.png         # 诊断图
└── preprocessed_data/
    ├── normalization_stats.json    # IMU 通道均值/标准差
    ├── X_train.npy / Y_train.npy   # 训练窗口/标签
    └── X_test.npy / Y_test.npy     # 测试窗口/标签
```
