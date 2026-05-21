# 工作日志

## 2026-05-21

### 环境配置

- 创建conda环境 `traj312` (Python 3.12)
- 安装依赖：numpy, pandas, scipy, matplotlib, scikit-learn, tensorflow, torch, plotly
- 使用清华镜像源加速下载
- tensorflow因Windows虚拟内存不足(页面文件太小)暂无法加载，其他包正常

### 项目理解

整体架构：BiasNet(轻量CNN预测陀螺零偏) + 6状态EKF(融合IMU/轮速/NHC/GNSS)

当前问题：

- tensorflow无法在traj312环境中加载（DLL错误，页面文件不足）

### 调参

- `config.py` 调整两个参数以改善outage期间定位精度：
  - `q_vel`: 0.05² → 0.03²（减少outage期间速度过程噪声，降低漂移）
  - `r_nhc`: 0.08² → 0.06²（收紧横向非完整约束，直道更稳定）

### 训练日志功能

- `train_ekf.py` 新增 `save_training_log()`：每次训练完在 `training_logs/` 生成 `train_YYYYMMDD_HHMMSS.json`，记录 config 参数、训练样本数、best val MAE、预测残差统计
- `validate_ekf.py` 新增 `save_validation_log()`：每次验证完在 `training_logs/` 生成 `validate_YYYYMMDD_HHMMSS.json`，记录 config 参数、outage 设置、DR/EKF 全部指标(RMSE/outage最大/航向等)

### 航向精度改进

- `ekf_navigator.py` `EKFNavigatorNP.run()` 新增 `enable_gnss_heading` 参数（默认True）
- 当 GPS 有效时，调用 `EKF6D.update_gnss_heading()` 用 GPS 航向直接校正 EKF 航向
- 之前 EKF 只在初始化时用 GPS 航向，运行时完全靠轮速/NHC间接校正航向，导致航向误差积累
- 修复 `validate_ekf.py` 日志函数中的变量名 bug（tg → seq['Time_s']）
- 将航向量测移到 NHC/轮速更新之后，确保航向修正是最后一个量测更新，不被覆盖
- 降低 `r_gps_heading_rad2`: 3° → 1.5° std，提高航向量测权重
- `_init_state` 初始化航向改进：优先用 gps_theta（即使静止），降低位移条件 2m→1m
- **根源问题**：Data05 前62秒车辆静止，k_start=627（首次运动帧），远晚于outage开始(帧150)，导致航向量测在整个GPS有效期从未触发。初始化阶段改用gps_theta后航向恢复正常

### 效果（vs 初始配置）

- EKF RMSE: 5.21m → 3.64m（↓30%）
- Outage最大误差: 17.90m → 13.15m（↓26%）
- 航向跟踪(GPS有效时段运动段): mean≈0.0°, std≈2.1°
- 90秒outage仅漂移13m，整体定位精度良好

### 知陷阱

- `abs(gps_theta) > 1e-6` 不能用于 ENU 航向过滤（东方向=0），会导致车往东走时航向量测被跳过
- `.pyc` 缓存可能导致代码修改不生效，需删除 `__pycache__` 强制重编译

### 零偏标签净化

- `train_ekf.py` `compute_bias_labels` 重写：新增 `is_moving`（v_ms>0.5m/s）和 `is_straight`（|ω|<2°/s）硬约束
- 三合一严格掩码：`strict_ok = gps_valid & is_moving & is_straight`
- 只有窗口两端同时满足时才计算零偏标签，无效段线性插值填补
- 效果：BiasNet MAE 从 0.97→0.11°/s，残差p95 从 3.02→0.33°/s

### Outage 时序修复

- `validate_ekf.py` `simulate_gps_loss` 改为自动计算 outage 开始时间：首次运动 + 15s
- 修复前：outage 在 15s 开始（车还没动）
- 修复后：outage 在 75.8s 开始（车动后 15s 初始化再丢 GPS）
- 航向评估排除静止期（v_ms<0.5 时 truth_yaw=NaN）
- 航向子图静止段隐藏

### CovAdapterNet 端到端训练

- `train_e2e.py` 改造：
  - 去掉 QUICK_TEST 模式，改为全量数据训练
  - Epochs 5→20，LR 1e-4→3e-4
  - 训练时使用 BiasNet 真实零偏预测（之前全0，与实际推理条件不匹配）
  - 新增训练日志保存
- 当前 CovAdapterNet 权重已失效（用旧 BiasNet 训练），需重训


### 放宽 is_straight 阈值（包含转弯训练数据）

- `train_ekf.py` `compute_bias_labels`：`is_straight` 从 2°/s → 8°/s
- 原因：原2°排除了几乎所有转弯数据，BiasNet 在八字路段预测失准
- 效果：八字路段outage漂移仍有28m，转弯时零偏标签本身脏，阈值调不了

### 图表时间戳 & 防覆盖

- `validate_ekf.py`：图片文件名加时间戳 `ekf_validation_YYYYMMDD_HHMMSS_tag.png`
- 图标题显示生成时间，方便对比每次差异
- 诊断图同理

### BiasNet 增加轮速输入通道

- `train_ekf.py` `load_calibration_seq`：`imu_mat` 从 (T,6) → (T,7)，v_ms 作为第7通道
- `train_ekf.py` `load_or_compute_norm`：新增 `'VehicleSpeed_ms'` 到归一化统计
- `ekf_navigator.py` `_normalize_imu`：新增 `'VehicleSpeed_ms'` 归一化
- `ekf_navigator.py` `EKFNavigatorNP.__init__`：BiasNet 构建用7通道dummy，兼容旧6通道权重
- `ekf_navigator.py` `EKFNavigatorNP.run()`：推理时将 `imu_raw`(T,6) 与 `v_ms`(T,) 拼成 (T,7) 再送入 BiasNet；CovAdapterNet 仍用6通道 IMU 输入
- 训练后 BiasNet 能通过车速区分"转弯"vs"零偏漂移"

### Outage 覆盖八字路段

- `validate_ekf.py` `simulate_gps_loss`：outage 改为 100s-200s（覆盖八字路段）
- 八字路段航向漂移22°，位置漂移28m（比直道难很多）
- 原因是连续转弯时 NHC 被放松，航向主要靠陀螺积分

### BiasNet 隧道增强训练

- `train_ekf.py` `build_samples` 新增隧道增强逻辑：
  - 对每段训练数据，在运动段中间模拟 8 秒 GPS 丢失
  - 丢失期间 `gps_valid` 设为 False，`compute_bias_labels` 自动插值填补零偏标签
  - 模拟丢失段的 IMU 窗口加入训练集，让 BiasNet 学会 GPS 丢失场景下的零偏预测
  - 验证集不增强，保证评估公正

### BiasNet 端到端微调（两阶段训练）
- `ekf_torch.py` 新增 `BiasNetTorch`（PyTorch版，7通道输入，与Keras架构一致）
- `ekf_torch.py` 新增 `export_biasnet_torch_to_keras()` 权重导出函数
- 新建 `train_biasnet_e2e.py`：端到端微调脚本
  - 训练模式：`60s 有GPS → 丢10s GPS → GPS恢复 → 算位置误差 → 反向传播`
  - 加载已有 Keras 权重 → 转 PyTorch → 可微 EKF 微调 → 导回 Keras
  - 输出 `biasnet_finetuned.weights.h5`（微调版）并覆盖原权重路径
  - 小学习率 1e-5，10 个 epoch

### 验证图第4子图改为零偏预测图
- `validate_ekf.py` `plot_results` 子图 (d) 从"速度诊断"改为"陀螺零偏"
- 三条线：BiasNet 预测蓝、EKF 残差红、合计绿
- 右侧显示均值/标准差统计
