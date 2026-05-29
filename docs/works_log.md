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

## 2026-05-28

### CMCC 训练配置（面向零偏精度）调整

- `train_biasnet_cmcc.py` 默认 Deep 窗口从 `200` 调整为 `300`（增强长时序信息）
- 阶段1/2损失从六轴等权改为加权：acc `[0.8, 0.8, 0.6]`，gyro `[2.0, 2.0, 1.5]`（提高陀螺X/Y优化权重）
- 阶段3损失权重调整：acc `[8.0, 8.0, 6.0]`，gyro `[2.0, 2.5, 2.5]`
- 阶段1/2 deep 学习率调整：`5e-4 -> 3e-4`、`1e-4 -> 5e-5`
- 阶段3 `acc_sample_dup` 从 `2` 调整为 `1`，减弱 acc 任务对梯度主导
- 新增并固化训练超参常量：`STAGE1_LR_DEEP`、`STAGE2_LR_DEEP`、`STAGE3_ACC_SAMPLE_DUP` 等

### 汇报文档输出

- 新增脚本 `scripts/generate_optimization_report_docx.py`，用于自动生成领导汇报版优化文档
- 生成报告：`reports/零偏模型优化阶段汇报_20260527.docx`
- 报告内容覆盖：已完成优化点、相较之前变动、当前问题、后续两周计划、阶段结论

### 加入时间一致性损失（2026-05-28）

- `train_biasnet_cmcc.py` 新增 `cmcc_huber_with_temporal_consistency()`：在加权 Huber 基础上增加相邻样本预测差分约束
- 时间一致性仅作用陀螺三轴（`bg_x/bg_y/bg_z`），默认权重 `[1.0, 1.0, 1.2]`
- 正则强度默认 `TEMP_CONSISTENCY_LAMBDA = 0.05`
- 阶段2训练切换为该损失，并设置 `shuffle=False` 保留样本时序关系
- `biasnet_info_cmcc.json` 的 `stage2` 元数据新增 `temporal_consistency` 配置记录

### 上次时间一致性修改效果复盘（2026-05-28）

- 对比日志：`cmcc_eval_20260527_193507.json`（基线） vs `cmcc_eval_20260528_150140.json`（首次时间一致性）
- Data05 `cmcc_stable` 关键变化：
  - `bg_z_mae`: `0.0377 -> 0.0831`（明显变差）
  - `bg_y_mae`: `0.0421 -> 0.0457`（小幅变差）
  - `bg_x_mae`: `0.0439 -> 0.0396`（小幅改善）
- 结论：首次时间一致性约束过强，导致陀螺通道（尤其 `bg_z`）出现过平滑与系统偏差，不可直接采用

### 时间一致性二次修正（2026-05-28）

- 正则强度降低：`TEMP_CONSISTENCY_LAMBDA` 从 `0.05` 调整为 `0.008`
- 一致性约束范围收缩：仅约束 `bg_x/bg_y`，不再直接约束 `bg_z`
- 新增标签差分门控：仅当相邻标签差分 `<= 0.03 deg/s` 时计入一致性损失，尽量规避跨段/突变样本的错误平滑
- `biasnet_info_cmcc.json` 的 `stage2.temporal_consistency` 同步记录新参数（`gyro_xy_weights`、`max_label_delta`）

### 二次修正后结果评估（2026-05-28）

- 本轮日志：`training_logs/cmcc_eval_20260528_153136.json`
- 对比对象：
  - 基线（未加时间一致性）：`cmcc_eval_20260527_193507.json`
  - 首次时间一致性（失败版）：`cmcc_eval_20260528_150140.json`

- Data05 `cmcc_stable`（本轮 vs 失败版）：
  - `ba_x_mae`: `0.0173 -> 0.0087`（明显改善）
  - `ba_y_mae`: `0.0272 -> 0.0203`（改善）
  - `ba_z_mae`: `0.0518 -> 0.0449`（改善）
  - `bg_x_mae`: `0.0396 -> 0.0252`（明显改善）
  - `bg_y_mae`: `0.0457 -> 0.0455`（基本持平，微改善）
  - `bg_z_mae`: `0.0831 -> 0.0314`（大幅改善，核心问题修复）

- Data05 `cmcc_stable`（本轮 vs 基线）：
  - `ba_x_mae`: `0.0192 -> 0.0087`（明显改善）
  - `ba_y_mae`: `0.0268 -> 0.0203`（改善）
  - `ba_z_mae`: `0.0455 -> 0.0449`（小幅改善）
  - `bg_x_mae`: `0.0439 -> 0.0252`（明显改善）
  - `bg_y_mae`: `0.0421 -> 0.0455`（小幅变差，当前主要短板）
  - `bg_z_mae`: `0.0377 -> 0.0314`（改善）

- 图像结论（`trained_models/cmcc_bias_compare_20260528_153136_Data05_0109_4q跑道.png`）：
  - `bg_z` 从失败版的系统偏差恢复，预测曲线与 CMCC 真值贴合明显改善
  - `bg_x` 拟合稳定性提升
  - `bg_y` 仍有系统性偏差，下一轮优先做 `bg_y` 定向权重微调

### bg_y 定向加权后的结果复盘（2026-05-28）

- 本轮日志：`training_logs/cmcc_eval_20260528_163047.json`
- 对比对象：`training_logs/cmcc_eval_20260528_153136.json`
- Data05 `cmcc_stable` 结果变化：
  - `bg_y_mae`: `0.0455 -> 0.0284`（明显改善，达到 `<=0.043` 目标）
  - `bg_z_mae`: `0.0314 -> 0.0724`（明显变差）
  - `bg_x_mae`: `0.0252 -> 0.0272`（小幅变差）
  - `ba_x_mae`: `0.0087 -> 0.0113`（小幅变差）
  - `ba_y_mae`: `0.0203 -> 0.0224`（小幅变差）
  - `ba_z_mae`: `0.0449 -> 0.0413`（小幅改善）
- 结论：单独提高 `bg_y` 权重可以显著压低 `bg_y` 误差，但会破坏通道平衡，导致 `bg_z` 明显退化，不作为最终方案

### 陀螺权重折中调整（2026-05-28）

- 修改文件：`train_biasnet_cmcc.py`
- 修改项：主损失陀螺权重 `LOSS_GYRO_WEIGHTS` 从 `[2.0, 2.4, 1.5]` 调整为 `[2.0, 2.25, 1.9]`
- 调整意图：保留部分 `bg_y` 改善，同时补偿 `bg_z` 权重，减少整体指标失衡
- 待验证重点：
  - `bg_y_mae` 仍尽量保持在 `<=0.043`
  - `bg_z_mae` 回落到接近 `0.03~0.04` 区间

### 陀螺权重折中验证结果（2026-05-28，当前推荐）

- 本轮日志：`training_logs/cmcc_eval_20260528_165527.json`
- 配置：`LOSS_GYRO_WEIGHTS = [2.0, 2.25, 1.9]`
- Data05 `cmcc_stable`：`ba_x` 0.0125，`ba_y` 0.0222，`ba_z` 0.0356；`bg_x` 0.0260，`bg_y` 0.0351，`bg_z` 0.0242
- 结论：**165527 + 折中陀螺权重** 为现阶段推荐 checkpoint

### 汇报文档（2026-05-28）

- 脚本：`scripts/generate_optimization_report_docx_20260528.py`
- 报告：`reports/零偏模型优化阶段汇报_20260528.docx`

### bg_y 定向权重微调（2026-05-28）

- 目标：保持当前方案不变，仅尝试压低 Data05 的 `bg_y_mae`
- 修改文件：`train_biasnet_cmcc.py`
- 修改项：主损失陀螺权重 `LOSS_GYRO_WEIGHTS` 从 `[2.0, 2.0, 1.5]` 调整为 `[2.0, 2.4, 1.5]`
- 说明：仅提高 `bg_y` 通道损失权重，`bg_x/bg_z` 与其它训练配置保持不变
- 待验证指标：下一轮重点观察 `cmcc_eval_*.json` 中 Data05 `cmcc_stable.per_channel.bg_y_degs.mae` 是否回落到 `<= 0.043`

### 二次修正后结果评估（2026-05-28）

- 本轮日志：`training_logs/cmcc_eval_20260528_153136.json`
- 对比对象：
  - 基线（未加时间一致性）：`cmcc_eval_20260527_193507.json`
  - 首次时间一致性（失败版）：`cmcc_eval_20260528_150140.json`

- Data05 `cmcc_stable`（本轮 vs 失败版）：
  - `ba_x_mae`: `0.0173 -> 0.0087`（明显改善）
  - `ba_y_mae`: `0.0272 -> 0.0203`（改善）
  - `ba_z_mae`: `0.0518 -> 0.0449`（改善）
  - `bg_x_mae`: `0.0396 -> 0.0252`（明显改善）
  - `bg_y_mae`: `0.0457 -> 0.0455`（基本持平，微改善）
  - `bg_z_mae`: `0.0831 -> 0.0314`（大幅改善，核心问题修复）

- Data05 `cmcc_stable`（本轮 vs 基线）：
  - `ba_x_mae`: `0.0192 -> 0.0087`（明显改善）
  - `ba_y_mae`: `0.0268 -> 0.0203`（改善）
  - `ba_z_mae`: `0.0455 -> 0.0449`（小幅改善）
  - `bg_x_mae`: `0.0439 -> 0.0252`（明显改善）
  - `bg_y_mae`: `0.0421 -> 0.0455`（小幅变差，当前主要短板）
  - `bg_z_mae`: `0.0377 -> 0.0314`（改善）

- 图像结论（`trained_models/cmcc_bias_compare_20260528_153136_Data05_0109_4q跑道.png`）：
  - `bg_z` 从失败版的系统偏差恢复，预测曲线与 CMCC 真值贴合明显改善
  - `bg_x` 拟合稳定性提升
  - `bg_y` 仍有系统性偏差，下一轮优先做 `bg_y` 定向权重微调
