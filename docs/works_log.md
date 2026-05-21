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
