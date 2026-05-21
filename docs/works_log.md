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
