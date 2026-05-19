# 模型流程说明

从原始数据到最终轨迹，整个过程分三大步，每一步都有明确的输入和输出。

---

## 整体流程（一句话）

> 用 IMU + 轮速 + GPS 数据训练一个神经网络（BiasNet），再用它辅助卡尔曼滤波器（EKF）在 GPS 丢失时也能推算位置。

---

## 第一步：数据预处理 `data_preprocessing_v2.py`

### 做什么

把车载传感器记录的原始文件，整理成统一格式、统一时间间隔的数据。

### 输入

```
data/标定数据/         ← 标定采集的原始数据文件夹
   Data01/
     IMU.txt           ← 加速度计 + 陀螺仪（原始值）
     速度.txt           ← 轮速脉冲转换的车速
     GPS.txt            ← GPS 经纬度 + 航向
   Data02/
     ...                ← 同上格式
   ...

260316_Data/
   260316_Data.csv      ← 另一批测试数据（和标定数据格式不同）
```

### 做了什么处理

1. **时间对齐**：IMU、轮速、GPS 三个传感器的采样时刻不一样，统一插值到 **10Hz（每 0.1 秒一帧）**
2. **GPS 去跳点**：速度超过 150 km/h 的异常 GPS 点剔除
3. **坐标转换**：GPS 经纬度（度）→ ENU 坐标（米），以第一个有效 GPS 点为原点
4. **生成滑窗**：把连续的 IMU 数据切成 30 帧（3 秒）的滑动窗口，用于训练

### 输出

```
preprocessed_data/
   normalization_stats.json   ← 每个传感器通道的均值/标准差（给 BiasNet 归一化用）
   X_train.npy                ← 训练 IMU 窗口 (N, 30, 6)
   Y_train.npy                ← 训练零偏标签 (N, 1)
   X_test.npy                 ← 测试 IMU 窗口
   Y_test.npy                 ← 测试零偏标签
```

> **从这一步你能带走什么？** 归一化参数（训练要用），以及切好的数据窗口。

---

## 第二步：训练 BiasNet `train_ekf.py`

### 做什么

训练一个小型神经网络，让它学会"看到一段 IMU 数据，猜出陀螺仪当前有多偏"。

### 输入

```
preprocessed_data/
   normalization_stats.json   ← 归一化参数（每个通道的均值/标准差）

data/标定数据/
   Data01 ~ Data04            ← 训练集（4 段不同场景的标定数据）
   Data05                     ← 验证集（训练过程中看效果，不算成绩）
   Data06                     ← 额外的测试数据

260316_Data/
   260316_Data.csv            ← 另一段真实数据的前 70% 也加入训练
```

### BiasNet（神经网络）长什么样

```
输入: 30帧 × 6通道 的 IMU 窗口
  │
  ├─ Conv1D(32, 5)     ← 卷积层，提取局部特征
  ├─ Conv1D(32, 5)     ← 空洞卷积，看更广的时间范围
  ├─ Conv1D(16, 3)     ← 再压缩
  ├─ GlobalAvgPool     ← 把 30 帧压缩成一个值
  ├─ Dense(32)         ← 全连接层
  └─ Dense(1)          ← 输出零偏值
```

只有 **~8,000 个参数**，非常轻量。

### 零偏标签怎么来的

陀螺仪测量值 = 真实角速度 + 零偏。零偏就是"车子不动时陀螺也在转的那个值"。

计算方式：

```
① 用 GPS 位置变化算出真实航向变化率
② 测量值 - 真实值 = 零偏
```

具体是积分法（比求导法噪声小）：

```
对每 3 秒窗口：
  GPS 航向变化 = θ_gps[窗口尾] - θ_gps[窗口头]
  陀螺航向变化 = Σ(gyro_z) × 0.1 秒
  零偏 = (陀螺变化 - GPS 变化) / 3 秒
```

### 输出

```
trained_models/
   biasnet_weights.weights.h5  ← 训练好的神经网络权重（核心产物）
   biasnet_info.json           ← 训练记录：用了多少样本、最终误差多少
```

> 一个 `.weights.h5` 文件，这就是 BiasNet 的"大脑"。

---

## 第三步：EKF 验证 `validate_ekf.py`

### 做什么

模拟 GPS 信号丢失（比如进隧道），看只用 IMU + 轮速 + BiasNet 能撑多久不偏。

### 输入

```
trained_models/
   biasnet_weights.weights.h5  ← 训练好的 BiasNet

preprocessed_data/
   normalization_stats.json    ← 归一化参数

标定验证集 Data05 的完整数据：
   IMU 6 通道   ← 加速度 + 陀螺
   轮速 v_ms    ← 车速（米/秒）
   GPS 位置     ← 当作"地面真值"（用来对比误差）
   GPS 有效标志 ← 用来模拟 GPS 丢失
```

### EKF 是什么（6 状态卡尔曼滤波器）

EKF 内部维护 6 个状态：

```
状态 = [px, py,     ← 位置（东、北）
        vx, vy,     ← 速度（东、北）
        yaw,        ← 航向（车头朝向，东=0°）
        bg]         ← 残余陀螺零偏
```

每帧做两件事：

```
1. 预测：用陀螺 + BiasNet 推算航向变化 → 更新位置/速度
2. 更新：有 GPS 就校正位置，有轮速就校正速度，NHC 保证横向速度≈0
```

### EKF 预测公式（核心）

```
ω = gyro_z - bias_net - bg     ← 减去零偏得到真实角速度
yaw += ω × dt                  ← 航向积分

px += vx × dt                  ← 位置积分
py += vy × dt

vx = vx×cos(Δyaw) - vy×sin(Δyaw)   ← 速度随车头旋转
vy = vx×sin(Δyaw) + vy×cos(Δyaw)
```

### 模拟 GPS 丢失

```
时间轴：
  0─15 秒：  有 GPS  → EKF 正常校正，初始化
  15─105 秒：丢 GPS  → EKF 全靠 IMU + 轮速推算（模拟隧道）
  105 秒后：  有 GPS  → 看 EKF 能不能快速找回正确位置
```

### 输出

```
trained_models/
   ekf_validation.png    ← 四个子图的验证结果图
   ekf_diagnostics.png   ← 速度诊断图
```

### 验证结果怎么看

| 图 | 内容 | 关键看点 |
|----|------|---------|
| 左上 | 轨迹对比 | 橙色段（GPS 丢失时）离绿线（真值）有多远 |
| 右上 | 位置误差 | 蓝线（EKF）越低越好，红虚线（纯 DR）通常很高 |
| 左下 | 航向对比 | 蓝线是否跟绿线走，丢 GPS 时有没有漂 |
| 右下 | 速度诊断 | 蓝线（EKF 速度）是否贴近绿线（轮速），红线（横向速度）是否接近 0 |

### 控制台输出指标

```
RMSE        = 全程均方根误差（越小越好）
Outage 最大 = GPS 丢失期间的最大偏离（核心指标）
终点误差    = 最后时刻离真值多远（看能不能找回）
```

---

## 一句话总结

| 步骤 | 脚本 | 输入 | 产出 |
|------|------|------|------|
| ① 数据整理 | `data_preprocessing_v2.py` | 原始传感器 CSV | 对齐后的 10Hz 数据 + 归一化参数 |
| ② 训练网络 | `train_ekf.py` | 对齐后数据 + 归一化参数 | BiasNet 权重 (.weights.h5) |
| ③ 验证效果 | `validate_ekf.py` | 权重 + 归一化参数 + 测试数据 | 轨迹图 + 误差指标 |

---

## 常用命令

```bash
# 全流程跑一遍（按顺序）
conda activate traj312
python data_preprocessing_v2.py    # 数据预处理
python train_ekf.py                # 训练（约 90 秒）
python validate_ekf.py             # 验证（约 20 秒）
```
