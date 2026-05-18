"""
EKF 航向估计器 + BiasNet 陀螺零偏预测网络
参考 ai-imu-dr (Martin et al. 2021) 简化为地面车辆二维场景

设计思路
--------
一、根本问题：陀螺 Z 轴存在随时间漂移的零偏 b_ω，导致航向累积误差。

二、解决方案（分层）：
  1. BiasNet (神经网络)：
     - 输入  : 最近 W 帧 IMU 数据窗口 (W × 6)
     - 输出  : 当前时刻陀螺 Z 轴零偏估计 b̂_ω (rad/s)
     - 训练  : 监督回归，标签 = 陀螺 Z 测量 − GPS 推算航向率 (在 GPS 有效段)

  2. EKF (卡尔曼滤波器)：
     - 状态  : x = [θ, δb_ω] (航向 + 残余零偏 = 网络预测误差)
     - 传播  : θ ← θ + (ω_z − b̂_net − δb_ω) × dt
     - 量测  : GPS 有效时 z = θ_GPS，校正 θ 与 δb_ω
     - 目标  : 精细校正网络的预测残差，并在 GPS 丢失后维持状态

  3. 位置累积（NHC）：
     - Δx = v × cos(θ) × dt  (非完整约束：仅前向位移，零侧向位移)
     - Δy = v × sin(θ) × dt
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from pathlib import Path

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
EPS     = 1e-8


# ============================================================================
# BiasNet —— 陀螺 Z 零偏预测
# ============================================================================

class BiasNet(Model):
    """
    输入：归一化后的 IMU 窗口  (batch, window, 6) — [AccX,AccY,AccZ,GyroX,GyroY,GyroZ]
    输出：(batch, 1) — 陀螺 Z 零偏 b̂_ω (rad/s)

    架构：两层膨胀因果卷积 + 全连接，参考 ai-imu-dr MesNet。
    膨胀 = 1, 4 → 感受野 ≈ 12 帧（1.2 s @ 10 Hz），轻量高效。
    """

    def __init__(self, window_size: int = 30):
        super().__init__(name='BiasNet')
        self.conv1 = layers.Conv1D(32, 5, padding='causal', activation='relu')
        self.conv2 = layers.Conv1D(32, 5, dilation_rate=4,
                                   padding='causal', activation='relu')
        self.conv3 = layers.Conv1D(16, 3, padding='causal', activation='relu')
        self.pool  = layers.GlobalAveragePooling1D()
        self.fc1   = layers.Dense(32, activation='relu')
        self.drop  = layers.Dropout(0.2)
        self.out   = layers.Dense(1, activation='linear')   # rad/s

    def call(self, x, training=False):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.pool(h)
        h = self.fc1(h)
        h = self.drop(h, training=training)
        return self.out(h)                   # (batch, 1)


# ============================================================================
# Numpy EKF（推理阶段）
# ============================================================================

class EKF2D:
    """
    2 状态卡尔曼滤波器：状态 x = [θ, δb_ω]
      θ    : 当前航向 (rad, ENU 坐标，东轴为 0，逆时针正)
      δb_ω : 残余零偏（网络预测误差），单位 rad/s

    BiasNet 已估计大部分零偏，EKF 只估计残余部分，
    因此初始状态和过程噪声量级都很小。
    """

    def __init__(self,
                 theta_init: float = 0.0,
                 q_theta: float   = 0.0,       # 航向过程噪声（确定性传播，置 0）
                 q_db: float      = 1e-7,       # 残余零偏随机游走
                 r_theta: float   = 1e-3,       # GPS 航向量测噪声 (rad²)
                 P_init=None):
        self.theta = float(theta_init)
        self.db    = 0.0                        # 残余零偏初始值 = 0
        self.P     = P_init if P_init is not None else np.diag([0.5, 1e-4])
        self.q_theta = q_theta
        self.q_db    = q_db
        self.r_theta = r_theta

    # ---- 时间更新 ----
    def propagate(self, gyro_z_rad: float, bias_net: float, dt: float):
        """
        gyro_z_rad : 当前帧陀螺 Z 角速度 (rad/s)
        bias_net   : BiasNet 预测的零偏 (rad/s)
        dt         : 时间步长 (s)
        """
        # 状态传播
        self.theta += (gyro_z_rad - bias_net - self.db) * dt
        # 零偏为随机游走，均值不变

        # 协方差传播  F = [[1, -dt], [0, 1]]
        p00 = (self.P[0, 0]
               - dt * self.P[1, 0]
               - dt * (self.P[0, 1] - dt * self.P[1, 1])
               + self.q_theta)
        p01 = self.P[0, 1] - dt * self.P[1, 1]
        p10 = self.P[1, 0] - dt * self.P[1, 1]
        p11 = self.P[1, 1] + self.q_db
        self.P = np.array([[p00, p01], [p10, p11]])

    # ---- 量测更新（GPS 航向）----
    def update_gps_heading(self, gps_theta: float, r_theta: float = None):
        """
        gps_theta : GPS 推算的 ENU 航向 (rad)
        """
        r = r_theta if r_theta is not None else self.r_theta
        # H = [1, 0]，量测 = θ
        S  = self.P[0, 0] + r + EPS
        K  = self.P[:, 0] / S             # (2,) 卡尔曼增益
        inn = gps_theta - self.theta       # 创新量
        # 限制创新量（防止 GPS 角度跳变带来的大修正）
        inn = np.clip(inn, -np.pi / 2, np.pi / 2)
        self.theta += K[0] * inn
        self.db    += K[1] * inn
        # Joseph 形式协方差更新
        I_KH    = np.eye(2) - np.outer(K, [1.0, 0.0])
        self.P  = I_KH @ self.P

    @property
    def heading(self):
        return self.theta

    @property
    def bias(self):
        return self.db


# ============================================================================
# 完整推理导航器
# ============================================================================

class EKFNavigatorNP:
    """
    推理阶段导航器（纯 Numpy，不依赖 TF 计算图）：
      1. BiasNet 预测每步陀螺 Z 零偏
      2. EKF 传播航向并在 GPS 有效时更新
      3. NHC 位置累积

    用法示例
    --------
    nav = EKFNavigatorNP('trained_models/biasnet_weights', norm_stats)
    enu_x, enu_y, headings, biases = nav.run(
        imu_raw, v_ms, gyro_z_rad, gps_theta, gps_valid, dt)
    """

    def __init__(self,
                 weights_path: str,
                 norm_stats: dict,
                 window_size: int = 30,
                 q_db: float = 1e-7,
                 r_theta: float = 1e-3):
        self.window_size = window_size
        self.norm_stats  = norm_stats
        self.q_db    = q_db
        self.r_theta = r_theta

        # 加载 BiasNet
        self.biasnet = BiasNet(window_size)
        dummy = np.zeros((1, window_size, 6), dtype=np.float32)
        self.biasnet(dummy)                         # 初始化权重
        wp = str(weights_path)
        if not wp.endswith('.weights.h5'):
            wp += '.weights.h5'
        self.biasnet.load_weights(wp)
        print(f"[EKFNavigator] BiasNet 加载成功：{weights_path}")

    def _normalize_imu(self, imu_raw: np.ndarray) -> np.ndarray:
        """imu_raw: (T, 6) → 归一化后 (T, 6)"""
        keys = ['AccX_g', 'AccY_g', 'AccZ_g',
                'GyroX_degs', 'GyroY_degs', 'GyroZ_degs']
        out = np.empty_like(imu_raw, dtype=np.float32)
        for i, k in enumerate(keys):
            mu  = self.norm_stats[k]['mean']
            std = self.norm_stats[k]['std']
            out[:, i] = (imu_raw[:, i] - mu) / (std + EPS)
        return out

    def _predict_bias(self, imu_norm: np.ndarray) -> np.ndarray:
        """
        imu_norm : (T, 6) 归一化 IMU
        返回     : (T,) 零偏估计 (rad/s)，前 window_size 帧用 0
        """
        T = len(imu_norm)
        W = self.window_size
        bias = np.zeros(T, dtype=np.float32)
        if T < W:
            return bias

        # 批量构造窗口并推理
        windows = np.stack(
            [imu_norm[i: i + W] for i in range(T - W + 1)], axis=0)
        pred = self.biasnet(windows.astype(np.float32),
                            training=False).numpy().flatten()  # (T-W+1,)
        bias[W - 1:] = pred
        return bias

    def run(self,
            imu_raw: np.ndarray,
            v_ms: np.ndarray,
            gyro_z_rad: np.ndarray,
            gps_theta: np.ndarray,
            gps_valid: np.ndarray,
            dt: float,
            theta_init: float = None) -> tuple:
        """
        参数
        ----
        imu_raw    : (T, 6) 原始 IMU — [AccX_g, AccY_g, AccZ_g, GyroX_degs, GyroY_degs, GyroZ_degs]
        v_ms       : (T,) 车速 m/s
        gyro_z_rad : (T,) 陀螺 Z 角速度 rad/s
        gps_theta  : (T,) GPS 推算 ENU 航向 rad（无效帧任意值）
        gps_valid  : (T,) bool
        dt         : 时间步长 s

        返回
        ----
        enu_x, enu_y : (T,) 累积位置 (m)，相对于起点
        headings     : (T,) 估计航向 rad
        net_biases   : (T,) 网络预测零偏 rad/s
        """
        T = len(v_ms)

        # 步骤 1：归一化并预测零偏
        imu_norm  = self._normalize_imu(imu_raw)
        net_bias  = self._predict_bias(imu_norm)

        # 步骤 2：确定初始航向
        if theta_init is None:
            valid_idx = np.where(gps_valid)[0]
            theta_init = float(gps_theta[valid_idx[0]]) if len(valid_idx) > 0 else 0.0

        ekf = EKF2D(theta_init=theta_init,
                    q_db=self.q_db,
                    r_theta=self.r_theta)

        # 步骤 3：逐帧运行 EKF
        headings = np.empty(T, dtype=np.float32)
        dx_arr   = np.empty(T, dtype=np.float32)
        dy_arr   = np.empty(T, dtype=np.float32)

        for k in range(T):
            ekf.propagate(gyro_z_rad[k], float(net_bias[k]), dt)

            if gps_valid[k]:
                ekf.update_gps_heading(float(gps_theta[k]))

            theta_k      = ekf.heading
            headings[k]  = theta_k
            dx_arr[k]    = float(v_ms[k]) * np.cos(theta_k) * dt
            dy_arr[k]    = float(v_ms[k]) * np.sin(theta_k) * dt

        # 步骤 4：位置累积
        enu_x = np.cumsum(dx_arr)
        enu_y = np.cumsum(dy_arr)

        return enu_x, enu_y, headings, net_bias


# ============================================================================
# 工具：归一化统计量加载
# ============================================================================

def load_norm_stats(path: str) -> dict:
    """
    从 normalization_stats.json 加载归一化参数，
    返回 dict: {'AccX_g': {'mean': ..., 'std': ...}, ...}
    """
    import json
    with open(path) as f:
        data = json.load(f)
    # 兼容两种格式
    if 'stats' in data:
        return data['stats']
    return data
