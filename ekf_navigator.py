"""
EKF 二维地面车辆 GNSS/INS/Wheel 融合导航 + BiasNet 6 轴零偏预测

状态 x = [px, py, vx, vy, yaw, bgx, bgy, bgz, bax, bay, baz, vel_scale]^T  (12维)
  px, py  : ENU 位置 (m)
  vx, vy  : ENU 速度 (m/s)
  yaw     : 航向 (rad, 东为 0, 逆时针为正)
  bgx/y/z : 陀螺残余零偏 (rad/s)，BiasNet 已扣除主零偏
  bax/y/z : 加速度计残余零偏 (g)，BiasNet 已扣除主零偏
  vel_scale : 轮速比例因子 (无量纲)

量测更新:
  - GNSS 位置 (px, py)
  - 轮速前向 v_fwd = vx*cos(yaw)+vy*sin(yaw) ≈ v_wheel * vel_scale
  - NHC 横向 v_lat ≈ 0（转弯时动态放宽 R）
  - 加速度计前向伪量测（约束 ba_x）
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from pathlib import Path
from typing import Optional, Tuple

from config import EKFConfig, DEFAULT_EKF_CONFIG

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
GRAVITY = 9.80665          # m/s² per g
EPS = 1e-8
N_STATE = 12
IDX_PX, IDX_PY, IDX_VX, IDX_VY, IDX_YAW = 0, 1, 2, 3, 4
IDX_BGX, IDX_BGY, IDX_BGZ = 5, 6, 7
IDX_BAX, IDX_BAY, IDX_BAZ = 8, 9, 10
IDX_VEL_SCALE = 11


def wrap_angle(angle: float) -> float:
    """Wrap to [-pi, pi]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def enu_to_body(vx: float, vy: float, yaw: float) -> Tuple[float, float]:
    """
    ENU 速度 → 车体速度 (前向 v_fwd, 横向 v_lat)。
    R_body_to_enu = [[cos(yaw), -sin(yaw)], [sin(yaw), cos(yaw)]]
    [vx, vy]^T = R @ [v_fwd, v_lat]^T
    """
    c, s = np.cos(yaw), np.sin(yaw)
    v_fwd = vx * c + vy * s
    v_lat = -vx * s + vy * c
    return float(v_fwd), float(v_lat)


def body_to_enu(v_fwd: float, v_lat: float, yaw: float) -> Tuple[float, float]:
    """车体 → ENU。"""
    c, s = np.cos(yaw), np.sin(yaw)
    vx = v_fwd * c - v_lat * s
    vy = v_fwd * s + v_lat * c
    return float(vx), float(vy)


def clip_biasnet_output(raw: np.ndarray, max_deg: float) -> np.ndarray:
    """BiasNet 原始输出 → ±max_deg deg/s，再在外部转 rad/s。"""
    return (max_deg * np.tanh(raw)).astype(np.float32)


def clip_biasnet_6d(raw: np.ndarray, max_acc_g: float, max_gyro_deg: float) -> np.ndarray:
    """
    BiasNet 6 维输出 → 物理量
    前3通道 (acc): ±max_acc_g [g]
    后3通道 (gyro): ±max_gyro_deg [deg/s]
    返回 (N, 6) 或 (6,) float32
    """
    raw = np.atleast_2d(raw) if raw.ndim == 1 else raw
    out = np.zeros_like(raw, dtype=np.float32)
    out[:, 0:3] = max_acc_g * np.tanh(raw[:, 0:3])      # acc bias [g]
    out[:, 3:6] = max_gyro_deg * np.tanh(raw[:, 3:6])   # gyro bias [deg/s]
    return out.squeeze()


def simulate_gnss_outage(
    gps_valid: np.ndarray,
    time_s: np.ndarray,
    start_time: float,
    end_time: float,
) -> Tuple[np.ndarray, int, int]:
    """
    在 [start_time, end_time)（相对 time_s[0]）内屏蔽 GNSS 量测更新。

    返回
    ----
    gps_valid_new, i_start, i_end
    """
    gps_v = gps_valid.copy()
    n = len(time_s)
    if n == 0:
        return gps_v, 0, 0
    t0 = float(time_s[0])
    i0 = int(np.searchsorted(time_s - t0, start_time))
    i1 = int(np.searchsorted(time_s - t0, end_time))
    i0 = min(max(i0, 0), n - 1)
    i1 = min(max(i1, i0 + 1), n)
    gps_v[i0:i1] = False
    return gps_v, i0, i1


# ============================================================================
# BiasNet —— 陀螺 Z 零偏预测（训练架构不变）
# ============================================================================

class BiasNet(Model):
    """
    浅版 BiasNet（保留作为基线/兼容旧权重）。
    输入：归一化后的 IMU 窗口  (batch, window, 7)
    输出：(batch, 6) — [ba_x, ba_y, ba_z, bg_x, bg_y, bg_z]
      ba_x/y/z : 加速度计零偏 [g]
      bg_x/y/z : 陀螺零偏 [rad/s]（原始输出为 deg/s，由 clip_biasnet_6d 转换）
    """

    def __init__(self, window_size: int = 30):
        super().__init__(name='BiasNet')
        self.conv1 = layers.Conv1D(32, 5, padding='causal', activation='relu')
        self.conv2 = layers.Conv1D(32, 5, dilation_rate=4,
                                   padding='causal', activation='relu')
        self.conv3 = layers.Conv1D(16, 3, padding='causal', activation='relu')
        self.pool = layers.GlobalAveragePooling1D()
        self.fc1 = layers.Dense(32, activation='relu')
        self.drop = layers.Dropout(0.2)
        self.out = layers.Dense(6, activation='linear')   # 6 维: 3 acc + 3 gyro

    def call(self, x, training=False):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.pool(h)
        h = self.fc1(h)
        h = self.drop(h, training=training)
        return self.out(h)


# ============================================================================
# DeepBiasNet —— 扩窗 + 深 TCN（用于学习慢变零偏）
# ============================================================================

class _TCNBlock(layers.Layer):
    """
    两层 causal conv + BN + ReLU + 残差。
    Kernel=3，receptive field per block ≈ 2*(k-1)*dilation = 4*dilation。
    """

    def __init__(self, channels: int, kernel: int, dilation: int, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv1D(channels, kernel, dilation_rate=dilation,
                                   padding='causal')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(channels, kernel, dilation_rate=dilation,
                                   padding='causal')
        self.bn2 = layers.BatchNormalization()

    def call(self, x, training=False):
        residual = x
        h = self.conv1(x)
        h = self.bn1(h, training=training)
        h = tf.nn.relu(h)
        h = self.conv2(h)
        h = self.bn2(h, training=training)
        return tf.nn.relu(h + residual)


class DeepBiasNet(Model):
    """
    扩窗 TCN：默认 window=200（20s @ 10Hz），dilations=(1,2,4,8,16,32)
    感受野 ≈ 1 + 4*sum(dilations) = 253 帧，覆盖整窗。

    参数量 ~80k（channels=48 时）。
    """

    def __init__(self,
                 window_size: int = 200,
                 channels: int = 48,
                 kernel: int = 3,
                 dilations: Tuple[int, ...] = (1, 2, 4, 8, 16, 32),
                 dropout: float = 0.2):
        super().__init__(name='DeepBiasNet')
        self.window_size = window_size
        self.stem_conv = layers.Conv1D(channels, 5, padding='causal')
        self.stem_bn = layers.BatchNormalization()
        self.tcn_blocks = [
            _TCNBlock(channels, kernel, d, name=f'tcn_d{d}') for d in dilations
        ]
        self.pool = layers.GlobalAveragePooling1D()
        self.fc1 = layers.Dense(64, activation='relu')
        self.drop = layers.Dropout(dropout)
        self.out_layer = layers.Dense(6, activation='linear')

    def call(self, x, training=False):
        h = self.stem_conv(x)
        h = self.stem_bn(h, training=training)
        h = tf.nn.relu(h)
        for blk in self.tcn_blocks:
            h = blk(h, training=training)
        h = self.pool(h)
        h = self.fc1(h)
        h = self.drop(h, training=training)
        return self.out_layer(h)


def make_biasnet(arch: str, window_size: int) -> Model:
    """
    arch: 'shallow' = 原 BiasNet；'deep' = DeepBiasNet（扩窗 TCN）
    """
    arch = (arch or 'shallow').lower()
    if arch in ('deep', 'tcn', 'deeptcn', 'deepbiasnet'):
        return DeepBiasNet(window_size=window_size)
    return BiasNet(window_size=window_size)


# ============================================================================
# CovAdapterNet —— AI 噪声参数适配器（论文 Section V-A）
# ============================================================================

class CovAdapterNet(Model):
    """
    输入：归一化后的 IMU 窗口 (batch, window, 6)
    输出：(batch, 2) — [z_lat, z_up]，用于动态缩放 NHC 和轮速量测噪声。

    架构与 BiasNet 相同，仅输出维度从 1 变为 2。
    """

    def __init__(self, window_size: int = 30):
        super().__init__(name='CovAdapterNet')
        self.conv1 = layers.Conv1D(32, 5, padding='causal', activation='relu')
        self.conv2 = layers.Conv1D(32, 5, dilation_rate=4,
                                   padding='causal', activation='relu')
        self.conv3 = layers.Conv1D(16, 3, padding='causal', activation='relu')
        self.pool = layers.GlobalAveragePooling1D()
        self.fc1 = layers.Dense(32, activation='relu')
        self.drop = layers.Dropout(0.2)
        self.out = layers.Dense(2, activation='linear')   # → [z_lat, z_up]

    def call(self, x, training=False):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.pool(h)
        h = self.fc1(h)
        h = self.drop(h, training=training)
        return self.out(h)


def map_noise_scale(z: np.ndarray, sigma_base: float, beta: float = 3.0) -> np.ndarray:
    """
    论文公式 (17)：z → 动态噪声标准差 σ_dyn。

        σ_dyn = σ_base * 10^(β * tanh(z))

    其中 β=3 时，缩放范围 ≈ 0.001x ~ 1000x。

    参数
    ----
    z          : (N, 2) 或 (2,) — [z_lat, z_up] 原始网络输出
    sigma_base : (2,) — [σ_lat, σ_up] 基础噪声标准差
    beta       : 动态范围控制

    返回
    ----
    sigma_dyn  : (N, 2) 或 (2,) — 方差
    """
    factor = 10.0 ** (beta * np.tanh(z))
    sigma = sigma_base.reshape(1, -1) * factor if z.ndim == 2 else sigma_base * factor
    return sigma ** 2  # 返回方差 R


# ============================================================================
# 6 状态 EKF
# ============================================================================

class EKF6D:
    """
    完整二维 GNSS/INS/Wheel 扩展卡尔曼滤波器（12 状态）。

    过程模型（加速度计积分速度）:
      ψ_{k+1} = wrap(ψ_k + (ω_z - b_net_z - b_gz) dt)
      a_body  = (accel_meas - ba_total) * g0        # 修正后加速度 [m/s²]
      a_enu   = R(yaw) @ a_body[:2]                  # 旋转到 ENU
      v_{k+1} = v_k + a_enu * dt                     # 加速度计积分
      p_{k+1} = p_k + v_k * dt
      零偏随机游走

    量测:
      GNSS → (px, py); 轮速 → v_fwd ≈ v_wheel;
      NHC → v_lat ≈ 0; 加速度计前向 → 约束 ba_x
    """

    def __init__(self, cfg: EKFConfig = None, x0: np.ndarray = None, P0: np.ndarray = None):
        self.cfg = cfg if cfg is not None else DEFAULT_EKF_CONFIG
        self.x = np.zeros(N_STATE, dtype=np.float64) if x0 is None else np.asarray(x0, dtype=np.float64)
        # 确保 vel_scale 初始化为 1
        if x0 is None:
            self.x[IDX_VEL_SCALE] = 1.0
        if P0 is not None:
            self.P = np.asarray(P0, dtype=np.float64)
        else:
            c = self.cfg
            self.P = np.diag([
                c.p_init_pos, c.p_init_pos,
                c.p_init_vel, c.p_init_vel,
                c.p_init_yaw,
                c.p_init_bg_xy, c.p_init_bg_xy, c.p_init_bg,
                c.p_init_ba, c.p_init_ba, c.p_init_ba,
                0.05 ** 2,   # vel_scale 初始方差（~5%不确定，允许估计比例因子）
            ]).astype(np.float64)

    # ---- 协方差维护 ----
    def _symmetrize_P(self):
        self.P = 0.5 * (self.P + self.P.T)
        d = np.diag(self.P).copy()
        d = np.maximum(d, self.cfg.eps)
        np.fill_diagonal(self.P, d)

    def _joseph_update(self, H: np.ndarray, innov: np.ndarray, R: np.ndarray):
        """Joseph form: P = (I-KH)P(I-KH)^T + K R K^T"""
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.solve(S, np.eye(S.shape[0]))
        self.x = self.x + K @ innov
        self.x[IDX_YAW] = wrap_angle(self.x[IDX_YAW])
        I_KH = np.eye(N_STATE) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self._symmetrize_P()

    # ---- 时间更新（轮速驱动前向速度）----
    def predict(self, gyro_meas: np.ndarray, accel_meas: np.ndarray,
                bias_net: np.ndarray, dt: float,
                freeze_yaw: bool = False, v_wheel: float = None):
        """
        gyro_meas  : (3,) 陀螺三轴 [rad/s]
        accel_meas : (3,) 加速度计三轴 [g]
        bias_net   : (6,) BiasNet 前馈零偏 [ba_x,ba_y,ba_z, bg_x,bg_y,bg_z]
        v_wheel    : 轮速 [m/s]，用于混合前向速度估计
        """
        px, py, vx, vy, yaw = self.x[:5]
        bgx, bgy, bgz = self.x[IDX_BGX], self.x[IDX_BGY], self.x[IDX_BGZ]
        bax, bay, baz = self.x[IDX_BAX], self.x[IDX_BAY], self.x[IDX_BAZ]
        vel_scale = self.x[IDX_VEL_SCALE]

        ba_net = bias_net[0:3].astype(np.float64)
        bg_net = bias_net[3:6].astype(np.float64) * DEG2RAD
        ba_total = ba_net + np.array([bax, bay, baz])
        bg_total = bg_net + np.array([bgx, bgy, bgz])

        # 1. 航向积分
        if freeze_yaw:
            dpsi = 0.0
        else:
            gyro_z_corr = float(gyro_meas[2]) - float(bg_total[2])
            dpsi = gyro_z_corr * dt
        yaw_new = wrap_angle(yaw + dpsi)
        c_yaw_new, s_yaw_new = np.cos(yaw_new), np.sin(yaw_new)
        c_yaw, s_yaw = np.cos(yaw), np.sin(yaw)

        # 2. 速度传播 — 前向用轮速(80%)混合EKF估计(20%)
        c_dpsi, s_dpsi = np.cos(dpsi), np.sin(dpsi)
        vx_rot = vx * c_dpsi - vy * s_dpsi
        vy_rot = vx * s_dpsi + vy * c_dpsi

        # 侧向速度保持 EKF 估计（受 NHC 约束）
        v_lat_ekf = -vx_rot * s_yaw_new + vy_rot * c_yaw_new

        if v_wheel is not None and v_wheel > 0.5:
            v_fwd_ekf = vx_rot * c_yaw_new + vy_rot * s_yaw_new
            v_fwd_wheel = float(v_wheel) * float(vel_scale)
            v_fwd = 0.8 * v_fwd_wheel + 0.2 * v_fwd_ekf
        else:
            v_fwd = vx_rot * c_yaw_new + vy_rot * s_yaw_new

        vx_new = v_fwd * c_yaw_new - v_lat_ekf * s_yaw_new
        vy_new = v_fwd * s_yaw_new + v_lat_ekf * c_yaw_new

        # 3. 位置积分
        px_new = px + v_fwd * c_yaw_new * dt
        py_new = py + v_fwd * s_yaw_new * dt

        # 4. 零偏随机游走
        self.x = np.array([
            px_new, py_new, vx_new, vy_new, yaw_new,
            bgx, bgy, bgz, bax, bay, baz, vel_scale,
        ], dtype=np.float64)

        # ---- 雅可比 F (12×12) ----
        F = np.eye(N_STATE, dtype=np.float64)
        # 位置 → vx,vy（车体前向速度投影）
        F[IDX_PX, IDX_VX] = c_yaw * c_yaw_new * dt * 0.2
        F[IDX_PX, IDX_VY] = s_yaw * c_yaw_new * dt * 0.2
        F[IDX_PY, IDX_VX] = c_yaw * s_yaw_new * dt * 0.2
        F[IDX_PY, IDX_VY] = s_yaw * s_yaw_new * dt * 0.2
        # 速度 → vx,vy
        F[IDX_VX, IDX_VX] = 0.2 * (c_dpsi * c_yaw_new - (-s_dpsi) * s_yaw_new)
        F[IDX_VX, IDX_VY] = 0.2 * (-s_dpsi * c_yaw_new - c_dpsi * s_yaw_new)
        F[IDX_VY, IDX_VX] = 0.2 * (c_dpsi * s_yaw_new + (-s_dpsi) * c_yaw_new)
        F[IDX_VY, IDX_VY] = 0.2 * (-s_dpsi * s_yaw_new + c_dpsi * c_yaw_new)
        # 位置 → yaw
        F[IDX_PX, IDX_YAW] = -v_fwd * s_yaw_new * dt
        F[IDX_PY, IDX_YAW] = v_fwd * c_yaw_new * dt
        # 位置 → vel_scale（前向速度来源）
        if v_wheel is not None and v_wheel > 0.5:
            F[IDX_PX, IDX_VEL_SCALE] = 0.8 * float(v_wheel) * c_yaw_new * dt
            F[IDX_PY, IDX_VEL_SCALE] = 0.8 * float(v_wheel) * s_yaw_new * dt
        # 航向 → bgz
        if not freeze_yaw:
            F[IDX_YAW, IDX_BGZ] = -dt

        c_cfg = self.cfg
        # 转弯自适应 Q：|yaw_rate| 越大，Q 越大（让滤波器在弯道更保守）
        yr = abs(float(gyro_meas[2]) - float(bg_total[2]))  # 修正后偏航率
        turn_factor = min(yr / c_cfg.turn_yaw_rate_max, 1.0)
        q_yaw_t = c_cfg.q_yaw * (1.0 + (c_cfg.turn_q_scale_yaw - 1.0) * turn_factor)
        q_vel_t = c_cfg.q_vel * (1.0 + (c_cfg.turn_q_scale_vel - 1.0) * turn_factor)
        q_bg_t  = c_cfg.q_bg  * (1.0 + (c_cfg.turn_q_scale_bg  - 1.0) * turn_factor)
        Q = np.diag([
            c_cfg.q_pos, c_cfg.q_pos,
            q_vel_t, q_vel_t,
            q_yaw_t,
            c_cfg.q_bg_xy, c_cfg.q_bg_xy, q_bg_t,
            c_cfg.q_ba, c_cfg.q_ba, c_cfg.q_ba,
            c_cfg.q_vel_scale,
        ])
        self.P = F @ self.P @ F.T + Q
        self._symmetrize_P()

    # ---- GNSS 位置量测 ----
    def update_gnss_position(self, px_gps: float, py_gps: float, r_xy: float = None):
        """z = [px, py]^T，线性量测，不使用 GPS heading。"""
        r = r_xy if r_xy is not None else self.cfg.r_gps_xy
        H = np.zeros((2, N_STATE))
        H[0, IDX_PX] = 1.0
        H[1, IDX_PY] = 1.0
        z = np.array([px_gps, py_gps], dtype=np.float64)
        z_pred = H @ self.x
        innov = z - z_pred
        R = np.diag([r, r])
        self._joseph_update(H, innov, R)

    # ---- GNSS 航向量测（新增）----
    def update_gnss_heading(self, heading_gps: float, r_heading: float = None):
        """
        z = heading_GPS，直接观测 yaw。
        仅在速度足够、GPS 航向可信时调用。
        """
        r = r_heading if r_heading is not None else self.cfg.r_gps_heading_rad2
        H = np.zeros((1, N_STATE))
        H[0, IDX_YAW] = 1.0
        z = np.array([float(heading_gps)], dtype=np.float64)
        z_pred = np.array([self.x[IDX_YAW]], dtype=np.float64)
        innov = z - z_pred
        innov[0] = wrap_angle(innov[0])   # 航向差值 wrap
        R = np.array([[r]], dtype=np.float64)
        self._joseph_update(H, innov, R)

    # ---- 轮速：车体前向速度量测 ----
    def update_wheel_forward(self, v_wheel: float, r_wheel: float = None):
        """
        h(x) = v_fwd - v_wheel * vel_scale ≈ 0
          v_fwd = vx*cos(yaw) + vy*sin(yaw)   (EKF 估计的车体前向速度)
        vel_scale 不在此量测中更新（用 update_wheel_scale 单独估计）。
        """
        r = r_wheel if r_wheel is not None else self.cfg.r_wheel
        yaw = self.x[IDX_YAW]
        vx, vy = self.x[IDX_VX], self.x[IDX_VY]
        vel_scale = self.x[IDX_VEL_SCALE]
        c, s = np.cos(yaw), np.sin(yaw)
        v_fwd = vx * c + vy * s
        v_wheel_corrected = float(v_wheel) * float(vel_scale)
        innov = np.array([v_wheel_corrected - v_fwd])
        H = np.zeros((1, N_STATE))
        H[0, IDX_VX] = c
        H[0, IDX_VY] = s
        H[0, IDX_YAW] = -vx * s + vy * c
        # vel_scale 不在此更新，避免与 vx/vy/yaw 不可区分
        R = np.array([[r]], dtype=np.float64)
        self._joseph_update(H, innov, R)

    # ---- 轮速比例因子估计（GPS 速度 / 轮速）----
    def update_wheel_scale(self, gps_speed_ms: float, v_wheel: float,
                           r_scale: float = None):
        """
        直接观测 vel_scale = v_gps / v_wheel。
        仅 GPS 有效且速度足够时调用，解耦于速度状态。
        """
        r = r_scale if r_scale is not None else 0.05 ** 2  # 5% 观测噪声（GPS 差分速度有噪声）
        if v_wheel < 0.5:
            return
        z_scale = float(gps_speed_ms) / float(v_wheel)
        z_scale = float(np.clip(z_scale, 0.5, 2.0))  # 防异常值
        vel_scale = self.x[IDX_VEL_SCALE]
        innov = np.array([z_scale - vel_scale])
        H = np.zeros((1, N_STATE))
        H[0, IDX_VEL_SCALE] = 1.0
        R = np.array([[r]], dtype=np.float64)
        self._joseph_update(H, innov, R)

    def update_wheel_speed(self, v_wheel: float, r_wheel: float = None):
        """h(x) = ||v|| ≈ v_wheel（低速或作补充）。"""
        r = r_wheel if r_wheel is not None else self.cfg.r_wheel
        vx, vy = self.x[IDX_VX], self.x[IDX_VY]
        speed = float(np.hypot(vx, vy))
        speed_safe = max(speed, self.cfg.min_speed_ms)
        innov = np.array([float(v_wheel) - speed])
        H = np.zeros((1, N_STATE))
        H[0, IDX_VX] = vx / speed_safe
        H[0, IDX_VY] = vy / speed_safe
        R = np.array([[r]], dtype=np.float64)
        self._joseph_update(H, innov, R)

    # ---- NHC 横向速度伪量测 ----
    def update_nhc(self, yaw_rate: float = 0.0, r_nhc: float = None):
        """
        车体横向速度 v_lat ≈ 0:
          v_lat = -vx*sin(yaw) + vy*cos(yaw)
        转弯时 |yaw_rate| > 阈值 → R 放大，减弱约束。
        """
        r_base = r_nhc if r_nhc is not None else self.cfg.r_nhc
        scale = (self.cfg.nhc_r_scale_turn
                 if abs(float(yaw_rate)) > self.cfg.nhc_yaw_rate_thresh else 1.0)
        R = np.array([[r_base * scale]], dtype=np.float64)

        yaw = self.x[IDX_YAW]
        vx, vy = self.x[IDX_VX], self.x[IDX_VY]
        sn, cs = np.sin(yaw), np.cos(yaw)
        h = -vx * sn + vy * cs
        innov = np.array([-h])  # z = 0

        H = np.zeros((1, N_STATE))
        H[0, IDX_VX] = -sn
        H[0, IDX_VY] = cs
        H[0, IDX_YAW] = -vx * cs - vy * sn
        self._joseph_update(H, innov, R)

    # ---- 转弯航向伪量测：a_lat ≈ v_fwd * yaw_rate ----
    def update_turn_heading(self, gyro_z_meas: float, acc_y_g: float,
                            ba_total_y: float, bg_net_z: float,
                            r_turn: float = None):
        """
        转弯运动学约束：侧向加速度应满足 a_lat = v_fwd * ω_z。
        若实测 a_lat 与预测值不匹配，说明 ω_z（航向变化率）有偏差 → 修正 bgz/航向。

        仅在 |yaw_rate| > 阈值时调用（明显转弯）。
        """
        yaw = self.x[IDX_YAW]
        vx, vy = self.x[IDX_VX], self.x[IDX_VY]
        bgz = self.x[IDX_BGZ]
        c, s = np.cos(yaw), np.sin(yaw)

        yaw_rate = float(gyro_z_meas) - bg_net_z - bgz
        v_fwd = vx * c + vy * s

        # 预测侧向加速度（运动学）
        a_lat_pred = v_fwd * yaw_rate
        # 实测侧向加速度（IMU，已修正零偏）
        a_lat_meas = (float(acc_y_g) - ba_total_y) * GRAVITY

        innov = np.array([a_lat_meas - a_lat_pred])
        H = np.zeros((1, N_STATE))
        H[0, IDX_VX] = c * yaw_rate
        H[0, IDX_VY] = s * yaw_rate
        H[0, IDX_YAW] = (-vx * s + vy * c) * yaw_rate
        H[0, IDX_BGZ] = -v_fwd
        H[0, IDX_BAY] = GRAVITY

        r = r_turn if r_turn is not None else (1.0 ** 2)  # 1.0 m/s² 噪声
        R = np.array([[r]], dtype=np.float64)
        self._joseph_update(H, innov, R)

    # ---- 加速度计前向量测（约束 ba_x）----
    def update_accel_forward(self, acc_x_g: float, dv_wheel_dt: float,
                             r_accel: float = None):
        """
        利用轮速导数观测加速度计 X 轴零偏 ba_x：
          dv_wheel/dt ≈ (acc_x - ba_x) * g0
          → ba_x ≈ acc_x - dv_wheel/dt / g0

        z = acc_x_g - dv_wheel_dt / g0  (观测到的 ba_x)
        h(x) = bax                       (状态估计的 ba_x)
        """
        r = r_accel if r_accel is not None else self.cfg.r_accel
        z_ba_x = acc_x_g - dv_wheel_dt / GRAVITY     # 观测 ba_x [g]
        bax = self.x[IDX_BAX]
        innov = np.array([z_ba_x - bax])
        H = np.zeros((1, N_STATE))
        H[0, IDX_BAX] = 1.0
        R = np.array([[r]], dtype=np.float64)
        self._joseph_update(H, innov, R)

    # ---- 状态访问 ----
    @property
    def px(self) -> float:
        return float(self.x[IDX_PX])

    @property
    def py(self) -> float:
        return float(self.x[IDX_PY])

    @property
    def vx(self) -> float:
        return float(self.x[IDX_VX])

    @property
    def vy(self) -> float:
        return float(self.x[IDX_VY])

    @property
    def yaw(self) -> float:
        return float(self.x[IDX_YAW])

    @property
    def bg(self) -> float:
        return float(self.x[IDX_BGZ])

    @property
    def bg_vec(self) -> np.ndarray:
        """返回陀螺三轴残余零偏 [rad/s]"""
        return self.x[IDX_BGX:IDX_BGZ+1].copy()

    @property
    def ba_vec(self) -> np.ndarray:
        """返回加速度计三轴残余零偏 [g]"""
        return self.x[IDX_BAX:IDX_BAZ+1].copy()


# 向后兼容：旧 2 状态航向 EKF 名称保留为别名（新代码请用 EKF6D）
EKF2D = EKF6D


# ============================================================================
# 推理导航器
# ============================================================================

class EKFNavigatorNP:
    """
    BiasNet + 6 状态 EKF 融合导航器（纯 NumPy 推理循环）。

    用法
    ----
    nav = EKFNavigatorNP(weights_path, norm_stats)
    px, py, yaw, net_bias, vx, vy, ekf_bg = nav.run(
        imu_raw, v_ms, gyro_z_rad,
        gps_enu_x, gps_enu_y, gps_valid, dt)
    """

    def __init__(self,
                 weights_path: str,
                 norm_stats: dict,
                 window_size: int = 30,
                 ekf_config: EKFConfig = None,
                 cov_weights_path: Optional[str] = None):
        self.window_size = window_size
        self.norm_stats = norm_stats
        self.ekf_config = ekf_config if ekf_config is not None else DEFAULT_EKF_CONFIG

        # BiasNet 输入: 6 IMU 通道 + 1 轮速通道 = 7
        self.biasnet = BiasNet(window_size)
        dummy = np.zeros((1, window_size, 7), dtype=np.float32)
        self.biasnet(dummy)
        wp = str(weights_path)
        if not wp.endswith('.weights.h5'):
            wp += '.weights.h5'
        self.biasnet.load_weights(wp)
        print(f"[EKFNavigator] BiasNet (6D 输出) 加载成功：{weights_path}")

        # CovAdapterNet（可选）
        self.cov_adapter = None
        if cov_weights_path is not None:
            self.cov_adapter = CovAdapterNet(window_size)
            dummy_cov = np.zeros((1, window_size, 6), dtype=np.float32)
            self.cov_adapter(dummy_cov)
            cwp = str(cov_weights_path)
            if not cwp.endswith('.weights.h5'):
                cwp += '.weights.h5'
            self.cov_adapter.load_weights(cwp)
            print(f"[EKFNavigator] CovAdapterNet 加载成功：{cov_weights_path}")
        else:
            print("[EKFNavigator] CovAdapterNet 未加载（使用固定噪声）")

    def _normalize_imu(self, imu_raw: np.ndarray) -> np.ndarray:
        """归一化 7 通道 [AccX/Y/Z, GyroX/Y/Z, VehicleSpeed_ms]。"""
        keys = ['AccX_g', 'AccY_g', 'AccZ_g',
                'GyroX_degs', 'GyroY_degs', 'GyroZ_degs',
                'VehicleSpeed_ms']
        out = np.empty_like(imu_raw, dtype=np.float32)
        for i, k in enumerate(keys):
            mu = self.norm_stats[k]['mean']
            std = self.norm_stats[k]['std']
            out[:, i] = (imu_raw[:, i] - mu) / (std + EPS)
        return out

    def _predict_bias(self, imu_norm: np.ndarray) -> np.ndarray:
        """
        BiasNet → 6 维物理量 [ba_x, ba_y, ba_z, bg_x, bg_y, bg_z]
          前3通道: acc bias [g]
          后3通道: gyro bias [deg/s]（调用方自行转 rad/s）
        返回 (T, 6) float32
        """
        T = len(imu_norm)
        W = self.window_size
        bias = np.zeros((T, 6), dtype=np.float32)
        if T < W:
            return bias
        windows = np.stack(
            [imu_norm[i: i + W] for i in range(T - W + 1)], axis=0)
        raw = self.biasnet(windows.astype(np.float32),
                           training=False).numpy()            # (N, 6)
        bias_phys = clip_biasnet_6d(raw, self.ekf_config.biasnet_max_acc_g,
                                    self.ekf_config.biasnet_max_deg)
        bias[W - 1:] = bias_phys.astype(np.float32)
        return bias

    def _predict_noise(self, imu_norm: np.ndarray) -> np.ndarray:
        """
        CovAdapterNet → [z_lat, z_up] → 动态量测噪声方差 (T, 2)。

        若未加载 CovAdapterNet，返回基础噪声（从 config 读取）。
        返回列: [R_lat, R_up] 单位为 (m/s)²。
        """
        cfg = self.ekf_config
        T = len(imu_norm)
        W = self.window_size
        # 默认基础噪声
        base = np.tile([cfg.r_nhc, cfg.r_wheel], (T, 1)).astype(np.float32)

        if self.cov_adapter is None or T < W:
            return base

        windows = np.stack(
            [imu_norm[i: i + W] for i in range(T - W + 1)], axis=0)
        z_raw = self.cov_adapter(windows.astype(np.float32),
                                 training=False).numpy()           # (N, 2)
        z_full = np.zeros((T, 2), dtype=np.float32)
        z_full[W - 1:] = z_raw

        sigma_base = np.array([np.sqrt(cfg.r_nhc), np.sqrt(cfg.r_wheel)],
                              dtype=np.float32)
        for k in range(W - 1, T):
            r_dyn = map_noise_scale(z_full[k], sigma_base, cfg.beta_noise_scale)
            base[k, 0] = float(r_dyn[0])
            base[k, 1] = float(r_dyn[1])

        return base  # (T, 2) — [R_lat, R_up] 方差

    def _init_state(self, gps_x, gps_y, gps_v, v_ms, gps_theta, gyro_z, net_bias,
                     dt=0.1, time_s=None):
        """
        初始化 12 维状态并确定 EKF 循环起始帧 k0。

        net_bias: (T, 6) — BiasNet 6 维前馈零偏 [ba_x, ba_y, ba_z, bg_x, bg_y, bg_z]
                  单位: ba [g], bg [deg/s]
        """
        cfg = self.ekf_config
        min_hdg_ms = 5.0 / 3.6  # 5 km/h
        min_wheel_ms = cfg.min_speed_wheel_ms  # 轮速更新门限

        ok = np.where(gps_v & np.isfinite(gps_x) & np.isfinite(gps_y))[0]
        if len(ok) == 0:
            x0 = np.zeros(N_STATE)
            x0[IDX_VEL_SCALE] = 1.0
            return x0, 0
        k0_gps = int(ok[0])

        # 位置：用首个有效 GNSS 帧
        px0 = float(gps_x[k0_gps])
        py0 = float(gps_y[k0_gps])
        v0 = float(v_ms[k0_gps]) if np.isfinite(v_ms[k0_gps]) else 0.0

        # ---- 航向初始化 ----
        yaw0 = 0.0
        if np.isfinite(gps_theta[k0_gps]) and abs(gps_theta[k0_gps]) > 1e-6:
            yaw0 = float(gps_theta[k0_gps])
        else:
            moving = np.where((v_ms >= min_hdg_ms)
                              & np.isfinite(gps_x) & np.isfinite(gps_y))[0]
            if len(moving) > 0:
                i_m = int(moving[0])
                if np.isfinite(gps_theta[i_m]) and abs(gps_theta[i_m]) > 1e-6:
                    yaw0 = float(gps_theta[i_m])
                else:
                    i_end = min(i_m + 10, len(gps_x) - 1)
                    dx = float(gps_x[i_end] - gps_x[i_m])
                    dy = float(gps_y[i_end] - gps_y[i_m])
                    if dx * dx + dy * dy > 1.0:
                        yaw0 = float(np.arctan2(dy, dx))

        # ---- 从静止时段估计陀螺 Z 初始残余零偏 ----
        bgz0 = 0.0

        # 1) 静止时段估计：v_ms ~ 0 时 gyro_z ≈ 零偏
        still = (np.abs(v_ms) < 0.1) & np.isfinite(gyro_z) & gps_v
        still_idx = np.where(still)[0]
        if len(still_idx) >= 10:
            gaps = np.diff(still_idx)
            long_run = np.where(gaps <= 3)[0]
            if len(long_run) >= 5:
                # 扣除 BiasNet 前馈后的残余
                nb_z = net_bias[still_idx, 5] * DEG2RAD  # deg/s → rad/s
                bgz0 = float(np.median(gyro_z[still_idx] - nb_z))
                bgz0 = np.clip(bgz0, -cfg.bg_init_max_bg_rads, cfg.bg_init_max_bg_rads)
        else:
            # 2) 运动时段：用 GPS 位移推算
            high_spd = (gps_v & (v_ms >= cfg.bg_init_min_speed_ms)
                        & np.isfinite(gps_x) & np.isfinite(gps_y))
            spd_idx = np.where(high_spd)[0]
            if len(spd_idx) >= 6:
                total_s = float(time_s[spd_idx[-1]] - time_s[spd_idx[0]]) if time_s is not None else len(spd_idx) * dt
                if total_s >= cfg.bg_init_window_s * 0.5:
                    mid = len(spd_idx) // 2
                    h1_idx = spd_idx[:mid]
                    h2_idx = spd_idx[mid:]

                    def avg_heading(idx_win):
                        dx = float(gps_x[idx_win[-1]] - gps_x[idx_win[0]])
                        dy = float(gps_y[idx_win[-1]] - gps_y[idx_win[0]])
                        if dx * dx + dy * dy < 1.0:
                            return None, 0.0
                        return np.arctan2(dy, dx), np.sqrt(dx*dx + dy*dy)

                    hdg1, d1 = avg_heading(h1_idx)
                    hdg2, d2 = avg_heading(h2_idx)
                    if hdg1 is not None and hdg2 is not None and h1_idx[-1] < h2_idx[0]:
                        gps_delta = wrap_angle(float(hdg2) - float(hdg1))

                        t1 = float(time_s[h1_idx[-1]]) if time_s is not None else h1_idx[-1] * dt
                        t2 = float(time_s[h2_idx[0]]) if time_s is not None else h2_idx[0] * dt
                        i1 = min(h1_idx[-1], len(gyro_z) - 1)
                        i2 = min(h2_idx[0], len(gyro_z) - 1)
                        gyro_sum = 0.0
                        for j in range(i1, i2):
                            dt_j = float(time_s[j+1] - time_s[j]) if time_s is not None else dt
                            nb_z = float(net_bias[j, 5]) * DEG2RAD if net_bias.shape[1] > 5 else 0.0
                            gyro_sum += (float(gyro_z[j]) - nb_z) * dt_j

                        duraction_s = t2 - t1
                        if duraction_s > 1.0:
                            bgz0 = (gyro_sum - gps_delta) / duraction_s
                            bgz0 = np.clip(bgz0, -cfg.bg_init_max_bg_rads, cfg.bg_init_max_bg_rads)

        # ---- 确定 EKF 循环起始帧 ----
        if v0 < min_wheel_ms:
            motion = np.where((v_ms >= min_wheel_ms)
                              & np.isfinite(gps_x) & np.isfinite(gps_y))[0]
            if len(motion) > 0:
                k0 = int(motion[0])
            else:
                k0 = k0_gps
        else:
            k0 = k0_gps

        vk = float(v_ms[k0]) if np.isfinite(v_ms[k0]) else 0.0
        vx0, vy0 = body_to_enu(vk, 0.0, yaw0)
        x0 = np.zeros(N_STATE, dtype=np.float64)
        x0[IDX_PX:IDX_YAW+1] = [px0, py0, vx0, vy0, wrap_angle(yaw0)]
        x0[IDX_BGZ] = bgz0
        x0[IDX_VEL_SCALE] = 1.0
        return x0, k0

    def run(self,
            imu_raw: np.ndarray,
            v_ms: np.ndarray,
            gyro_z_rad: np.ndarray,
            gps_enu_x: np.ndarray,
            gps_enu_y: np.ndarray,
            gps_valid: np.ndarray,
            dt: float = 0.1,
            time_s: Optional[np.ndarray] = None,
            gps_theta: Optional[np.ndarray] = None,
            enable_wheel_meas: bool = True,
            enable_nhc: bool = True,
            enable_gnss_heading: bool = True,
            enable_accel_meas: bool = True,
            ) -> tuple:
        """
        参数
        ----
        imu_raw     : (T, 6) 原始 IMU [AccX,Y,Z(g), GyroX,Y,Z(deg/s)]
        v_ms        : (T,) 车速 m/s（已 km/h→m/s）
        gyro_z_rad  : (T,) 陀螺 Z rad/s（向后兼容，从 imu_raw[:,5] 转换）
        gps_enu_x/y : (T,) GNSS ENU 位置 (m)
        gps_valid   : (T,) bool，False 时不做 GNSS 更新（含 outage 模拟）
        gps_theta   : (T,) 可选，用于初始化航向 + 航向量测

        返回
        ----
        enu_x, enu_y, headings, net_biases, vel_x, vel_y, ekf_bgz,
        ekf_bg_vec, ekf_ba_vec
          net_biases   : (T, 6) BiasNet 6 维前馈 [ba_x,ba_y,ba_z, bg_x,bg_y,bg_z]
          ekf_bgz      : (T,)  EKF 陀螺 Z 残余零偏 [rad/s]
          ekf_bg_vec   : (T, 3) EKF 陀螺三轴残余零偏 [rad/s]
          ekf_ba_vec   : (T, 3) EKF 加速度计三轴残余零偏 [g]
        """
        T = len(v_ms)
        if gps_theta is None:
            gps_theta = np.zeros(T, dtype=np.float32)

        # 将轮速拼入 IMU 作为 BiasNet 第7通道
        imu_7ch = np.column_stack([imu_raw, v_ms]) if imu_raw.shape[1] == 6 else imu_raw
        imu_norm = self._normalize_imu(imu_7ch)
        net_bias = self._predict_bias(imu_norm)        # (T, 6)
        noise_r = self._predict_noise(imu_norm[:, :6])

        # 轮速导数（用于加速度计前向约束）
        dv_wheel = np.zeros(T, dtype=np.float32)
        if T > 1:
            # 用中值滤波平滑后求导
            from scipy.ndimage import median_filter
            v_smooth = median_filter(v_ms, size=5)
            dv_wheel[1:] = np.diff(v_smooth) / np.clip(np.diff(time_s) if time_s is not None else dt, 0.05, 0.5)

        x0, k_start = self._init_state(
            gps_enu_x, gps_enu_y, gps_valid, v_ms, gps_theta, gyro_z_rad, net_bias,
            dt=dt, time_s=time_s)
        ekf = EKF6D(cfg=self.ekf_config, x0=x0)

        enu_x = np.full(T, np.nan, np.float32)
        enu_y = np.full(T, np.nan, np.float32)
        headings = np.zeros(T, np.float32)
        vel_x = np.zeros(T, np.float32)
        vel_y = np.zeros(T, np.float32)
        ekf_bgz = np.zeros(T, np.float32)
        ekf_bg_vec = np.zeros((T, 3), np.float32)
        ekf_ba_vec = np.zeros((T, 3), np.float32)
        ekf_vel_scale = np.ones(T, np.float32)

        cfg = self.ekf_config
        for k in range(k_start, T):
            if time_s is not None and k > 0:
                dt_k = float(time_s[k] - time_s[k - 1])
                dt_k = float(np.clip(dt_k, cfg.dt_min, cfg.dt_max))
            else:
                dt_k = float(dt)

            # 陀螺三轴 [rad/s]，加速度计三轴 [g]
            gyro_meas = np.array([
                float(imu_raw[k, 3]) * DEG2RAD,
                float(imu_raw[k, 4]) * DEG2RAD,
                float(gyro_z_rad[k]) if np.isfinite(gyro_z_rad[k]) else 0.0,
            ], dtype=np.float64)
            accel_meas = np.array([
                float(imu_raw[k, 0]),
                float(imu_raw[k, 1]),
                float(imu_raw[k, 2]),
            ], dtype=np.float64)

            bn = net_bias[k].astype(np.float64)        # (6,) [g, g, g, deg/s, deg/s, deg/s]
            vw = float(v_ms[k]) if np.isfinite(v_ms[k]) else 0.0

            still = vw < cfg.freeze_yaw_below_ms
            ekf.predict(gyro_meas, accel_meas, bn, dt_k,
                        freeze_yaw=still, v_wheel=vw)

            gps_ok_now = (gps_valid[k] and np.isfinite(gps_enu_x[k])
                          and np.isfinite(gps_enu_y[k]))
            if gps_ok_now:
                ekf.update_gnss_position(float(gps_enu_x[k]), float(gps_enu_y[k]))
                # 轮速比例因子：GPS 速度 / 轮速（前后帧都需 GPS 有效）
                if (k > k_start and enable_wheel_meas
                        and gps_valid[k - 1]
                        and np.isfinite(gps_enu_x[k - 1])):
                    gps_spd = float(np.hypot(
                        gps_enu_x[k] - gps_enu_x[k - 1],
                        gps_enu_y[k] - gps_enu_y[k - 1])) / max(dt_k, 0.05)
                    if gps_spd > 0.5:
                        ekf.update_wheel_scale(gps_spd, vw)

            if enable_wheel_meas and vw >= cfg.min_speed_wheel_ms:
                r_wheel_dyn = float(noise_r[k, 1])
                ekf.update_wheel_forward(vw, r_wheel=r_wheel_dyn)

            if enable_nhc and vw >= cfg.min_speed_nhc_ms:
                gn = bn[5] * DEG2RAD                                 # net gyro_z bias [rad/s]
                yaw_rate = float(gyro_meas[2]) - gn - float(ekf.x[IDX_BGZ])
                r_nhc_dyn = float(noise_r[k, 0])
                ekf.update_nhc(yaw_rate=yaw_rate, r_nhc=r_nhc_dyn)

                # 转弯航向伪量测：a_lat ≈ v_fwd * yaw_rate（仅outage时启用）
                if (abs(yaw_rate) > 0.15
                        and not (gps_valid[k] and np.isfinite(gps_enu_x[k])
                                 and np.isfinite(gps_enu_y[k]))):
                    ba_net_y = bn[1]
                    ekf.update_turn_heading(
                        float(gyro_meas[2]), float(accel_meas[1]),
                        float(ba_net_y + ekf.x[IDX_BAY]),
                        gn)

            # 加速度计前向约束（用轮速导数约束 ba_x）
            if enable_accel_meas and vw >= cfg.min_speed_wheel_ms:
                ekf.update_accel_forward(
                    float(accel_meas[0]), float(dv_wheel[k]))

            if (enable_gnss_heading and gps_valid[k]
                    and np.isfinite(gps_theta[k])
                    and vw >= cfg.min_speed_wheel_ms):
                ekf.update_gnss_heading(float(gps_theta[k]))

            enu_x[k] = ekf.px
            enu_y[k] = ekf.py
            headings[k] = ekf.yaw
            vel_x[k] = ekf.vx
            vel_y[k] = ekf.vy
            ekf_bgz[k] = float(ekf.x[IDX_BGZ])
            ekf_bg_vec[k] = ekf.bg_vec.astype(np.float32)
            ekf_ba_vec[k] = ekf.ba_vec.astype(np.float32)
            ekf_vel_scale[k] = float(ekf.x[IDX_VEL_SCALE])

        # 初始化前帧用首帧状态填充
        if k_start > 0:
            headings[:k_start] = headings[k_start]
            vel_x[:k_start] = vel_x[k_start]
            vel_y[:k_start] = vel_y[k_start]
            ekf_bgz[:k_start] = ekf_bgz[k_start]
            ekf_bg_vec[:k_start] = ekf_bg_vec[k_start]
            ekf_ba_vec[:k_start] = ekf_ba_vec[k_start]
            ekf_vel_scale[:k_start] = ekf_vel_scale[k_start]
            enu_x[:k_start] = enu_x[k_start]
            enu_y[:k_start] = enu_y[k_start]

        return (enu_x, enu_y, headings, net_bias, vel_x, vel_y, ekf_bgz,
                ekf_bg_vec, ekf_ba_vec, ekf_vel_scale)


def load_norm_stats(path: str) -> dict:
    import json
    with open(path) as f:
        data = json.load(f)
    if 'stats' in data:
        return data['stats']
    return data


def evaluate_trajectory(
    pred_x: np.ndarray,
    pred_y: np.ndarray,
    pred_yaw: np.ndarray,
    truth_x: np.ndarray,
    truth_y: np.ndarray,
    truth_yaw: np.ndarray,
    eval_mask: np.ndarray,
    outage_mask: Optional[np.ndarray] = None,
) -> dict:
    """RMSE、中值、终点误差、航向误差、outage 最大误差。"""
    idx = np.where(eval_mask & np.isfinite(truth_x) & np.isfinite(truth_y))[0]
    if len(idx) == 0:
        return {}

    err = np.sqrt((pred_x[idx] - truth_x[idx]) ** 2 + (pred_y[idx] - truth_y[idx]) ** 2)
    metrics = {
        'rmse_m': float(np.sqrt(np.mean(err ** 2))),
        'median_m': float(np.median(err)),
        'final_m': float(err[-1]),
        'max_m': float(np.max(err)),
    }

    if outage_mask is not None:
        oidx = idx[outage_mask[idx]]
        if len(oidx) > 0:
            oerr = np.sqrt(
                (pred_x[oidx] - truth_x[oidx]) ** 2
                + (pred_y[oidx] - truth_y[oidx]) ** 2)
            metrics['outage_max_m'] = float(np.max(oerr))
            metrics['outage_median_m'] = float(np.median(oerr))
            metrics['outage_final_m'] = float(oerr[-1])

    yaw_idx = idx[np.isfinite(truth_yaw[idx])]
    if len(yaw_idx) > 0:
        dyaw = np.array([
            wrap_angle(float(pred_yaw[i]) - float(truth_yaw[i]))
            for i in yaw_idx
        ])
        metrics['heading_rmse_deg'] = float(np.sqrt(np.mean(dyaw ** 2)) * RAD2DEG)
        metrics['heading_median_deg'] = float(np.median(np.abs(dyaw)) * RAD2DEG)

        # 按 outage 拆分航向误差
        if outage_mask is not None:
            no_idx = yaw_idx[~outage_mask[yaw_idx]]
            if len(no_idx) > 0:
                dyaw_no = np.array([
                    wrap_angle(float(pred_yaw[i]) - float(truth_yaw[i]))
                    for i in no_idx
                ])
                metrics['heading_rmse_no_outage_deg'] = float(np.sqrt(np.mean(dyaw_no ** 2)) * RAD2DEG)
            o_yaw = yaw_idx[outage_mask[yaw_idx]]
            if len(o_yaw) > 0:
                dyaw_o = np.array([
                    wrap_angle(float(pred_yaw[i]) - float(truth_yaw[i]))
                    for i in o_yaw
                ])
                metrics['heading_rmse_outage_deg'] = float(np.sqrt(np.mean(dyaw_o ** 2)) * RAD2DEG)

    return metrics
