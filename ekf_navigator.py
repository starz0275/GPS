"""
EKF 二维地面车辆 GNSS/INS/Wheel 融合导航 + BiasNet 陀螺零偏预测

状态 x = [px, py, vx, vy, yaw, bg]^T
  px, py : ENU 位置 (m)
  vx, vy : ENU 速度 (m/s)
  yaw    : 航向 (rad, 东为 0, 逆时针为正)
  bg     : 陀螺 Z 残余零偏 (rad/s)，BiasNet 已扣除主零偏

量测更新:
  - GNSS 位置 (px, py)
  - 轮速前向 v_fwd = vx*cos(yaw)+vy*sin(yaw) ≈ v_wheel
  - NHC 横向 v_lat ≈ 0（转弯时动态放宽 R）
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
EPS = 1e-8
N_STATE = 6
IDX_PX, IDX_PY, IDX_VX, IDX_VY, IDX_YAW, IDX_BG = range(6)


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
    t0 = float(time_s[0])
    i0 = int(np.searchsorted(time_s - t0, start_time))
    i1 = int(np.searchsorted(time_s - t0, end_time))
    i1 = min(max(i1, i0 + 1), len(time_s))
    gps_v[i0:i1] = False
    return gps_v, i0, i1


# ============================================================================
# BiasNet —— 陀螺 Z 零偏预测（训练架构不变）
# ============================================================================

class BiasNet(Model):
    """
    输入：归一化后的 IMU 窗口  (batch, window, 6)
    输出：(batch, 1) — 陀螺 Z 零偏 b̂_ω (rad/s)
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
        self.out = layers.Dense(1, activation='linear')

    def call(self, x, training=False):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.pool(h)
        h = self.fc1(h)
        h = self.drop(h, training=training)
        return self.out(h)


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
    完整二维 GNSS/INS/Wheel 扩展卡尔曼滤波器。

    过程模型（vx, vy 为状态，轮速仅作量测）:
      ψ_{k+1} = wrap(ψ_k + (ω_z - b_net - b_g) dt)
      [vx, vy]_{k+1} = R(Δψ) [vx, vy]_k^T     # 航向变化时旋转 ENU 速度
      px_{k+1} = px_k + vx_k dt
      py_{k+1} = py_k + vy_k dt
      b_g 随机游走

    量测:
      GNSS → (px, py); 轮速 → ||v||;  NHC → v_lat ≈ 0
    """

    def __init__(self, cfg: EKFConfig = None, x0: np.ndarray = None, P0: np.ndarray = None):
        self.cfg = cfg if cfg is not None else DEFAULT_EKF_CONFIG
        self.x = np.zeros(N_STATE, dtype=np.float64) if x0 is None else np.asarray(x0, dtype=np.float64)
        if P0 is not None:
            self.P = np.asarray(P0, dtype=np.float64)
        else:
            c = self.cfg
            self.P = np.diag([
                c.p_init_pos, c.p_init_pos,
                c.p_init_vel, c.p_init_vel,
                c.p_init_yaw, c.p_init_bg,
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

    # ---- 时间更新（禁止用轮速覆盖 vx, vy）----
    def predict(self, gyro_z_rad: float, bias_net: float, dt: float,
                freeze_yaw: bool = False):
        """
        gyro_z_rad : 原始陀螺 Z (rad/s)
        bias_net   : BiasNet 预测零偏 (rad/s)，不写入状态
        dt         : 时间步长 (s)
        freeze_yaw : True 时不积分航向（零速/停车，避免陀螺零偏积到 yaw）
        """
        px, py, vx, vy, yaw, bg = self.x
        if freeze_yaw:
            dpsi = 0.0
        else:
            omega = float(gyro_z_rad) - float(bias_net) - bg
            dpsi = omega * dt
        c, s = np.cos(dpsi), np.sin(dpsi)

        # 位置：用当前 ENU 速度积分
        px_new = px + vx * dt
        py_new = py + vy * dt
        # 速度：随航向变化旋转（协调转弯），不由轮速强制赋值
        vx_new = vx * c - vy * s
        vy_new = vx * s + vy * c
        yaw_new = wrap_angle(yaw + dpsi)

        self.x = np.array([px_new, py_new, vx_new, vy_new, yaw_new, bg], dtype=np.float64)

        # 雅可比 F
        F = np.eye(N_STATE, dtype=np.float64)
        F[IDX_PX, IDX_VX] = dt
        F[IDX_PY, IDX_VY] = dt
        F[IDX_VX, IDX_VX] = c
        F[IDX_VX, IDX_VY] = -s
        F[IDX_VY, IDX_VX] = s
        F[IDX_VY, IDX_VY] = c
        # ∂vx+/∂bg, ∂vy+/∂bg 通过 dpsi
        dvx_dbg = (vx * s + vy * c) * dt
        dvy_dbg = (-vx * c + vy * s) * dt
        F[IDX_VX, IDX_BG] = dvx_dbg
        F[IDX_VY, IDX_BG] = dvy_dbg
        F[IDX_YAW, IDX_BG] = -dt

        c_cfg = self.cfg
        Q = np.diag([
            c_cfg.q_pos, c_cfg.q_pos,
            c_cfg.q_vel, c_cfg.q_vel,
            c_cfg.q_yaw, c_cfg.q_bg,
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

    # ---- 轮速：车体前向速度量测（优于仅用标量速率，可约束方向）----
    def update_wheel_forward(self, v_wheel: float, r_wheel: float = None):
        """
        h(x) = v_fwd - v_wheel = (vx*cos(yaw) + vy*sin(yaw)) - v_wheel ≈ 0
        不覆盖 vx,vy，仅通过 EKF 校正。
        """
        r = r_wheel if r_wheel is not None else self.cfg.r_wheel
        yaw = self.x[IDX_YAW]
        vx, vy = self.x[IDX_VX], self.x[IDX_VY]
        c, s = np.cos(yaw), np.sin(yaw)
        v_fwd = vx * c + vy * s
        innov = np.array([float(v_wheel) - v_fwd])
        H = np.zeros((1, N_STATE))
        H[0, IDX_VX] = c
        H[0, IDX_VY] = s
        H[0, IDX_YAW] = -vx * s + vy * c
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
        return float(self.x[IDX_BG])


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

        self.biasnet = BiasNet(window_size)
        dummy = np.zeros((1, window_size, 6), dtype=np.float32)
        self.biasnet(dummy)
        wp = str(weights_path)
        if not wp.endswith('.weights.h5'):
            wp += '.weights.h5'
        self.biasnet.load_weights(wp)
        print(f"[EKFNavigator] BiasNet 加载成功：{weights_path}")

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
        keys = ['AccX_g', 'AccY_g', 'AccZ_g',
                'GyroX_degs', 'GyroY_degs', 'GyroZ_degs']
        out = np.empty_like(imu_raw, dtype=np.float32)
        for i, k in enumerate(keys):
            mu = self.norm_stats[k]['mean']
            std = self.norm_stats[k]['std']
            out[:, i] = (imu_raw[:, i] - mu) / (std + EPS)
        return out

    def _predict_bias(self, imu_norm: np.ndarray) -> np.ndarray:
        """BiasNet → tanh 限幅 ±biasnet_max_deg deg/s → rad/s。"""
        T = len(imu_norm)
        W = self.window_size
        bias = np.zeros(T, dtype=np.float32)
        if T < W:
            return bias
        windows = np.stack(
            [imu_norm[i: i + W] for i in range(T - W + 1)], axis=0)
        raw = self.biasnet(windows.astype(np.float32),
                           training=False).numpy().flatten()
        bias_deg = clip_biasnet_output(raw, self.ekf_config.biasnet_max_deg)
        bias[W - 1:] = bias_deg * DEG2RAD
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
        初始化状态并确定 EKF 循环起始帧 k0。

        车辆静止启动时：
          - 位置来自首个有效 GNSS 帧
          - 航向来自首次运动的 GPS 位移
          - k0 设在首次运动帧，避免静止期协方差膨胀
        """
        cfg = self.ekf_config
        min_hdg_ms = 5.0 / 3.6  # 5 km/h
        min_wheel_ms = cfg.min_speed_wheel_ms  # 轮速更新门限

        ok = np.where(gps_v & np.isfinite(gps_x) & np.isfinite(gps_y))[0]
        if len(ok) == 0:
            return np.zeros(N_STATE), 0
        k0_gps = int(ok[0])

        # 位置：用首个有效 GNSS 帧
        px0 = float(gps_x[k0_gps])
        py0 = float(gps_y[k0_gps])
        v0 = float(v_ms[k0_gps]) if np.isfinite(v_ms[k0_gps]) else 0.0

        # ---- 航向初始化 ----
        # 若首帧静止，向后找第一段运动，用 GPS 位移推算航向
        yaw0 = 0.0
        if v0 >= min_hdg_ms and np.isfinite(gps_theta[k0_gps]):
            yaw0 = float(gps_theta[k0_gps])
        else:
            moving = np.where((v_ms >= min_hdg_ms)
                              & np.isfinite(gps_x) & np.isfinite(gps_y))[0]
            if len(moving) >= 5:
                i_start = int(moving[0])
                i_end = min(i_start + 10, len(gps_x) - 1)
                dx = float(gps_x[i_end] - gps_x[i_start])
                dy = float(gps_y[i_end] - gps_y[i_start])
                if dx * dx + dy * dy > 4.0:  # ≥ 2m 位移
                    yaw0 = float(np.arctan2(dy, dx))
            if yaw0 == 0.0 and v0 > cfg.min_speed_ms:
                k1 = min(k0_gps + 5, len(gps_x) - 1)
                yaw0 = float(np.arctan2(
                    float(gps_y[k1]) - py0, float(gps_x[k1]) - px0))

        # ---- 从静止或 GPS 位移推算初始零偏 ----
        bg0 = 0.0

        # 1) 静止时段估计：v_ms ~ 0 时 gyro_z ≈ 零偏
        still = (np.abs(v_ms) < 0.1) & np.isfinite(gyro_z) & gps_v
        still_idx = np.where(still)[0]
        if len(still_idx) >= 10:
            gaps = np.diff(still_idx)
            long_run = np.where(gaps <= 3)[0]
            if len(long_run) >= 5:
                bg0 = float(np.median(gyro_z[still_idx]))
                bg0 = np.clip(bg0, -cfg.bg_init_max_bg_rads, cfg.bg_init_max_bg_rads)
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
                            nb = float(net_bias[j]) if np.isfinite(net_bias[j]) else 0.0
                            gyro_sum += (float(gyro_z[j]) - nb) * dt_j

                        duraction_s = t2 - t1
                        if duraction_s > 1.0:
                            bg0 = (gyro_sum - gps_delta) / duraction_s
                            bg0 = np.clip(bg0, -cfg.bg_init_max_bg_rads, cfg.bg_init_max_bg_rads)

        # ---- 确定 EKF 循环起始帧 ----
        # 首帧静止 → 延迟到首次运动，避免长串预测使协方差膨胀
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
        return np.array([px0, py0, vx0, vy0, wrap_angle(yaw0), bg0]), k0

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
            ) -> tuple:
        """
        参数
        ----
        imu_raw     : (T, 6) 原始 IMU
        v_ms        : (T,) 车速 m/s（已 km/h→m/s）
        gyro_z_rad  : (T,) 陀螺 Z rad/s
        gps_enu_x/y : (T,) GNSS ENU 位置 (m)
        gps_valid   : (T,) bool，False 时不做 GNSS 更新（含 outage 模拟）
        gps_theta   : (T,) 可选，仅用于初始化航向

        返回
        ----
        enu_x, enu_y, headings, net_biases, vel_x, vel_y, ekf_bg
        """
        T = len(v_ms)
        if gps_theta is None:
            gps_theta = np.zeros(T, dtype=np.float32)

        imu_norm = self._normalize_imu(imu_raw)
        net_bias = self._predict_bias(imu_norm)
        noise_r = self._predict_noise(imu_norm)  # (T, 2) [R_lat, R_up]

        x0, k_start = self._init_state(
            gps_enu_x, gps_enu_y, gps_valid, v_ms, gps_theta, gyro_z_rad, net_bias,
            dt=dt, time_s=time_s)
        ekf = EKF6D(cfg=self.ekf_config, x0=x0)

        enu_x = np.full(T, np.nan, np.float32)
        enu_y = np.full(T, np.nan, np.float32)
        headings = np.zeros(T, np.float32)
        vel_x = np.zeros(T, np.float32)
        vel_y = np.zeros(T, np.float32)
        ekf_bg = np.zeros(T, np.float32)

        cfg = self.ekf_config
        for k in range(k_start, T):
            if time_s is not None and k > 0:
                dt_k = float(time_s[k] - time_s[k - 1])
                dt_k = float(np.clip(dt_k, cfg.dt_min, cfg.dt_max))
            else:
                dt_k = float(dt)

            gz = float(gyro_z_rad[k]) if np.isfinite(gyro_z_rad[k]) else 0.0
            bn = float(net_bias[k]) if np.isfinite(net_bias[k]) else 0.0
            vw = float(v_ms[k]) if np.isfinite(v_ms[k]) else 0.0

            still = vw < cfg.freeze_yaw_below_ms
            ekf.predict(gz, bn, dt_k, freeze_yaw=still)
            if still:
                ekf.x[IDX_VX] = 0.0
                ekf.x[IDX_VY] = 0.0

            if (gps_valid[k] and np.isfinite(gps_enu_x[k])
                    and np.isfinite(gps_enu_y[k])):
                ekf.update_gnss_position(float(gps_enu_x[k]), float(gps_enu_y[k]))

            if enable_wheel_meas and vw >= cfg.min_speed_wheel_ms:
                r_wheel_dyn = float(noise_r[k, 1])  # 动态前向轮速噪声
                ekf.update_wheel_forward(vw, r_wheel=r_wheel_dyn)

            if enable_nhc and vw >= cfg.min_speed_nhc_ms:
                yaw_rate = gz - bn - ekf.bg
                r_nhc_dyn = float(noise_r[k, 0])  # 动态横向速度噪声
                ekf.update_nhc(yaw_rate=yaw_rate, r_nhc=r_nhc_dyn)

            enu_x[k] = ekf.px
            enu_y[k] = ekf.py
            headings[k] = ekf.yaw
            vel_x[k] = ekf.vx
            vel_y[k] = ekf.vy
            ekf_bg[k] = ekf.bg

        # 初始化前帧用首帧状态填充
        if k_start > 0:
            headings[:k_start] = headings[k_start]
            vel_x[:k_start] = vel_x[k_start]
            vel_y[:k_start] = vel_y[k_start]
            ekf_bg[:k_start] = ekf_bg[k_start]
            enu_x[:k_start] = enu_x[k_start]
            enu_y[:k_start] = enu_y[k_start]

        return enu_x, enu_y, headings, net_bias, vel_x, vel_y, ekf_bg


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

    return metrics
