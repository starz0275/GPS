"""
config.py — 二维 GNSS/INS/Wheel EKF 融合导航可调参数
"""

from dataclasses import dataclass


@dataclass
class EKFConfig:
    """6 状态 EKF: [px, py, vx, vy, yaw, bg]"""

    # ---- 过程噪声（每步）----
    q_yaw: float = 2e-5              # 航向传播 rad²/step
    q_vel: float = 0.05 ** 2         # ENU 速度随机游走 (m/s)²/step（非轮速覆盖）
    q_bg: float = 1e-8               # 残余陀螺零偏随机游走 (rad/s)²/step（小值防 outage 下漂移）
    q_pos: float = 1e-6              # 位置附加（通常由 vx,vy 传播主导）

    # ---- 量测噪声 ----
    r_gps_xy: float = 2.0 ** 2       # GNSS 平面位置 (m)² / 轴
    r_wheel: float = 0.12 ** 2       # 前向轮速 (m/s)²
    r_nhc: float = 0.08 ** 2         # 横向速度伪量测 (m/s)²

    # ---- 动态 NHC（转弯削弱约束）----
    nhc_yaw_rate_thresh: float = 0.12   # rad/s
    nhc_r_scale_turn: float = 50.0

    # ---- BiasNet 推理限幅 ----
    biasnet_max_deg: float = 2.0     # |bias| <= 2 deg/s via tanh

    # ---- CovAdapterNet（AI 噪声适配器）----
    beta_noise_scale: float = 3.0    # 噪声动态范围 10^(±β) = 0.001x ~ 1000x

    # ---- GNSS 航向量测（备选，当前未启用）----
    r_gps_heading_rad2: float = (3.0 * 3.14159 / 180.0) ** 2

    # ---- 初始零偏估计（用 GPS 位移推算）----
    bg_init_window_s: float = 10.0           # 估计窗口（秒）
    bg_init_min_speed_ms: float = 8.0 / 3.6  # 最低速度 ≈ 2.2 m/s
    bg_init_max_bg_rads: float = 5.0 * 3.14159 / 180.0  # |bg| 截断 ≤ 5°/s

    # ---- 量测门控 ----
    min_speed_wheel_ms: float = 0.5  # 低于此速度不做轮速更新 (~1.8 km/h)
    min_speed_nhc_ms: float = 0.5
    min_speed_ms: float = 0.3       # 轮速 Jacobian 防除零
    freeze_yaw_below_ms: float = 0.5 # 低于此速度冻结航向（不积分陀螺），并 ZUPT 速度

    # ---- 数值 ----
    p_init_pos: float = 10.0 ** 2
    p_init_vel: float = 1.0 ** 2
    p_init_yaw: float = 0.3 ** 2
    p_init_bg: float = 0.001 ** 2
    dt_min: float = 0.02
    dt_max: float = 0.25
    eps: float = 1e-8


DEFAULT_EKF_CONFIG = EKFConfig()
