"""
config.py — 二维 GNSS/INS/Wheel EKF 融合导航可调参数
"""

from dataclasses import dataclass


@dataclass
class EKFConfig:
    """12 状态 EKF: [px, py, vx, vy, yaw, bgx, bgy, bgz, bax, bay, baz, vel_scale]"""

    # ---- 过程噪声（每步）----
    q_yaw: float = 2e-5              # 航向传播 rad²/step
    q_vel: float = 0.03 ** 2         # ENU 速度随机游走 (m/s)²/step（0.03→outage下漂移更慢）
    q_bg: float = 1e-8               # 残余陀螺Z零偏随机游走 (rad/s)²/step（小值防 outage 下漂移）
    q_bg_xy: float = 1e-10           # 陀螺X/Y零偏随机游走 (rad/s)²/step（2D 非关键轴）
    q_ba: float = 1e-6               # 加速度计零偏随机游走 (m/s²)²/step
    q_pos: float = 1e-6              # 位置附加（通常由 vx,vy 传播主导）
    q_vel_scale: float = 1e-6        # 轮速比例因子随机游走（允许缓慢收敛）

    # ---- 量测噪声 ----
    r_gps_xy: float = 2.0 ** 2       # GNSS 平面位置 (m)² / 轴
    r_wheel: float = 0.12 ** 2       # 前向轮速 (m/s)²
    r_nhc: float = 0.06 ** 2         # 横向速度伪量测 (m/s)²（0.06→直道横向约束更强）
    r_accel: float = 0.3 ** 2        # 加速度计前向伪量测 (m/s²)²（用于轮速导数约束 ba_x）

    # ---- 动态 NHC（转弯削弱约束）----
    nhc_yaw_rate_thresh: float = 0.12   # rad/s
    nhc_r_scale_turn: float = 50.0

    # ---- BiasNet 推理限幅 ----
    biasnet_max_deg: float = 2.0     # |gyro_bias| <= 2 deg/s via tanh
    biasnet_max_acc_g: float = 0.1   # |acc_bias| <= 0.1g ≈ 1 m/s² via tanh

    # ---- 加速度计融合权重（0=纯协调转弯，1=全加速度计积分）----
    accel_fusion_gain: float = 0.0   # 默认关闭；提高需更好的零偏估计

    # ---- CovAdapterNet（AI 噪声适配器）----
    beta_noise_scale: float = 3.0    # 噪声动态范围 10^(±β) = 0.001x ~ 1000x

    # ---- GNSS 航向量测（备选，当前未启用）----
    r_gps_heading_rad2: float = (0.5 * 3.14159 / 180.0) ** 2  # 0.5° std，强航向约束

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
    p_init_bg: float = 0.001 ** 2     # gyro_z 零偏初始方差 (rad/s)²
    p_init_bg_xy: float = 0.0001 ** 2 # gyro_x/y 零偏初始方差 (rad/s)²
    p_init_ba: float = 0.05 ** 2      # accel 零偏初始方差 (m/s²)²
    dt_min: float = 0.02
    dt_max: float = 0.25
    eps: float = 1e-8


DEFAULT_EKF_CONFIG = EKFConfig()
