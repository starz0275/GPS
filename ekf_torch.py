"""
ekf_torch.py — PyTorch 可微 EKF + CovAdapterNet
================================================

- DifferentiableEKF  : 6 状态 EKF，所有操作可反向传播
- CovAdapterNetTorch : 与 Keras 版架构一致，输出 [z_lat, z_up]
- 训练后可通过 export_weights_to_keras() 导出为 .weights.h5 供 NumPy EKF 使用
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# ============================================================================
# 索引常量
# ============================================================================
IDX_PX, IDX_PY, IDX_VX, IDX_VY, IDX_YAW, IDX_BG = range(6)
N_STATE = 6


# ============================================================================
# CovAdapterNet (PyTorch)
# ============================================================================

class CovAdapterNetTorch(nn.Module):
    """
    与 Keras 版 CovAdapterNet 架构完全一致。
    输入: (batch, window, 6) 归一化 IMU 窗口
    输出: (batch, 2) — [z_lat, z_up]
    """
    def __init__(self, window_size=30, in_channels=6):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, 5, padding='same')
        self.conv2 = nn.Conv1d(32, 32, 5, dilation=4, padding='same')
        self.conv3 = nn.Conv1d(32, 16, 3, padding='same')
        self.fc1 = nn.Linear(16, 32)
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(32, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, W, C) → (B, C, W) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.mean(dim=-1)  # GlobalAvgPool1d
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.out(x)
        return x  # (B, 2)


# ============================================================================
# 可微 EKF
# ============================================================================

class DifferentiableEKF(nn.Module):
    """
    6 状态可微 EKF。

    过程模型:
      ψ_{k+1} = ψ_k + (ω_z - b_net - b_g) dt
      vx_{k+1} = vx cos(Δψ) - vy sin(Δψ)
      vy_{k+1} = vx sin(Δψ) + vy cos(Δψ)
      px_{k+1} = px + vx dt
      py_{k+1} = py + vy dt
      bg_{k+1} = bg

    量测:
      GNSS → [px, py]
      Wheel → v_fwd = vx cos(ψ) + vy sin(ψ)
      NHC  → v_lat = -vx sin(ψ) + vy cos(ψ) ≈ 0
    """

    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

    def _build_Q(self, cfg):
        """过程噪声矩阵"""
        return torch.diag(torch.tensor([
            cfg['q_pos'], cfg['q_pos'],
            cfg['q_vel'], cfg['q_vel'],
            cfg['q_yaw'], cfg['q_bg'],
        ], device=self.device))

    def _build_P0(self, cfg):
        """初始协方差"""
        return torch.diag(torch.tensor([
            cfg['p_init_pos'], cfg['p_init_pos'],
            cfg['p_init_vel'], cfg['p_init_vel'],
            cfg['p_init_yaw'], cfg['p_init_bg'],
        ], device=self.device))

    def _joseph_update(self, x, P, H, innov, R):
        """Joseph form: P = (I-KH)P(I-KH)^T + KRK^T"""
        S = H @ P @ H.mT + R
        eye = torch.eye(S.shape[0], device=self.device)
        K = P @ H.mT @ torch.linalg.solve(S, eye)
        x = x + K @ innov
        I_KH = torch.eye(N_STATE, device=self.device) - K @ H
        P = I_KH @ P @ I_KH.mT + K @ R @ K.mT
        P = 0.5 * (P + P.mT)
        return x, P

    def forward(self, x0, P0, gyro_z, net_bias, v_ms, gps_valid,
                gps_x, gps_y, z_lat_dyn, z_up_dyn, dt_arr, cfg,
                chunk_size=120):
        """
        整条轨迹一次性前向传播。

        z_lat_dyn : (T,)  CovAdapterNet 输出 → 控制 NHC 噪声
        z_up_dyn  : (T,)  CovAdapterNet 输出 → 控制轮速噪声

        返回: (T, 6) 状态轨迹
        """
        T = len(gyro_z)
        x = x0.clone()
        P = P0.clone()
        Q = self._build_Q(cfg)

        beta = cfg['beta_noise_scale']
        sigma_lat = np.sqrt(cfg['r_nhc'])
        sigma_up = np.sqrt(cfg['r_wheel'])

        states = []

        freeze_spd = cfg.get('freeze_yaw_below_ms', 0.5)

        for t in range(T):
            # ====== Predict ======
            if v_ms[t] < freeze_spd:
                # 零速冻结航向：不积分陀螺，避免静止零偏积累到 yaw
                dpsi = 0.0
                omega = 0.0
                c, s = 1.0, 0.0
            else:
                omega = gyro_z[t] - net_bias[t] - x[IDX_BG]
                dpsi = omega * dt_arr[t]
                c, s = torch.cos(dpsi), torch.sin(dpsi)

            # 速度旋转
            vx_rot = x[IDX_VX] * c - x[IDX_VY] * s
            vy_rot = x[IDX_VX] * s + x[IDX_VY] * c
            # 位置传播
            px_new = x[IDX_PX] + x[IDX_VX] * dt_arr[t]
            py_new = x[IDX_PY] + x[IDX_VY] * dt_arr[t]

            x = torch.stack([
                px_new, py_new,
                vx_rot, vy_rot,
                x[IDX_YAW] + dpsi,
                x[IDX_BG],
            ])

            # Jacobian F
            F = torch.eye(N_STATE, device=self.device)
            F[IDX_PX, IDX_VX] = dt_arr[t]
            F[IDX_PY, IDX_VY] = dt_arr[t]
            F[IDX_VX, IDX_VX] = c
            F[IDX_VX, IDX_VY] = -s
            F[IDX_VY, IDX_VX] = s
            F[IDX_VY, IDX_VY] = c
            # bg → velocity/heading 仅在非冻结时非零
            if v_ms[t] >= freeze_spd:
                F[IDX_VX, IDX_BG] = (x[IDX_VX].detach() * s + x[IDX_VY].detach() * c) * dt_arr[t]
                F[IDX_VY, IDX_BG] = (-x[IDX_VX].detach() * c + x[IDX_VY].detach() * s) * dt_arr[t]
                F[IDX_YAW, IDX_BG] = -dt_arr[t]

            P = F @ P @ F.mT + Q

            # ====== GNSS Update ======
            if gps_valid[t]:
                H = torch.zeros(2, N_STATE, device=self.device)
                H[0, IDX_PX] = 1.0
                H[1, IDX_PY] = 1.0
                z_pred = H @ x
                innov = torch.tensor([gps_x[t], gps_y[t]], device=self.device) - z_pred
                R_gps = torch.diag(torch.tensor([cfg['r_gps_xy']] * 2, device=self.device))
                x, P = self._joseph_update(x, P, H, innov, R_gps)

            # ====== Wheel Update + 动态噪声 ======
            if v_ms[t] >= cfg['min_speed_wheel_ms']:
                c_y, s_y = torch.cos(x[IDX_YAW]), torch.sin(x[IDX_YAW])
                v_fwd_pred = x[IDX_VX] * c_y + x[IDX_VY] * s_y
                innov = (v_ms[t] - v_fwd_pred).unsqueeze(0)

                H = torch.zeros(1, N_STATE, device=self.device)
                H[0, IDX_VX] = c_y
                H[0, IDX_VY] = s_y
                H[0, IDX_YAW] = -x[IDX_VX] * s_y + x[IDX_VY] * c_y

                # 公式 (17): σ² * 10^(β·tanh(z))
                scale_up = 10.0 ** (beta * torch.tanh(z_up_dyn[t]))
                r_up_dyn = sigma_up ** 2 * scale_up
                R_wheel = r_up_dyn.unsqueeze(0).unsqueeze(0)

                x, P = self._joseph_update(x, P, H, innov, R_wheel)

            # ====== NHC Update + 动态噪声 ======
            if v_ms[t] >= cfg['min_speed_nhc_ms']:
                c_y, s_y = torch.cos(x[IDX_YAW]), torch.sin(x[IDX_YAW])
                v_lat_pred = -x[IDX_VX] * s_y + x[IDX_VY] * c_y
                innov = (-v_lat_pred).unsqueeze(0)  # z=0

                H = torch.zeros(1, N_STATE, device=self.device)
                H[0, IDX_VX] = -s_y
                H[0, IDX_VY] = c_y
                H[0, IDX_YAW] = -x[IDX_VX] * c_y - x[IDX_VY] * s_y

                # 公式 (17): σ² * 10^(β·tanh(z))
                scale_lat = 10.0 ** (beta * torch.tanh(z_lat_dyn[t]))
                r_lat_dyn = sigma_lat ** 2 * scale_lat
                R_nhc = r_lat_dyn.unsqueeze(0).unsqueeze(0)

                x, P = self._joseph_update(x, P, H, innov, R_nhc)

            # ====== ZUPT: 零速时强制速度 → 0 ======
            if v_ms[t] < freeze_spd:
                H_zupt = torch.zeros(2, N_STATE, device=self.device)
                H_zupt[0, IDX_VX] = 1.0
                H_zupt[1, IDX_VY] = 1.0
                innov_zupt = -x[IDX_VX:IDX_VY+1]    # z=0
                R_zupt = torch.diag(torch.tensor([0.01, 0.01], device=self.device))  # 0.1 m/s std
                x, P = self._joseph_update(x, P, H_zupt, innov_zupt, R_zupt)

            # 定期截断 P 的梯度以节省显存
            if t > 0 and t % chunk_size == 0:
                P = P.detach()

            states.append(x)

        return torch.stack(states)  # (T, 6)


# ============================================================================
# 权重导出：PyTorch → Keras .weights.h5
# ============================================================================

def export_torch_to_keras(torch_model, output_path, window_size=30, in_channels=6):
    """
    将训练好的 PyTorch CovAdapterNet 权重导出为 Keras .weights.h5。

    通过构建一个 Keras 同构模型，逐层复制权重的 numpy 数组实现。
    """
    import tensorflow as tf

    # 构建 Keras 同构模型（与 ekf_navigator.CovAdapterNet 一致）
    from tensorflow.keras import layers, Model

    class KerasCovAdapter(Model):
        def __init__(self):
            super().__init__(name='CovAdapterNet')
            self.conv1 = layers.Conv1D(32, 5, padding='causal', activation='relu')
            self.conv2 = layers.Conv1D(32, 5, dilation_rate=4, padding='causal', activation='relu')
            self.conv3 = layers.Conv1D(16, 3, padding='causal', activation='relu')
            self.pool = layers.GlobalAveragePooling1D()
            self.fc1 = layers.Dense(32, activation='relu')
            self.drop = layers.Dropout(0.2)
            self.out = layers.Dense(2, activation='linear')

        def call(self, x, training=False):
            h = self.conv1(x)
            h = self.conv2(h)
            h = self.conv3(h)
            h = self.pool(h)
            h = self.fc1(h)
            h = self.drop(h, training=training)
            return self.out(h)

    keras_model = KerasCovAdapter()
    dummy = np.zeros((1, window_size, in_channels), dtype=np.float32)
    keras_model(dummy)  # build

    # PyTorch → Keras 权重映射
    pt_state = torch_model.state_dict()
    # Conv1d 权重: (out_ch, in_ch, kernel) → (kernel, in_ch, out_ch)
    # Keras Conv1D kernel: (kernel, in_ch, out_ch), bias: (out_ch,)

    keras_model.conv1.set_weights([
        pt_state['conv1.weight'].permute(2, 1, 0).cpu().numpy(),
        pt_state['conv1.bias'].cpu().numpy(),
    ])
    keras_model.conv2.set_weights([
        pt_state['conv2.weight'].permute(2, 1, 0).cpu().numpy(),
        pt_state['conv2.bias'].cpu().numpy(),
    ])
    keras_model.conv3.set_weights([
        pt_state['conv3.weight'].permute(2, 1, 0).cpu().numpy(),
        pt_state['conv3.bias'].cpu().numpy(),
    ])
    keras_model.fc1.set_weights([
        pt_state['fc1.weight'].T.cpu().numpy(),
        pt_state['fc1.bias'].cpu().numpy(),
    ])
    keras_model.out.set_weights([
        pt_state['out.weight'].T.cpu().numpy(),
        pt_state['out.bias'].cpu().numpy(),
    ])

    keras_model.save_weights(str(output_path))
    print(f"[Export] 权重已导出到 {output_path}")
    return keras_model
