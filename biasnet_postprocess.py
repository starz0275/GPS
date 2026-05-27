"""
BiasNet 推理后处理：仅在 cmcc_stable 段输出，并平滑抑制滑窗抖动。
（已去掉中值滤波，仅保留 EMA + 步长限幅）
"""

from __future__ import annotations

import numpy as np

from train_ekf import TARGET_DT

# 默认：EMA 时间常数 2s + 每步最大变化（@10Hz）
DEFAULT_SMOOTH_S = 2.0
DEFAULT_MAX_STEP_ACC_G = 0.0015
DEFAULT_MAX_STEP_GYRO_DPS = 0.015


def _iter_stable_runs(stable: np.ndarray):
    T = len(stable)
    i = 0
    while i < T:
        if not stable[i]:
            i += 1
            continue
        j = i
        while j < T and stable[j]:
            j += 1
        yield i, j
        i = j


def _ema_1d(x: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def _rate_limit_1d(x: np.ndarray, max_step: float) -> np.ndarray:
    if len(x) < 2 or max_step <= 0:
        return x.copy()
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        d = float(np.clip(x[i] - out[i - 1], -max_step, max_step))
        out[i] = out[i - 1] + d
    return out


def smooth_bias_in_stable(
    bias: np.ndarray,
    stable: np.ndarray,
    smooth_s: float = DEFAULT_SMOOTH_S,
    max_step_acc_g: float = DEFAULT_MAX_STEP_ACC_G,
    max_step_gyro_dps: float = DEFAULT_MAX_STEP_GYRO_DPS,
) -> np.ndarray:
    """仅在 stable 连续段内做 EMA + 步长限幅（不含中值滤波）。"""
    out = bias.copy()
    if not stable.any() or smooth_s <= 0:
        return out

    alpha = TARGET_DT / (smooth_s + TARGET_DT)
    max_steps = np.array(
        [max_step_acc_g, max_step_acc_g, max_step_acc_g,
         max_step_gyro_dps, max_step_gyro_dps, max_step_gyro_dps],
        dtype=np.float32,
    )

    for i0, i1 in _iter_stable_runs(stable):
        seg = out[i0:i1].copy()
        for c in range(6):
            y = seg[:, c].astype(np.float64)
            y = _ema_1d(y, alpha).astype(np.float32)
            y = _rate_limit_1d(y, float(max_steps[c]))
            seg[:, c] = y
        out[i0:i1] = seg
    return out


def mask_bias_outside_stable(
    bias: np.ndarray,
    stable: np.ndarray,
    *,
    use_nan: bool = False,
) -> np.ndarray:
    """非 cmcc_stable 时段不输出预测（默认 0，绘图可用 NaN 断线）。"""
    out = bias.copy()
    fill = np.nan if use_nan else 0.0
    out[~stable] = fill
    return out


def postprocess_bias_6d(
    bias: np.ndarray,
    stable: np.ndarray,
    *,
    mask_outside: bool = True,
    use_nan_outside: bool = False,
    smooth_s: float = DEFAULT_SMOOTH_S,
    max_step_acc_g: float = DEFAULT_MAX_STEP_ACC_G,
    max_step_gyro_dps: float = DEFAULT_MAX_STEP_GYRO_DPS,
) -> np.ndarray:
    """
    推理后处理顺序：stable 内平滑 → 非 stable 置零/NaN。
    """
    out = bias
    if smooth_s > 0 and stable.any():
        out = smooth_bias_in_stable(
            out, stable, smooth_s, max_step_acc_g, max_step_gyro_dps
        )
    if mask_outside:
        out = mask_bias_outside_stable(out, stable, use_nan=use_nan_outside)
    return out
