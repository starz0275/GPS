"""
train_biasnet_cmcc.py — Data0109 CMCC 多阶段微调
==============================================
阶段1: 整段 cmcc_ok
阶段2: cmcc_ok 后再等 settle_s（默认60s）才训练（六轴等权）
阶段3: 稳定段 + 段内末 60s acc 均值标签 + acc 强化损失（专治 ba 贴 0）

最终权重: trained_models/biasnet_weights_cmcc.weights.h5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy.ndimage import median_filter
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from data0109_loader import (
    CMCC_SETTLE_S,
    CMCC_STABLE_TAIL_FRAC,
    DATA0109_TRAIN_SEGMENTS,
    DATA0109_VAL_SEGMENT,
    cmcc_stable_mask,
    load_data0109_segments,
)
from ekf_navigator import BiasNet, make_biasnet
from train_ekf import (
    BATCH_SIZE,
    TARGET_DT,
    WINDOW_SIZE as SHALLOW_WINDOW_SIZE,
    WEIGHTS_PATH,
    load_or_compute_norm,
    normalize_imu,
)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "trained_models"
NORM_JSON_0109 = ROOT / "preprocessed_data" / "normalization_stats_data0109.json"
STAGE1_WEIGHTS = MODEL_DIR / "biasnet_weights_cmcc_s1.weights.h5"
FINAL_WEIGHTS = MODEL_DIR / "biasnet_weights_cmcc.weights.h5"
FINAL_INFO = MODEL_DIR / "biasnet_info_cmcc.json"

CH_NAMES = ["ba_x[g]", "ba_y[g]", "ba_z[g]", "bg_x[d/s]", "bg_y[d/s]", "bg_z[d/s]"]
LABEL_SMOOTH_SIZE = 11
ACC_TAIL_S = 60.0
CMCC_MAX_ACC_G_TRAIN = 0.25
CMCC_MAX_GYRO_DEG_TRAIN = 2.0
ACC_CALIB_JSON = MODEL_DIR / "biasnet_cmcc_acc_calib.json"

# DeepBiasNet 默认窗口（20s @ 10Hz），shallow 仍 30 帧
DEFAULT_ARCH = "deep"
DEFAULT_WINDOW_DEEP = 300

# 训练配置（优先提升陀螺 X/Y 泛化）
LOSS_ACC_WEIGHTS = tf.constant([0.8, 0.8, 0.6], dtype=tf.float32)
LOSS_GYRO_WEIGHTS = tf.constant([2.0, 2.25, 1.9], dtype=tf.float32)
STAGE3_ACC_WEIGHTS = tf.constant([8.0, 8.0, 6.0], dtype=tf.float32)
STAGE3_GYRO_WEIGHTS = tf.constant([2.0, 2.5, 2.5], dtype=tf.float32)
STAGE1_LR_DEEP = 3e-4
STAGE2_LR_DEEP = 5e-5
STAGE3_ACC_SAMPLE_DUP = 1
TEMP_CONSISTENCY_LAMBDA = 0.008
TEMP_CONSISTENCY_GYRO_XY_WEIGHTS = tf.constant([1.0, 1.0], dtype=tf.float32)
TEMP_CONSISTENCY_MAX_LABEL_DELTA = 0.03


def cmcc_huber_loss_equal(y_true, y_pred):
    """六轴加权 Huber（提高陀螺通道权重）。"""
    weights = tf.concat([LOSS_ACC_WEIGHTS, LOSS_GYRO_WEIGHTS], axis=0)
    delta = 1.0
    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = 0.5 * tf.square(error)
    linear = delta * (abs_error - 0.5 * delta)
    per_ch = tf.where(abs_error <= delta, quadratic, linear)
    return tf.reduce_mean(per_ch * weights)


def cmcc_huber_with_temporal_consistency(y_true, y_pred):
    """
    六轴加权 Huber + 时间一致性损失。
    时间一致性仅作用于陀螺 X/Y 两轴，约束相邻样本预测变化率，抑制抖动。
    当相邻标签差分过大（通常是跨段或突变）时跳过该对样本，避免过度平滑。
    """
    base = cmcc_huber_loss_equal(y_true, y_pred)
    n = tf.shape(y_pred)[0]

    def _tc_term():
        pred_diff = y_pred[1:, 3:5] - y_pred[:-1, 3:5]  # 仅 bg_x/bg_y
        label_diff = tf.abs(y_true[1:, 3:5] - y_true[:-1, 3:5])
        pair_keep = tf.reduce_all(label_diff <= TEMP_CONSISTENCY_MAX_LABEL_DELTA, axis=1)
        pair_keep_f = tf.cast(pair_keep, tf.float32)
        weighted = tf.reduce_sum(
            tf.square(pred_diff) * TEMP_CONSISTENCY_GYRO_XY_WEIGHTS, axis=1
        ) / tf.reduce_sum(TEMP_CONSISTENCY_GYRO_XY_WEIGHTS)
        num = tf.reduce_sum(weighted * pair_keep_f)
        den = tf.maximum(tf.reduce_sum(pair_keep_f), 1.0)
        return num / den

    tc = tf.cond(tf.less(n, 2), lambda: tf.constant(0.0, dtype=tf.float32), _tc_term)
    return base + TEMP_CONSISTENCY_LAMBDA * tc


def _phys_from_raw(y_pred_raw):
    """与推理一致：acc/gyro 经 tanh 限幅。"""
    acc = CMCC_MAX_ACC_G_TRAIN * tf.tanh(y_pred_raw[:, 0:3])
    gyro = CMCC_MAX_GYRO_DEG_TRAIN * tf.tanh(y_pred_raw[:, 3:6])
    return tf.concat([acc, gyro], axis=1)


def cmcc_acc_inference_aligned_loss(y_true, y_pred_raw):
    """损失在物理量上计算（含 tanh），与 evaluate 推理一致。"""
    y_hat = _phys_from_raw(y_pred_raw)
    acc_w = STAGE3_ACC_WEIGHTS
    gyro_w = STAGE3_GYRO_WEIGHTS
    acc_err = y_true[:, 0:3] - y_hat[:, 0:3]
    acc_loss = tf.reduce_mean(tf.square(acc_err) * acc_w)
    g_err = y_true[:, 3:6] - y_hat[:, 3:6]
    abs_g = tf.abs(g_err)
    huber_g = tf.where(abs_g <= 1.0, 0.5 * tf.square(g_err), abs_g - 0.5)
    gyro_loss = tf.reduce_mean(huber_g * gyro_w)
    return acc_loss + gyro_loss


def cmcc_acc_focus_loss(y_true, y_pred):
    """线性输出上的 acc MSE（备用）。"""
    acc_w = STAGE3_ACC_WEIGHTS
    gyro_w = STAGE3_GYRO_WEIGHTS
    acc_err = y_true[:, 0:3] - y_pred[:, 0:3]
    acc_loss = tf.reduce_mean(tf.square(acc_err) * acc_w)
    g_err = y_true[:, 3:6] - y_pred[:, 3:6]
    abs_g = tf.abs(g_err)
    huber_g = tf.where(abs_g <= 1.0, 0.5 * tf.square(g_err), abs_g - 0.5)
    gyro_loss = tf.reduce_mean(huber_g * gyro_w)
    return acc_loss + gyro_loss


def smooth_cmcc_labels(
    bias_6d: np.ndarray, mask: np.ndarray, size: int = LABEL_SMOOTH_SIZE
) -> np.ndarray:
    out = bias_6d.copy()
    if mask.sum() < size:
        return out
    win = min(size, max(3, int(mask.sum()) // 20 * 2 + 1))
    for c in range(6):
        sm = median_filter(out[:, c], size=win)
        out[:, c] = np.where(mask, sm, out[:, c])
    return out


def stable_tail_acc_mean(seq: dict, tail_s: float = ACC_TAIL_S) -> np.ndarray:
    """段内 cmcc_stable 末 tail_s 秒的 acc 均值 [g]。"""
    stable = seq["cmcc_stable"]
    bias = seq["cmcc_bias_6d"]
    idx = np.where(stable)[0]
    if len(idx) < 5:
        return np.zeros(3, dtype=np.float32)
    n_tail = max(1, int(tail_s / TARGET_DT))
    tail_idx = idx[-n_tail:]
    return bias[tail_idx, 0:3].mean(axis=0).astype(np.float32)


def build_cmcc_samples(
    seqs,
    mu,
    std,
    window_size: int = DEFAULT_WINDOW_DEEP,
    use_stable: bool = False,
    label_smooth: bool = False,
    require_full_window: bool = False,
    acc_constant_tail: bool = False,
    acc_sample_dup: int = 1,
):
    """
    acc_constant_tail: 稳定段样本的 ba_x/y/z 标签改为该段末 60s CMCC 均值（常值偏置）。
    acc_sample_dup: 每个 acc 样本重复次数（强化 acc 梯度）。
    """
    X_list, Y_list = [], []
    for seq in seqs:
        imu_norm = normalize_imu(seq["imu"], mu, std)
        mask = seq["cmcc_stable"] if use_stable else seq["cmcc_ok"]
        y_cmcc = seq["cmcc_bias_6d"]
        if label_smooth:
            y_cmcc = smooth_cmcc_labels(y_cmcc, mask)
        acc_const = stable_tail_acc_mean(seq) if acc_constant_tail else None
        T = len(imu_norm)
        for i in range(T - window_size + 1):
            end = i + window_size - 1
            win_ok = mask[i: end + 1].all() if require_full_window else mask[end]
            if win_ok:
                y = y_cmcc[end].copy()
                if acc_const is not None:
                    y[0:3] = acc_const
                for _ in range(acc_sample_dup):
                    X_list.append(imu_norm[i: i + window_size])
                    Y_list.append(y)
    if not X_list:
        return (
            np.empty((0, window_size, 7), dtype=np.float32),
            np.empty((0, 6), dtype=np.float32),
        )
    return np.stack(X_list, axis=0).astype(np.float32), np.stack(Y_list, axis=0).astype(
        np.float32
    )


def _prepare_seqs(settle_s: float, stable_tail_frac: float | None):
    print(
        f"\n[加载] 训练 {len(DATA0109_TRAIN_SEGMENTS)} 段 + 验证 {DATA0109_VAL_SEGMENT} "
        f"(cmcc_ok 后 settle={settle_s:.0f}s 才训练) ..."
    )
    seqs_tr = load_data0109_segments(DATA0109_TRAIN_SEGMENTS)
    seqs_val = load_data0109_segments([DATA0109_VAL_SEGMENT])
    if not seqs_tr or not seqs_val:
        raise RuntimeError("数据加载失败")
    for seq in seqs_tr + seqs_val:
        seq["cmcc_stable"] = cmcc_stable_mask(
            seq["cmcc_ok"],
            settle_s=settle_s,
            tail_frac=stable_tail_frac,
        )
    mu, std, stats = load_or_compute_norm(seqs_tr + seqs_val, NORM_JSON_0109)
    return seqs_tr, seqs_val, mu, std, stats


def _fit_stage(
    model: BiasNet,
    X_tr,
    Y_tr,
    X_val,
    Y_val,
    out_weights: Path,
    lr: float,
    epochs: int,
    patience: int,
    stage_name: str,
    loss_fn=cmcc_huber_loss_equal,
    shuffle: bool = True,
):
    print(f"\n--- {stage_name} ---")
    print(f"  训练样本: {len(X_tr)}  验证: {len(X_val)}  LR={lr}  epochs={epochs}")
    for c, name in enumerate(CH_NAMES):
        print(f"  {name}: train={Y_tr[:, c].mean():+.5f}  val={Y_val[:, c].mean():+.5f}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss_fn,
        metrics=["mae"],
    )
    callbacks = [
        ModelCheckpoint(
            str(out_weights),
            save_weights_only=True,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=5e-7, verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]
    history = model.fit(
        X_tr,
        Y_tr,
        validation_data=(X_val, Y_val),
        batch_size=BATCH_SIZE,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
        shuffle=shuffle,
    )
    best = float(min(history.history["val_mae"]))
    print(f"  [{stage_name}] 最优 val_mae={best:.5f}  -> {out_weights.name}")
    return best, history


def predict_bias_numpy(model, imu_norm, max_acc=CMCC_MAX_ACC_G_TRAIN, max_gyr=CMCC_MAX_GYRO_DEG_TRAIN):
    """与 evaluate_bias_cmcc 相同的滑窗推理（无校准）。"""
    from ekf_navigator import clip_biasnet_6d
    T = len(imu_norm)
    W = int(getattr(model, "window_size", DEFAULT_WINDOW_DEEP))
    bias = np.zeros((T, 6), dtype=np.float32)
    if T < W:
        return bias
    windows = np.stack([imu_norm[i: i + W] for i in range(T - W + 1)], axis=0)
    raw = model(windows.astype(np.float32), training=False).numpy()
    phys = clip_biasnet_6d(raw, max_acc, max_gyr)
    bias[W - 1:] = phys.astype(np.float32)
    return bias


def fit_acc_calib(model, seqs_tr, mu, std) -> np.ndarray:
    """在训练段 cmcc_stable 上估计 acc 系统性偏差，pred += offset。"""
    from biasnet_postprocess import postprocess_bias_6d

    errs = []
    for seq in seqs_tr:
        pred = predict_bias_numpy(model, normalize_imu(seq["imu"], mu, std))
        ok = seq["cmcc_stable"]
        pred = postprocess_bias_6d(
            pred, ok, mask_outside=True, use_nan_outside=False, smooth_s=0.0
        )
        if ok.sum() < 10:
            continue
        errs.append(pred[ok, 0:3] - seq["cmcc_bias_6d"][ok, 0:3])
    if not errs:
        offset = np.zeros(3, dtype=np.float32)
    else:
        offset = -np.mean(np.concatenate(errs, axis=0), axis=0).astype(np.float32)
    ACC_CALIB_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(ACC_CALIB_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {"acc_offset_g": offset.tolist(), "note": "pred_acc += offset"},
            f,
            indent=2,
        )
    print(f"  acc 校准偏移 [g]: {offset}  -> {ACC_CALIB_JSON.name}")
    return offset


def run_acc_stage3(
    model,
    seqs_tr,
    seqs_val,
    mu,
    std,
    settle_s: float,
    stable_tail_frac: float | None,
) -> float:
    """阶段3：acc 段末常值标签 + MSE 强化。"""
    window_size = int(getattr(model, "window_size", DEFAULT_WINDOW_DEEP))
    X_tr, Y_tr = build_cmcc_samples(
        seqs_tr, mu, std,
        window_size=window_size,
        use_stable=True,
        label_smooth=True,
        require_full_window=True,
        acc_constant_tail=True,
        acc_sample_dup=STAGE3_ACC_SAMPLE_DUP,
    )
    X_val, Y_val = build_cmcc_samples(
        seqs_val, mu, std,
        window_size=window_size,
        use_stable=True,
        label_smooth=True,
        require_full_window=True,
        acc_constant_tail=True,
        acc_sample_dup=1,
    )
    if FINAL_WEIGHTS.exists():
        model.load_weights(str(FINAL_WEIGHTS))
    print(f"\n[阶段3] 从 {FINAL_WEIGHTS.name} 初始化（acc 末{ACC_TAIL_S:.0f}s 常值标签）")
    s3_mae, _ = _fit_stage(
        model, X_tr, Y_tr, X_val, Y_val,
        FINAL_WEIGHTS,
        lr=2e-5,
        epochs=80,
        patience=20,
        stage_name="阶段3 acc 稳态强化(tanh对齐)",
        loss_fn=cmcc_acc_inference_aligned_loss,
    )
    fit_acc_calib(model, seqs_tr, mu, std)
    return s3_mae


def run_dual_stage(
    settle_s: float = CMCC_SETTLE_S,
    stable_tail_frac: float | None = CMCC_STABLE_TAIL_FRAC,
    label_smooth_stage2: bool = True,
    skip_stage1: bool = False,
    with_acc_stage3: bool = True,
    stage1_use_ok: bool = False,
    arch: str = DEFAULT_ARCH,
    window_size: int = DEFAULT_WINDOW_DEEP,
):
    print("=" * 60)
    print(f"BiasNet CMCC 三阶段微调（arch={arch}, window={window_size}）")
    print("=" * 60)

    seqs_tr, seqs_val, mu, std, stats = _prepare_seqs(settle_s, stable_tail_frac)
    MODEL_DIR.mkdir(exist_ok=True)

    model = make_biasnet(arch, window_size)
    init = WEIGHTS_PATH if arch == "shallow" else None  # 深模型从零开始训

    n_s1_tr = n_s1_val = None
    # ---------- 阶段1: cmcc_ok 或 cmcc_stable（默认后者，跳过静止/收敛初期）----------
    if not skip_stage1:
        X1_tr, Y1_tr = build_cmcc_samples(
            seqs_tr, mu, std,
            window_size=window_size,
            use_stable=not stage1_use_ok,
            label_smooth=False,
            require_full_window=not stage1_use_ok,
        )
        X1_val, Y1_val = build_cmcc_samples(
            seqs_val, mu, std,
            window_size=window_size,
            use_stable=not stage1_use_ok,
            label_smooth=False,
            require_full_window=not stage1_use_ok,
        )
        model(X1_tr[:1])
        if init is not None and init.exists():
            model.load_weights(str(init))
            print(f"\n[阶段1] 从浅模型基线 {init.name} 初始化")
        else:
            print(f"\n[阶段1] 深模型从零开始训练")
        mask1 = "cmcc_ok" if stage1_use_ok else f"cmcc_stable(settle={settle_s:.0f}s)"
        print(f"  掩码={mask1}, 样本={len(X1_tr)}")
        n_s1_tr, n_s1_val = len(X1_tr), len(X1_val)
        s1_lr = STAGE1_LR_DEEP if arch == "deep" else 2e-5
        s1_mae, _ = _fit_stage(
            model, X1_tr, Y1_tr, X1_val, Y1_val,
            STAGE1_WEIGHTS,
            lr=s1_lr,
            epochs=60,
            patience=15,
            stage_name=f"阶段1 {mask1}",
        )
    else:
        print("\n[阶段1] 跳过，直接加载已有 s1 权重")
        s1_mae = None
        model(tf.zeros((1, window_size, 7), dtype=np.float32))
        model.load_weights(str(STAGE1_WEIGHTS))

    # ---------- 阶段2: 稳定后段 ----------
    X2_tr, Y2_tr = build_cmcc_samples(
        seqs_tr, mu, std,
        window_size=window_size,
        use_stable=True,
        label_smooth=label_smooth_stage2,
        require_full_window=True,
    )
    X2_val, Y2_val = build_cmcc_samples(
        seqs_val, mu, std,
        window_size=window_size,
        use_stable=True,
        label_smooth=label_smooth_stage2,
        require_full_window=True,
    )
    if STAGE1_WEIGHTS.exists():
        model.load_weights(str(STAGE1_WEIGHTS))
    print(f"\n[阶段2] 从 {STAGE1_WEIGHTS.name} 初始化")
    s2_lr = STAGE2_LR_DEEP if arch == "deep" else 1e-5
    s2_mae, history2 = _fit_stage(
        model, X2_tr, Y2_tr, X2_val, Y2_val,
        FINAL_WEIGHTS,
        lr=s2_lr,
        epochs=80,
        patience=15,
        stage_name=f"阶段2 cmcc_stable settle={settle_s:.0f}s",
        loss_fn=cmcc_huber_with_temporal_consistency,
        shuffle=False,
    )

    s3_mae = None
    if with_acc_stage3:
        s3_mae = run_acc_stage3(
            model, seqs_tr, seqs_val, mu, std, settle_s, stable_tail_frac
        )

    info = {
        "arch": arch,
        "window_size": int(window_size),
        "target_dt": 0.1,
        "output_dim": 6,
        "output_channels": CH_NAMES,
        "norm_stats": stats,
        "norm_json": str(NORM_JSON_0109),
        "train_segments": DATA0109_TRAIN_SEGMENTS,
        "val_segment": DATA0109_VAL_SEGMENT,
        "training": "dual_stage_plus_acc3",
        "stage1": {
            "mask": "cmcc_ok" if stage1_use_ok else "cmcc_stable",
            "settle_s": settle_s,
            "weights": str(STAGE1_WEIGHTS),
            "train_samples": n_s1_tr,
            "best_val_mae": s1_mae,
        },
        "stage2": {
            "mask": "cmcc_stable",
            "settle_s": settle_s,
            "stable_tail_frac": stable_tail_frac,
            "label_smooth": label_smooth_stage2,
            "temporal_consistency": {
                "enabled": True,
                "lambda": TEMP_CONSISTENCY_LAMBDA,
                "gyro_xy_weights": [1.0, 1.0],
                "max_label_delta": TEMP_CONSISTENCY_MAX_LABEL_DELTA,
            },
            "weights": str(FINAL_WEIGHTS),
            "train_samples": int(len(X2_tr)),
            "val_samples": int(len(X2_val)),
            "best_val_mae": s2_mae,
        },
        "stage3": {
            "acc_label": f"segment_stable_tail_{ACC_TAIL_S}s_mean",
            "acc_sample_dup": STAGE3_ACC_SAMPLE_DUP,
            "loss": "acc_mse_focus + gyro_huber",
            "best_val_mae": s3_mae,
        },
        "loss": "huber_equal_weights_all_6_channels",
        "biasnet_max_acc_g_inference": 0.25,
        "init_baseline": str(init) if init is not None else "(from scratch)",
    }
    with open(FINAL_INFO, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"\n最终权重: {FINAL_WEIGHTS}")
    print(f"阶段1权重: {STAGE1_WEIGHTS}")

    print("\n[评估] Data05 + 全段 5 条 ...")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "evaluate_bias_cmcc.py"),
            "--weights",
            str(FINAL_WEIGHTS),
            "--norm-json",
            str(NORM_JSON_0109),
        ],
        check=False,
    )
    return info


def main():
    ap = argparse.ArgumentParser(description="Data0109 CMCC 三阶段微调（DeepTCN）")
    ap.add_argument(
        "--arch",
        choices=["deep", "shallow"],
        default=DEFAULT_ARCH,
        help="deep=DeepBiasNet 扩窗 TCN（默认）；shallow=旧 BiasNet",
    )
    ap.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="窗口长度（帧 @ 10Hz）。默认 deep=200，shallow=30",
    )
    ap.add_argument(
        "--settle-s",
        type=float,
        default=CMCC_SETTLE_S,
        help="cmcc_ok 后再等待该秒数才纳入训练（默认60，可用50）",
    )
    ap.add_argument(
        "--stable-tail-frac",
        type=float,
        default=None,
        help="可选：在 settle 后再裁时间后段比例（默认不裁）",
    )
    ap.add_argument(
        "--stage1-cmcc-ok",
        action="store_true",
        help="阶段1仍用整段 cmcc_ok（含收敛初期，一般不推荐）",
    )
    ap.add_argument("--no-label-smooth", action="store_true", help="阶段2 不做标签平滑")
    ap.add_argument("--skip-stage1", action="store_true", help="跳过阶段1（需已有 s1 权重）")
    ap.add_argument(
        "--no-acc-stage3",
        action="store_true",
        help="跳过阶段3（acc 稳态强化）",
    )
    ap.add_argument(
        "--acc-only",
        action="store_true",
        help="仅跑阶段3（需已有 biasnet_weights_cmcc.weights.h5）",
    )
    ap.add_argument(
        "--stage",
        choices=["dual", "1", "2", "3"],
        default="dual",
        help="dual=1+2+3；3=仅 acc 强化",
    )
    args = ap.parse_args()

    arch = args.arch
    if args.window_size is not None:
        window_size = int(args.window_size)
    else:
        window_size = DEFAULT_WINDOW_DEEP if arch == "deep" else SHALLOW_WINDOW_SIZE

    if args.acc_only or args.stage == "3":
        seqs_tr, seqs_val, mu, std, stats = _prepare_seqs(
            args.settle_s, args.stable_tail_frac
        )
        model = make_biasnet(arch, window_size)
        model(tf.zeros((1, window_size, 7), dtype=tf.float32))
        if not FINAL_WEIGHTS.exists():
            raise FileNotFoundError(f"请先完成阶段1/2，或提供 {FINAL_WEIGHTS}")
        s3 = run_acc_stage3(
            model, seqs_tr, seqs_val, mu, std, args.settle_s, args.stable_tail_frac
        )
        print(f"\n阶段3完成 val_mae={s3:.5f}")
        subprocess.run(
            [sys.executable, str(ROOT / "evaluate_bias_cmcc.py"),
             "--weights", str(FINAL_WEIGHTS), "--norm-json", str(NORM_JSON_0109)],
            check=False,
        )
        return

    if args.stage == "dual":
        run_dual_stage(
            settle_s=args.settle_s,
            stable_tail_frac=args.stable_tail_frac,
            label_smooth_stage2=not args.no_label_smooth,
            skip_stage1=args.skip_stage1,
            with_acc_stage3=not args.no_acc_stage3,
            stage1_use_ok=args.stage1_cmcc_ok,
            arch=arch,
            window_size=window_size,
        )
    else:
        seqs_tr, seqs_val, mu, std, stats = _prepare_seqs(
            args.settle_s, args.stable_tail_frac
        )
        use_s1 = args.stage == "1"
        model = make_biasnet(arch, window_size)
        X_tr, Y_tr = build_cmcc_samples(
            seqs_tr, mu, std,
            window_size=window_size,
            use_stable=not (use_s1 and args.stage1_cmcc_ok),
            label_smooth=not use_s1 and not args.no_label_smooth,
            require_full_window=not use_s1,
        )
        X_val, Y_val = build_cmcc_samples(
            seqs_val, mu, std,
            window_size=window_size,
            use_stable=not (use_s1 and args.stage1_cmcc_ok),
            label_smooth=not use_s1 and not args.no_label_smooth,
            require_full_window=not (use_s1 and args.stage1_cmcc_ok),
        )
        out = STAGE1_WEIGHTS if use_s1 else FINAL_WEIGHTS
        model(X_tr[:1])
        if arch == "shallow" and WEIGHTS_PATH.exists():
            model.load_weights(str(WEIGHTS_PATH))
        elif (not use_s1) and STAGE1_WEIGHTS.exists():
            model.load_weights(str(STAGE1_WEIGHTS))
        lr = (5e-4 if use_s1 else 1e-4) if arch == "deep" else (2e-5 if use_s1 else 1e-5)
        _fit_stage(
            model, X_tr, Y_tr, X_val, Y_val, out,
            lr=lr,
            epochs=60 if use_s1 else 80,
            patience=15,
            stage_name=f"单阶段 {args.stage} ({arch})",
        )


if __name__ == "__main__":
    main()
