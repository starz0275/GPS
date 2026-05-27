"""
evaluate_bias_cmcc.py — Data0109 CMCC 零偏对比评估
====================================================
主指标与绘图均使用 cmcc_stable（cmcc_ok 后再等 settle_s，默认 60s）。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import DEFAULT_EKF_CONFIG, EKFConfig
from data0109_loader import (
    CMCC_SETTLE_S,
    DATA0109_ALL_SEGMENTS,
    DATA0109_VAL_SEGMENT,
    cmcc_stable_mask,
    load_data0109_seq,
)
from biasnet_postprocess import (
    DEFAULT_SMOOTH_S,
    postprocess_bias_6d,
)
from ekf_navigator import BiasNet, clip_biasnet_6d, make_biasnet
from train_ekf import (
    WINDOW_SIZE as SHALLOW_WINDOW_SIZE,
    TARGET_DT,
    load_or_compute_norm,
    normalize_imu,
)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "trained_models"
NORM_JSON_0109 = ROOT / "preprocessed_data" / "normalization_stats_data0109.json"
DEFAULT_WEIGHTS = MODEL_DIR / "biasnet_weights_cmcc.weights.h5"
FALLBACK_WEIGHTS = MODEL_DIR / "biasnet_weights_cmcc_stable.weights.h5"
INFO_JSON = MODEL_DIR / "biasnet_info_cmcc.json"
LOG_DIR = ROOT / "training_logs"

CMCC_MAX_ACC_G = 0.25
CMCC_MAX_GYRO_DEG = 2.0
ACC_CALIB_JSON = MODEL_DIR / "biasnet_cmcc_acc_calib.json"

CH_NAMES = ["ba_x[g]", "ba_y[g]", "ba_z[g]", "bg_x[d/s]", "bg_y[d/s]", "bg_z[d/s]"]
CH_KEYS = ["ba_x_g", "ba_y_g", "ba_z_g", "bg_x_degs", "bg_y_degs", "bg_z_degs"]
TAIL_S = 60.0


def cmcc_ekf_config() -> EKFConfig:
    return replace(
        DEFAULT_EKF_CONFIG,
        biasnet_max_acc_g=CMCC_MAX_ACC_G,
        biasnet_max_deg=CMCC_MAX_GYRO_DEG,
    )


def resolve_arch_and_window(arch: str | None, window_size: int | None) -> tuple[str, int]:
    """优先用 CLI 参数；否则读 info JSON；最后回退 deep+200。"""
    if arch and window_size:
        return arch, int(window_size)
    if INFO_JSON.exists():
        try:
            with open(INFO_JSON, encoding="utf-8") as f:
                info = json.load(f)
            arch = arch or info.get("arch", "deep")
            window_size = int(window_size or info.get("window_size", 200))
            return arch, window_size
        except Exception:
            pass
    return (arch or "deep"), int(window_size or 200)


def load_biasnet(
    weights_path: Path,
    arch: str | None = None,
    window_size: int | None = None,
):
    arch_used, w_used = resolve_arch_and_window(arch, window_size)
    model = make_biasnet(arch_used, w_used)
    model(tf.zeros((1, w_used, 7), dtype=tf.float32))
    model.load_weights(str(weights_path))
    print(f"[BiasNet] arch={arch_used}, window={w_used} 已加载: {weights_path}")
    return model


def load_acc_calib_offset() -> np.ndarray:
    if not ACC_CALIB_JSON.exists():
        return np.zeros(3, dtype=np.float32)
    with open(ACC_CALIB_JSON, encoding="utf-8") as f:
        data = json.load(f)
    return np.array(data.get("acc_offset_g", [0, 0, 0]), dtype=np.float32)


def predict_bias_6d_raw(
    model,
    imu_norm: np.ndarray,
    cfg: EKFConfig | None = None,
    acc_calib: np.ndarray | None = None,
) -> np.ndarray:
    """滑窗推理 + tanh 限幅 + acc 校准（未做 stable 掩码/平滑）。"""
    cfg = cfg or cmcc_ekf_config()
    if acc_calib is None:
        acc_calib = load_acc_calib_offset()
    T = len(imu_norm)
    W = int(getattr(model, "window_size", SHALLOW_WINDOW_SIZE))
    bias = np.zeros((T, 6), dtype=np.float32)
    if T < W:
        return bias
    windows = np.stack([imu_norm[i: i + W] for i in range(T - W + 1)], axis=0)
    raw = model(windows.astype(np.float32), training=False).numpy()
    bias_phys = clip_biasnet_6d(raw, cfg.biasnet_max_acc_g, cfg.biasnet_max_deg)
    bias_phys[:, 0:3] += acc_calib.reshape(1, 3)
    bias[W - 1:] = bias_phys.astype(np.float32)
    return bias


def predict_bias_6d(
    model,
    imu_norm: np.ndarray,
    cfg: EKFConfig | None = None,
    acc_calib: np.ndarray | None = None,
    stable_mask: np.ndarray | None = None,
    *,
    mask_outside: bool = True,
    smooth_s: float = DEFAULT_SMOOTH_S,
    plot_nan_outside: bool = True,
) -> np.ndarray:
    """完整推理：raw → 仅 cmcc_stable 段平滑输出，段外不预测。"""
    bias = predict_bias_6d_raw(model, imu_norm, cfg, acc_calib)
    if stable_mask is None or not mask_outside:
        return bias
    return postprocess_bias_6d(
        bias,
        stable_mask,
        mask_outside=True,
        use_nan_outside=plot_nan_outside,
        smooth_s=smooth_s,
    )


def _valid_metric_mask(mask: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """掩码内且预测有效（段外 NaN 不参与统计）。"""
    return mask & np.isfinite(y_pred)


def channel_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> dict:
    m = _valid_metric_mask(mask, y_pred)
    if m.sum() < 2:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "mean_err": float("nan")}
    t = y_true[m]
    p = y_pred[m]
    err = p - t
    return {
        "n": int(m.sum()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mean_err": float(np.mean(err)),
    }


def metrics_block(y_cmcc: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> dict:
    per_ch = {key: channel_metrics(y_cmcc[:, c], y_pred[:, c], mask) for c, key in enumerate(CH_KEYS)}
    tail = {
        key: tail_mean_metrics(y_cmcc[:, c], y_pred[:, c], mask)
        for c, key in enumerate(CH_KEYS)
    }
    return {"per_channel": per_ch, "tail_60s": tail}


def tail_mean_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray, tail_s: float = TAIL_S
) -> dict:
    n_tail = max(1, int(tail_s / TARGET_DT))
    idx = np.where(_valid_metric_mask(mask, y_pred))[0]
    if len(idx) < 2:
        return {"n": 0, "cmcc_mean": None, "pred_mean": None, "mean_err": float("nan")}
    tail_idx = idx[-n_tail:]
    cmcc_m = float(np.mean(y_true[tail_idx]))
    pred_m = float(np.mean(y_pred[tail_idx]))
    return {
        "n": int(len(tail_idx)),
        "cmcc_mean": cmcc_m,
        "pred_mean": pred_m,
        "mean_err": pred_m - cmcc_m,
    }


def evaluate_segment(
    seq: dict,
    model,
    mu: np.ndarray,
    std: np.ndarray,
    cfg: EKFConfig | None = None,
    settle_s: float = CMCC_SETTLE_S,
    smooth_s: float = DEFAULT_SMOOTH_S,
    mask_outside: bool = True,
) -> dict:
    cfg = cfg or cmcc_ekf_config()
    imu_norm = normalize_imu(seq["imu"], mu, std)
    ok = seq["cmcc_ok"]
    stable = seq.get("cmcc_stable", ok)
    y_pred = predict_bias_6d(
        model,
        imu_norm,
        cfg,
        stable_mask=stable,
        mask_outside=mask_outside,
        smooth_s=smooth_s,
        plot_nan_outside=True,
    )
    y_cmcc = seq["cmcc_bias_6d"]

    return {
        "segment": seq["segment"],
        "id": seq["id"],
        "n_frames": int(len(seq["Time_s"])),
        "cmcc_ok_ratio": float(ok.mean()),
        "cmcc_stable_ratio": float(stable.mean()),
        "cmcc_ok": metrics_block(y_cmcc, y_pred, ok),
        "cmcc_stable": metrics_block(y_cmcc, y_pred, stable),
        "Time_s": seq["Time_s"],
        "y_cmcc": y_cmcc,
        "y_pred": y_pred,
        "mask_ok": ok,
        "mask_stable": stable,
        "settle_s": settle_s,
        "smooth_s": smooth_s,
        "mask_outside": mask_outside,
    }


def plot_comparison(result: dict, out_path: Path) -> None:
    t = result["Time_s"]
    y_cmcc = result["y_cmcc"]
    y_pred = result["y_pred"]
    shade = result["mask_stable"]
    seg = result["segment"]
    s = result.get("settle_s", CMCC_SETTLE_S)
    tag = f"cmcc_stable(settle={s:.0f}s)"

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(f"CMCC vs BiasNet — {seg} ({tag})", fontsize=12)

    for c, ax in enumerate(axes.flat):
        ax.plot(t, y_cmcc[:, c], "k-", alpha=0.5, linewidth=0.8, label="CMCC")
        ax.plot(t, y_pred[:, c], "b-", alpha=0.7, linewidth=0.8, label="Pred(stable)")
        if shade.any():
            ymin = min(y_cmcc[:, c].min(), y_pred[:, c].min())
            ymax = max(y_cmcc[:, c].max(), y_pred[:, c].max())
            ax.fill_between(t, ymin, ymax, where=shade, alpha=0.10, color="green", label=tag)
        ax.set_ylabel(CH_NAMES[c])
        ax.grid(True, alpha=0.3)
        if c >= 4:
            ax.set_xlabel("Time [s]")
        if c == 0:
            ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存: {out_path}")


def _print_metrics(title: str, block: dict) -> None:
    print(f"  [{title}]")
    for c, key in enumerate(CH_KEYS):
        m = block["per_channel"][key]
        tail = block["tail_60s"][key]
        te = tail.get("mean_err", float("nan"))
        print(
            f"    {CH_NAMES[c]}: MAE={m['mae']:.5f}  RMSE={m['rmse']:.5f}  "
            f"末60s均值差={te:+.5f}"
        )


def run_evaluation(
    weights_path: Path,
    norm_json: Path,
    segments: list[str] | None = None,
    save_plots: bool = True,
    settle_s: float = CMCC_SETTLE_S,
    smooth_s: float = DEFAULT_SMOOTH_S,
    mask_outside: bool = True,
    arch: str | None = None,
    window_size: int | None = None,
) -> dict:
    segments = segments or DATA0109_ALL_SEGMENTS
    if not weights_path.exists() and FALLBACK_WEIGHTS.exists():
        weights_path = FALLBACK_WEIGHTS
        print(f"[提示] 使用备用权重: {weights_path}")

    model = load_biasnet(weights_path, arch=arch, window_size=window_size)
    cfg = cmcc_ekf_config()

    seqs = []
    for name in segments:
        seq = load_data0109_seq(name)
        if seq is not None:
            seq["cmcc_stable"] = cmcc_stable_mask(seq["cmcc_ok"], settle_s=settle_s)
            seqs.append(seq)
    if not seqs:
        raise RuntimeError("未加载任何 Data0109 段")

    mu, std, _ = load_or_compute_norm(seqs, norm_json)

    all_results = []
    report_key = "cmcc_stable"
    print("\n" + "=" * 60)
    print(f"CMCC 零偏评估（{report_key}，cmcc_ok 后 settle={settle_s:.0f}s）")
    if mask_outside:
        print(f"  推理: 仅 cmcc_stable 输出 (settle={settle_s:.0f}s)，段外 NaN")
    if smooth_s > 0:
        print(f"  平滑: EMA ~{smooth_s:.1f}s + 步长限幅")
    print("=" * 60)

    for seq in seqs:
        res = evaluate_segment(
            seq, model, mu, std, cfg,
            settle_s=settle_s,
            smooth_s=smooth_s,
            mask_outside=mask_outside,
        )
        plot_payload = {
            "segment": res["segment"],
            "id": res["id"],
            "Time_s": res["Time_s"],
            "y_cmcc": res["y_cmcc"],
            "y_pred": res["y_pred"],
            "mask_stable": res["mask_stable"],
            "settle_s": settle_s,
        }
        if save_plots:
            safe = seq["segment"].replace("+", "_").replace("圈", "q")
            plot_path = MODEL_DIR / f"cmcc_bias_compare_{safe}.png"
            plot_comparison(plot_payload, plot_path)

        print(
            f"\n[{res['segment']}] "
            f"cmcc_ok={res['cmcc_ok_ratio']:.1%}  cmcc_stable={res['cmcc_stable_ratio']:.1%}"
        )
        _print_metrics(report_key, res[report_key])

        log_entry = {
            "segment": res["segment"],
            "id": res["id"],
            "n_frames": res["n_frames"],
            "cmcc_ok_ratio": res["cmcc_ok_ratio"],
            "cmcc_stable_ratio": res["cmcc_stable_ratio"],
            "metrics": res["cmcc_stable"],
        }
        all_results.append(log_entry)

    val_res = next(
        (r for r in all_results if DATA0109_VAL_SEGMENT in r["segment"]), None
    )
    val_block = val_res["metrics"] if val_res else {}
    summary = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "weights": str(weights_path),
        "norm_json": str(norm_json),
        "report_mask": report_key,
        "cmcc_max_acc_g": CMCC_MAX_ACC_G,
        "arch": getattr(model, "name", "BiasNet"),
        "window_size": int(getattr(model, "window_size", SHALLOW_WINDOW_SIZE)),
        "settle_s": settle_s,
        "smooth_s": smooth_s,
        "mask_outside": mask_outside,
        "segments": [r["segment"] for r in all_results],
        "results": all_results,
        "val_segment": DATA0109_VAL_SEGMENT,
        "val_bg_z_mae": val_block.get("per_channel", {}).get("bg_z_degs", {}).get("mae"),
        "val_ba_x_mae": val_block.get("per_channel", {}).get("ba_x_g", {}).get("mae"),
    }
    return summary


def save_log(summary: dict) -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"cmcc_eval_{summary['timestamp']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[日志] {path}")
    return path


def main():
    ap = argparse.ArgumentParser(description="Data0109 CMCC 零偏评估")
    w_default = DEFAULT_WEIGHTS if DEFAULT_WEIGHTS.exists() else FALLBACK_WEIGHTS
    ap.add_argument("--weights", type=Path, default=w_default, help="BiasNet 权重")
    ap.add_argument("--norm-json", type=Path, default=NORM_JSON_0109, help="归一化统计 JSON")
    ap.add_argument(
        "--segment",
        action="append",
        dest="segments",
        help="仅评估指定段",
    )
    ap.add_argument("--no-plots", action="store_true", help="不保存对比图")
    ap.add_argument(
        "--settle-s",
        type=float,
        default=CMCC_SETTLE_S,
        help="cmcc_ok 后再等待该秒数才计为 stable（默认60）",
    )
    ap.add_argument(
        "--smooth-s",
        type=float,
        default=DEFAULT_SMOOTH_S,
        help="stable 段内预测平滑时间常数 [s]，0=关闭",
    )
    ap.add_argument(
        "--no-infer-mask",
        action="store_true",
        help="段外也输出预测（旧行为，不推荐）",
    )
    ap.add_argument(
        "--no-smooth",
        action="store_true",
        help="关闭 stable 段内平滑",
    )
    ap.add_argument(
        "--arch",
        choices=["deep", "shallow"],
        default=None,
        help="网络架构（不指定则从 biasnet_info_cmcc.json 自动读取）",
    )
    ap.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="窗口长度（不指定则从 biasnet_info_cmcc.json 自动读取）",
    )
    args = ap.parse_args()

    summary = run_evaluation(
        args.weights,
        args.norm_json,
        segments=args.segments,
        save_plots=not args.no_plots,
        settle_s=args.settle_s,
        smooth_s=0.0 if args.no_smooth else args.smooth_s,
        mask_outside=not args.no_infer_mask,
        arch=args.arch,
        window_size=args.window_size,
    )
    save_log(summary)

    if summary.get("val_bg_z_mae") is not None:
        print(f"\n[Data05] bg_z MAE ({summary['report_mask']}) = {summary['val_bg_z_mae']:.5f} deg/s")
    if summary.get("val_ba_x_mae") is not None:
        print(f"[Data05] ba_x MAE ({summary['report_mask']}) = {summary['val_ba_x_mae']:.5f} g")


if __name__ == "__main__":
    main()
