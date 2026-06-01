#!/usr/bin/env python3
"""评估 BiasAngleNet v3：全局零偏、全局安装角、fixed-only 与 residual 轨迹对比。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data0109_loader import DATA0109_ALL_SEGMENTS, load_data0109_segments
from scripts.train_bias_angle import (
    DEG2RAD,
    RAD2DEG,
    TARGET_DT,
    BiasAngleNet,
    predict_trajectory,
    rotation_imu_to_body,
    wrap_angle_np,
)
from train_ekf import load_or_compute_norm, normalize_imu

MODEL_DIR = ROOT / "trained_models"
NORM_JSON_0109 = ROOT / "preprocessed_data" / "normalization_stats_data0109.json"


def build_residual_series(model, imu_norm, window_size):
    T = len(imu_norm)
    residual = np.zeros((T, 6), dtype=np.float32)
    if T < window_size:
        return residual

    windows = np.stack(
        [imu_norm[i:i + window_size] for i in range(T - window_size + 1)]
    ).astype(np.float32)
    _, raw_residual, _ = model(windows, training=False)
    residual[window_size - 1:] = raw_residual.numpy()
    residual[:window_size - 1] = residual[window_size - 1]
    return residual


def rotate_np(v, install_rpy):
    v_tf = tf.constant(v[np.newaxis], dtype=tf.float32)
    rpy_tf = tf.constant(install_rpy, dtype=tf.float32)
    return rotation_imu_to_body(v_tf, rpy_tf).numpy()[0]


def integrate_np(seq, bias_series, install_rpy, mu, std):
    imu_norm = normalize_imu(seq["imu"], mu, std)
    acc_raw = imu_norm[:, :3] * std[:3] + mu[:3]
    gyro_raw = imu_norm[:, 3:6] * std[3:6] + mu[3:6]
    acc_corr = acc_raw - bias_series[:, :3]
    gyro_corr = gyro_raw - bias_series[:, 3:6]

    acc_body = rotate_np(acc_corr, install_rpy)
    gyro_body = rotate_np(gyro_corr, install_rpy)
    gyro_z = gyro_body[:, 2]

    T = len(imu_norm)
    pred_e = np.zeros(T, dtype=np.float32)
    pred_n = np.zeros(T, dtype=np.float32)
    pred_h = np.zeros(T, dtype=np.float32)
    pred_e[0] = seq["enu_x"][0]
    pred_n[0] = seq["enu_y"][0]
    pred_h[0] = seq["gps_theta"][0]

    for i in range(T - 1):
        psi = pred_h[i]
        cp, sp = np.cos(psi), np.sin(psi)
        ax_ms2 = acc_body[i, 0] * 9.80665
        ay_ms2 = acc_body[i, 1] * 9.80665
        acc_e = cp * ax_ms2 - sp * ay_ms2
        acc_n = sp * ax_ms2 + cp * ay_ms2
        v_e = seq["v_ms"][i] * cp + acc_e * TARGET_DT
        v_n = seq["v_ms"][i] * sp + acc_n * TARGET_DT
        pred_e[i + 1] = pred_e[i] + v_e * TARGET_DT
        pred_n[i + 1] = pred_n[i] + v_n * TARGET_DT
        pred_h[i + 1] = wrap_angle_np(psi + gyro_z[i] * DEG2RAD * TARGET_DT)

    return pred_e, pred_n, pred_h


def metric_stats(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean": None, "rmse": None, "p95": None, "max": None}
    return {
        "mean": float(np.mean(values)),
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def build_eval_windows(seq, mu, std, window_size):
    imu_norm = normalize_imu(seq["imu"], mu, std)
    pos_valid = seq.get("gps_pos_valid", seq["gps_valid"]).astype(bool)
    head_valid = seq["gps_valid"].astype(bool)
    starts, X, V, GE, GN, GH, PV, HV = [], [], [], [], [], [], [], []

    T = len(imu_norm)
    for i in range(0, T - window_size):
        end = i + window_size
        if not head_valid[i]:
            continue
        if pos_valid[i:end + 1].sum() < (window_size + 1) * 0.5:
            continue
        starts.append(i)
        X.append(imu_norm[i:end])
        V.append(seq["v_ms"][i:end])
        GE.append(seq["enu_x"][i:end + 1])
        GN.append(seq["enu_y"][i:end + 1])
        GH.append(seq["gps_theta"][i:end + 1])
        PV.append(pos_valid[i:end + 1])
        HV.append(head_valid[i:end + 1])

    if not starts:
        raise RuntimeError(f"{seq['segment']} 没有可评估窗口")

    return (
        np.array(starts, dtype=np.int32),
        np.stack(X).astype(np.float32),
        np.stack(V).astype(np.float32),
        np.stack(GE).astype(np.float32),
        np.stack(GN).astype(np.float32),
        np.stack(GH).astype(np.float32),
        np.stack(PV).astype(bool),
        np.stack(HV).astype(bool),
    )


def evaluate_segment(model, seq, mu, std, window_size, batch_size=64):
    starts, X, V, GE, GN, GH, PV, HV = build_eval_windows(seq, mu, std, window_size)
    T = len(seq["imu"])
    bias_base = model.bias_base.numpy().astype(np.float32)
    install_rpy = model.install_rpy.numpy().astype(np.float32)

    fixed_e_sum = np.zeros(T, dtype=np.float64)
    fixed_n_sum = np.zeros(T, dtype=np.float64)
    fixed_h_sum = np.zeros(T, dtype=np.complex128)
    res_e_sum = np.zeros(T, dtype=np.float64)
    res_n_sum = np.zeros(T, dtype=np.float64)
    res_h_sum = np.zeros(T, dtype=np.complex128)
    count = np.zeros(T, dtype=np.float64)
    residual_sum = np.zeros((T, 6), dtype=np.float64)
    residual_count = np.zeros(T, dtype=np.float64)

    for b0 in range(0, len(starts), batch_size):
        b1 = min(b0 + batch_size, len(starts))
        imu = tf.constant(X[b0:b1], dtype=tf.float32)
        v = tf.constant(V[b0:b1], dtype=tf.float32)
        ge = tf.constant(GE[b0:b1], dtype=tf.float32)
        gn = tf.constant(GN[b0:b1], dtype=tf.float32)
        gh = tf.constant(GH[b0:b1], dtype=tf.float32)

        fixed = predict_trajectory(model, imu, v, ge, gn, gh, mu, std, use_residual=False, training=False)
        residual_pred = predict_trajectory(model, imu, v, ge, gn, gh, mu, std, use_residual=True, training=False)
        fixed_e, fixed_n, fixed_h = [x.numpy() for x in fixed[:3]]
        res_e, res_n, res_h, _, residual, _ = residual_pred
        res_e, res_n, res_h, residual = res_e.numpy(), res_n.numpy(), res_h.numpy(), residual.numpy()

        for j, start in enumerate(starts[b0:b1]):
            sl = slice(start, start + window_size + 1)
            fixed_e_sum[sl] += fixed_e[j]
            fixed_n_sum[sl] += fixed_n[j]
            fixed_h_sum[sl] += np.exp(1j * fixed_h[j])
            res_e_sum[sl] += res_e[j]
            res_n_sum[sl] += res_n[j]
            res_h_sum[sl] += np.exp(1j * res_h[j])
            count[sl] += 1.0
            end_idx = start + window_size - 1
            residual_sum[end_idx] += residual[j]
            residual_count[end_idx] += 1.0

    valid_count = count > 0
    fixed_e = np.full(T, np.nan, dtype=np.float32)
    fixed_n = np.full(T, np.nan, dtype=np.float32)
    fixed_h = np.full(T, np.nan, dtype=np.float32)
    res_e = np.full(T, np.nan, dtype=np.float32)
    res_n = np.full(T, np.nan, dtype=np.float32)
    res_h = np.full(T, np.nan, dtype=np.float32)
    fixed_e[valid_count] = (fixed_e_sum[valid_count] / count[valid_count]).astype(np.float32)
    fixed_n[valid_count] = (fixed_n_sum[valid_count] / count[valid_count]).astype(np.float32)
    fixed_h[valid_count] = np.angle(fixed_h_sum[valid_count]).astype(np.float32)
    res_e[valid_count] = (res_e_sum[valid_count] / count[valid_count]).astype(np.float32)
    res_n[valid_count] = (res_n_sum[valid_count] / count[valid_count]).astype(np.float32)
    res_h[valid_count] = np.angle(res_h_sum[valid_count]).astype(np.float32)

    residual_series = np.zeros((T, 6), dtype=np.float32)
    has_residual = residual_count > 0
    residual_series[has_residual] = (residual_sum[has_residual] / residual_count[has_residual, None]).astype(np.float32)
    if has_residual.any():
        idx = np.where(has_residual)[0]
        for k in range(6):
            residual_series[:, k] = np.interp(np.arange(T), idx, residual_series[idx, k])

    bias_fixed = np.tile(bias_base.reshape(1, 6), (T, 1))
    bias_res = bias_fixed + residual_series

    pos_valid = seq.get("gps_pos_valid", seq["gps_valid"]).astype(bool) & valid_count
    head_valid = seq["gps_valid"].astype(bool) & valid_count

    fixed_pos_err = np.sqrt((fixed_e - seq["enu_x"]) ** 2 + (fixed_n - seq["enu_y"]) ** 2)
    res_pos_err = np.sqrt((res_e - seq["enu_x"]) ** 2 + (res_n - seq["enu_y"]) ** 2)
    fixed_head_err = np.abs(wrap_angle_np(fixed_h - seq["gps_theta"])) * RAD2DEG
    res_head_err = np.abs(wrap_angle_np(res_h - seq["gps_theta"])) * RAD2DEG

    return {
        "true_e": seq["enu_x"],
        "true_n": seq["enu_y"],
        "true_h": seq["gps_theta"],
        "fixed_e": fixed_e,
        "fixed_n": fixed_n,
        "fixed_h": fixed_h,
        "res_e": res_e,
        "res_n": res_n,
        "res_h": res_h,
        "pos_valid": pos_valid,
        "head_valid": head_valid,
        "bias_fixed": bias_fixed,
        "bias_residual": residual,
        "bias_used": bias_res,
        "install_rpy": install_rpy,
        "cmcc_bias_6d": seq.get("cmcc_bias_6d"),
        "cmcc_install_deg": seq.get("cmcc_install_deg"),
        "cmcc_stable": seq.get("cmcc_stable", np.zeros(T, dtype=bool)),
        "fixed_position_m": metric_stats(fixed_pos_err[pos_valid]),
        "residual_position_m": metric_stats(res_pos_err[pos_valid]),
        "fixed_heading_deg": metric_stats(fixed_head_err[head_valid]),
        "residual_heading_deg": metric_stats(res_head_err[head_valid]),
        "fixed_pos_err": fixed_pos_err,
        "res_pos_err": res_pos_err,
    }


def plot_results(results, segment_name, save_dir):
    t = np.arange(len(results["true_e"])) * TARGET_DT
    valid = results["pos_valid"]
    install_deg = results["install_rpy"] * RAD2DEG

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    ax.plot(results["true_e"][valid], results["true_n"][valid], "k-", lw=1.2, label="GNSS True")
    ax.plot(results["fixed_e"], results["fixed_n"], color="#1f77b4", lw=1.1, label="Fixed-only")
    ax.plot(results["res_e"], results["res_n"], "r--", lw=1.1, label="Fixed + residual")
    ax.set_title("Trajectory")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.35)
    ax.legend()

    ax = axes[0, 1]
    ax.axhline(install_deg[0], color="#9467bd", lw=1.4, label=f"pred roll {install_deg[0]:.2f}°")
    ax.axhline(install_deg[1], color="#2ca02c", lw=1.4, label=f"pred pitch {install_deg[1]:.2f}°")
    ax.axhline(install_deg[2], color="#d62728", lw=1.4, label=f"pred yaw {install_deg[2]:.2f}°")
    if results["cmcc_install_deg"] is not None:
        stable = results["cmcc_stable"]
        ax.plot(t[stable], results["cmcc_install_deg"][stable, 0], color="#2ca02c", alpha=0.45, lw=0.8, label="CMCC rbv_pitch")
        ax.plot(t[stable], results["cmcc_install_deg"][stable, 1], color="#d62728", alpha=0.45, lw=0.8, label="CMCC rbv_yaw")
    ax.set_title("Global Install Angles")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angle [deg]")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(t, results["fixed_pos_err"], color="#1f77b4", lw=0.9, label="Fixed-only")
    ax.plot(t, results["res_pos_err"], "r--", lw=0.9, label="Fixed + residual")
    ax.fill_between(t, 0, np.nanmax(results["res_pos_err"]) if np.isfinite(results["res_pos_err"]).any() else 1,
                    where=valid, alpha=0.08, color="green", label="GNSS valid")
    ax.set_title("Position Error")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Error [m]")
    ax.grid(True, alpha=0.35)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(t, results["bias_used"][:, 5], "r-", lw=0.9, label="bg_z used")
    ax.axhline(results["bias_fixed"][0, 5], color="#1f77b4", lw=1.1, label="bg_z base")
    if results["cmcc_bias_6d"] is not None:
        stable = results["cmcc_stable"]
        ax.plot(t[stable], results["cmcc_bias_6d"][stable, 5], "k-", alpha=0.5, lw=0.8, label="CMCC bg_z")
    ax.set_title("Gyro Z Bias")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Bias [deg/s]")
    ax.grid(True, alpha=0.35)
    ax.legend()

    fig.suptitle(
        f"BiasAngleNet v3 Evaluation: {segment_name} | "
        f"res RMSE={results['residual_position_m']['rmse']:.2f}m, "
        f"fixed RMSE={results['fixed_position_m']['rmse']:.2f}m",
        fontsize=13,
    )
    fig.tight_layout()
    out_path = save_dir / f"bias_angle_v3_eval_{segment_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存: {out_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate BiasAngleNet v3")
    parser.add_argument("--weights", type=str, default=str(MODEL_DIR / "bias_angle_v3.weights.h5"))
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--segments", nargs="+", default=None)
    parser.add_argument("--output-json", type=str, default=str(MODEL_DIR / "bias_angle_v3_eval_summary.json"))
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")

    model = BiasAngleNet(window_size=args.window_size)
    model(tf.zeros((1, args.window_size, 7), dtype=tf.float32))
    model.load_weights(str(weights_path))

    segments = args.segments if args.segments else DATA0109_ALL_SEGMENTS
    seqs = [s for s in load_data0109_segments(segments) if s is not None]
    mu, std, _ = load_or_compute_norm(seqs, NORM_JSON_0109)

    summary = {
        "weights": str(weights_path),
        "install_angle_deg": {
            "roll": float(model.install_rpy.numpy()[0] * RAD2DEG),
            "pitch": float(model.install_rpy.numpy()[1] * RAD2DEG),
            "yaw": float(model.install_rpy.numpy()[2] * RAD2DEG),
        },
        "bias_base": model.bias_base.numpy().astype(float).tolist(),
        "segments": {},
    }

    for seq in seqs:
        print(f"\n[评估] {seq['segment']}")
        results = evaluate_segment(model, seq, mu, std, args.window_size)
        print(
            f"  fixed-only: mean={results['fixed_position_m']['mean']:.2f}m "
            f"rmse={results['fixed_position_m']['rmse']:.2f}m "
            f"p95={results['fixed_position_m']['p95']:.2f}m"
        )
        print(
            f"  residual:   mean={results['residual_position_m']['mean']:.2f}m "
            f"rmse={results['residual_position_m']['rmse']:.2f}m "
            f"p95={results['residual_position_m']['p95']:.2f}m"
        )
        plot_results(results, seq["id"], MODEL_DIR)
        summary["segments"][seq["id"]] = {
            "name": seq["segment"],
            "fixed_position_m": results["fixed_position_m"],
            "residual_position_m": results["residual_position_m"],
            "fixed_heading_deg": results["fixed_heading_deg"],
            "residual_heading_deg": results["residual_heading_deg"],
        }

    output_json = Path(args.output_json)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Info] Summary saved: {output_json}")


if __name__ == "__main__":
    main()
