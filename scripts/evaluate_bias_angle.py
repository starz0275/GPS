#!/usr/bin/env python3
"""评估 BiasAngleNet 训练效果：零偏 + 安装角 + 轨迹对比。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from data0109_loader import (
    DATA0109_ALL_SEGMENTS,
    DATA0109_VAL_SEGMENT,
    load_data0109_segments,
)
from train_ekf import load_or_compute_norm, normalize_imu
from train_bias_angle import BiasAngleNet, integrate_trajectory, rotate_imu_to_body

MODEL_DIR = ROOT / "trained_models"
NORM_JSON_0109 = ROOT / "preprocessed_data" / "normalization_stats_data0109.json"

DEG2RAD = np.pi / 180.0
TARGET_DT = 0.1


def predict_bias_and_angle(model, imu_norm, window_size=200):
    """滑窗推理零偏和安装角。"""
    T = len(imu_norm)
    bias = np.zeros((T, 6), dtype=np.float32)
    angle = np.zeros((T, 1), dtype=np.float32)

    if T < window_size:
        return bias, angle

    windows = np.stack([imu_norm[i:i+window_size] for i in range(T - window_size + 1)])
    raw_bias, raw_angle = model(windows.astype(np.float32), training=False)
    bias[window_size-1:] = raw_bias.numpy()
    angle[window_size-1:] = raw_angle.numpy()
    # 前 window_size-1 帧用第一个有效值填充
    bias[:window_size-1] = bias[window_size-1]
    angle[:window_size-1] = angle[window_size-1]

    return bias, angle


def integrate_segment(seq, bias_pred, angle_pred, mu, std):
    """用预测的零偏和安装角做航位推算。"""
    imu_norm = normalize_imu(seq['imu'], mu, std)
    v_ms = seq['v_ms']
    T = len(imu_norm)

    # 还原原始 IMU 值
    acc_raw = imu_norm[:, :3] * std[:3] + mu[:3]  # (T, 3) [g]
    gyro_z_raw = imu_norm[:, 5] * std[5] + mu[5]  # (T,) [deg/s]

    # 去零偏
    ba = bias_pred[:, :3]  # (T, 3)
    bg_z = bias_pred[:, 5]  # (T,)
    acc_corr = acc_raw - ba
    gyro_corr_z = gyro_z_raw - bg_z

    # 旋转到车体系
    acc_body = np.zeros_like(acc_corr)
    gyro_body_z = np.zeros_like(gyro_corr_z)
    for t in range(T):
        psi = angle_pred[t, 0]  # [rad]
        ca, sa = np.cos(psi), np.sin(psi)
        acc_body[t, 0] = ca * acc_corr[t, 0] + sa * acc_corr[t, 1]
        acc_body[t, 1] = -sa * acc_corr[t, 0] + ca * acc_corr[t, 1]
        acc_body[t, 2] = acc_corr[t, 2]
        gyro_body_z[t] = gyro_corr_z[t]

    # 航位推算
    enu_e = seq['enu_x']
    enu_n = seq['enu_y']
    gps_theta = seq['gps_theta']

    # 用 tf 函数做积分
    init_e = tf.constant([enu_e[0]], dtype=tf.float32)
    init_n = tf.constant([enu_n[0]], dtype=tf.float32)
    init_h = tf.constant([gps_theta[0]], dtype=tf.float32)

    acc_tf = tf.constant(acc_body[np.newaxis], dtype=tf.float32)  # (1, T, 3)
    gyro_tf = tf.constant(gyro_body_z[np.newaxis], dtype=tf.float32)  # (1, T)
    v_tf = tf.constant(v_ms[np.newaxis], dtype=tf.float32)  # (1, T)

    pred_e, pred_n, pred_h = integrate_trajectory(
        init_e, init_n, init_h, acc_tf, gyro_tf, v_tf, TARGET_DT)

    # 返回 T+1 个点，去掉最后一个点以匹配 T
    return pred_e.numpy()[0][:T], pred_n.numpy()[0][:T], pred_h.numpy()[0][:T]


def evaluate_segment(model, seq, mu, std, window_size=200):
    """评估单段数据。"""
    imu_norm = normalize_imu(seq['imu'], mu, std)

    # 预测零偏和安装角
    bias_pred, angle_pred = predict_bias_and_angle(model, imu_norm, window_size)

    # 航位推算
    pred_e, pred_n, pred_h = integrate_segment(seq, bias_pred, angle_pred, mu, std)

    # GPS 真值
    true_e = seq['enu_x']
    true_n = seq['enu_y']
    true_h = seq['gps_theta']
    cmcc_ok = seq['cmcc_ok']
    cmcc_stable = seq['cmcc_stable']

    # 安装角真值
    install_gt = seq.get('cmcc_install_deg', None)

    return {
        'pred_e': pred_e, 'pred_n': pred_n, 'pred_h': pred_h,
        'true_e': true_e, 'true_n': true_n, 'true_h': true_h,
        'bias_pred': bias_pred, 'angle_pred': angle_pred,
        'install_gt': install_gt,
        'cmcc_ok': cmcc_ok, 'cmcc_stable': cmcc_stable,
        'cmcc_bias_6d': seq.get('cmcc_bias_6d', None),
    }


def plot_results(results, segment_name, save_dir):
    """绘制评估结果。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 轨迹对比
    ax = axes[0, 0]
    ax.plot(results['true_e'], results['true_n'], 'b-', label='GPS True', linewidth=1.5)
    ax.plot(results['pred_e'], results['pred_n'], 'r--', label='Predicted', linewidth=1.5)
    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    ax.set_title('Trajectory Comparison')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')

    # 2. 安装角对比
    ax = axes[0, 1]
    T = len(results['angle_pred'])
    t = np.arange(T) * TARGET_DT
    pred_angle_deg = results['angle_pred'][:, 0] * 180 / np.pi
    ax.plot(t, pred_angle_deg, 'r-', label='Predicted', linewidth=1)
    if results['install_gt'] is not None:
        ax.plot(t, results['install_gt'][:, 1], 'b-', label='GT (rbv_yaw)', linewidth=1)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Install Angle [deg]')
    ax.set_title('Install Angle (yaw)')
    ax.legend()
    ax.grid(True)

    # 3. 零偏对比 (陀螺 Z)
    ax = axes[1, 0]
    if results['cmcc_bias_6d'] is not None:
        ax.plot(t, results['cmcc_bias_6d'][:, 5], 'b-', label='CMCC GT', linewidth=1)
    ax.plot(t, results['bias_pred'][:, 5], 'r-', label='Predicted', linewidth=1)
    stable_mask = results['cmcc_stable']
    if stable_mask.any():
        ax.fill_between(t, 0, 1, where=stable_mask, alpha=0.2, color='green', label='Stable')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Gyro Z Bias [deg/s]')
    ax.set_title('Gyroscope Z Bias')
    ax.legend()
    ax.grid(True)

    # 4. 零偏对比 (加速度 X)
    ax = axes[1, 1]
    if results['cmcc_bias_6d'] is not None:
        ax.plot(t, results['cmcc_bias_6d'][:, 0], 'b-', label='CMCC GT', linewidth=1)
    ax.plot(t, results['bias_pred'][:, 0], 'r-', label='Predicted', linewidth=1)
    if stable_mask.any():
        ax.fill_between(t, 0, 1, where=stable_mask, alpha=0.2, color='green', label='Stable')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Acc X Bias [g]')
    ax.set_title('Accelerometer X Bias')
    ax.legend()
    ax.grid(True)

    plt.suptitle(f'BiasAngleNet Evaluation: {segment_name}', fontsize=14)
    plt.tight_layout()

    save_path = save_dir / f'bias_angle_eval_{segment_name}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'  图已保存: {save_path}')
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate BiasAngleNet')
    parser.add_argument('--weights', type=str, default=str(MODEL_DIR / 'bias_angle_v2.weights.h5'))
    parser.add_argument('--window-size', type=int, default=200)
    parser.add_argument('--segments', nargs='+', default=None)
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f'权重文件不存在: {weights_path}')
        return

    print(f'[Info] 加载权重: {weights_path}')
    model = BiasAngleNet(window_size=args.window_size)
    model(tf.zeros((1, args.window_size, 7), dtype=tf.float32))
    model.load_weights(str(weights_path))

    print('[Info] 加载数据...')
    segments = args.segments if args.segments else DATA0109_ALL_SEGMENTS
    seqs = load_data0109_segments(segments)
    seqs = [s for s in seqs if s is not None]

    mu, std, _ = load_or_compute_norm(seqs, NORM_JSON_0109)

    save_dir = MODEL_DIR
    save_dir.mkdir(exist_ok=True)

    all_results = []
    for seq in seqs:
        seg_name = seq['segment']
        print(f'\n[评估] {seg_name}')

        results = evaluate_segment(model, seq, mu, std, args.window_size)

        # 计算误差
        T = len(results['pred_e'])
        pos_err = np.sqrt((results['pred_e'] - results['true_e'])**2 +
                         (results['pred_n'] - results['true_n'])**2)

        # 安装角误差
        pred_angle_deg = results['angle_pred'][:, 0] * 180 / np.pi
        if results['install_gt'] is not None:
            angle_err = np.abs(pred_angle_deg - results['install_gt'][:, 1])
            stable = results['cmcc_stable']
            if stable.any():
                print(f'  安装角误差 (stable): mean={angle_err[stable].mean():.3f}° std={angle_err[stable].std():.3f}°')
                print(f'  安装角预测 (stable): mean={pred_angle_deg[stable].mean():.3f}°')
                print(f'  安装角真值 (stable): mean={results["install_gt"][stable, 1].mean():.3f}°')

        # 轨迹误差
        gps_valid = results['true_e'] != 0
        if gps_valid.any():
            print(f'  轨迹位置误差: mean={pos_err[gps_valid].mean():.2f}m max={pos_err[gps_valid].max():.2f}m')

        plot_results(results, seq['id'], save_dir)
        all_results.append((seg_name, results))

    # 打印总结
    print('\n' + '='*60)
    print('评估总结')
    print('='*60)

    # 加载训练结果
    result_json = weights_path.parent / (weights_path.stem.replace('.weights', '') + '_result.json')
    if result_json.exists():
        with open(result_json, 'r') as f:
            train_result = json.load(f)
        print(f'\n训练结果 ({result_json.name}):')
        print(f'  安装角 (yaw): {train_result.get("install_angle_deg", "N/A"):.4f}°')
        if 'install_gt_yaw_deg' in train_result:
            print(f'  安装角真值: {train_result["install_gt_yaw_deg"]:.4f}°')
            print(f'  安装角误差: {train_result["install_error_deg"]:.4f}°')
        if 'acc_bias_g' in train_result:
            ab = train_result['acc_bias_g']
            print(f'  加速度零偏 [g]: ba_x={ab["x"]:.6f} ba_y={ab["y"]:.6f} ba_z={ab["z"]:.6f}')
        if 'gyro_bias_degs' in train_result:
            gb = train_result['gyro_bias_degs']
            print(f'  陀螺仪零偏 [°/s]: bg_x={gb["x"]:.6f} bg_y={gb["y"]:.6f} bg_z={gb["z"]:.6f}')


if __name__ == '__main__':
    main()
