"""
隧道闭环模拟器 (双轨对比版：FP32 vs INT8)
同时运行未量化的 Keras 模型和量化后的 TFLite 模型，直观对比量化精度损失。
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def simulate_tunnel_compare():
    print("\n" + "="*80)
    print("隧道盲推模拟器 (FP32 与 INT8 精度对决)")
    print("="*80 + "\n")
    
    # 1. 路径配置
    keras_model_path = Path("trained_models/best_model.keras")
    tflite_model_path = Path("quantized_models/model_int8.tflite")
    x_path = Path("preprocessed_data/X_train.npy")
    csv_path = Path("preprocessed_data/aligned_data.csv")
    
    # 2. 检查并加载模型
    print("[1/4] 正在加载双精度模型...")
    if not keras_model_path.exists() or not tflite_model_path.exists():
        print("[FAIL] 错误：找不到模型文件，请确保 best_model.keras 和 model_int8.tflite 都存在。")
        return

    # 加载 PC 端 FP32 模型 (解除安全限制)
    print("  -> 加载 FP32 Keras 模型...")
    fp32_model = tf.keras.models.load_model(keras_model_path, safe_mode=False, compile=False)
    
    # 加载 S32K5 端 INT8 模型
    print("  -> 加载 INT8 TFLite 模型...")
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    in_scale, in_zp = input_details[0]['quantization']
    out_scale, out_zp = output_details[0]['quantization']
    
    # 3. 加载数据
    print("[2/4] 正在加载验证数据...")
    df = pd.read_csv(csv_path)
    X_data = np.load(x_path)
    window_size = X_data.shape[1] 
    
    # 4. 设置仿真参数
    start_idx = 1000       
    tunnel_length = 600    # 60 秒
    if start_idx + tunnel_length > len(X_data):
        tunnel_length = len(X_data) - start_idx

    # 初始化三套坐标系统
    true_x, true_y = [0.0], [0.0]
    fp32_x, fp32_y = [0.0], [0.0]
    int8_x, int8_y = [0.0], [0.0]
    
    curr_true_x, curr_true_y = 0.0, 0.0
    curr_fp32_x, curr_fp32_y = 0.0, 0.0
    curr_int8_x, curr_int8_y = 0.0, 0.0
    
    # 误差记录
    err_fp32_list, err_int8_list = [0.0], [0.0]
    time_axis_full = np.linspace(0, tunnel_length/10, tunnel_length + 1)
    
    print(f"[3/4] 驶入隧道，开始 {tunnel_length/10:.1f} 秒的平行时空盲推...")
    
    # 5. 核心双轨推算循环
    for step, i in enumerate(range(start_idx, start_idx + tunnel_length)):
        current_window = X_data[i:i+1].astype(np.float32)
        
        # --- 轨道 A: FP32 推理 (PC 级精度) ---
        # 注意: 使用 fp32_model(x) 比 fp32_model.predict(x) 在循环中快得多
        fp32_output = fp32_model(current_window, training=False)
        pred_dx_fp32 = float(fp32_output[0, 0])
        pred_dy_fp32 = float(fp32_output[0, 1])
        
        # --- 轨道 B: INT8 推理 (S32K5 级精度) ---
        q_input = (current_window / in_scale + in_zp).astype(np.int8)
        interpreter.set_tensor(input_details[0]['index'], q_input)
        interpreter.invoke()
        q_output = interpreter.get_tensor(output_details[0]['index'])
        pred_dx_int8 = (q_output[0, 0].astype(np.float32) - out_zp) * out_scale
        pred_dy_int8 = (q_output[0, 1].astype(np.float32) - out_zp) * out_scale
        
        # --- 物理底盘数据 ---
        real_row_idx = i + window_size 
        if real_row_idx >= len(df): real_row_idx = len(df) - 1
        base_dx = df['Base_dx'].iloc[real_row_idx]
        base_dy = df['Base_dy'].iloc[real_row_idx]
        true_res_dx = df['Target_dx'].iloc[real_row_idx]
        true_res_dy = df['Target_dy'].iloc[real_row_idx]
        
        # --- 坐标累加 ---
        curr_true_x += (base_dx + true_res_dx)
        curr_true_y += (base_dy + true_res_dy)
        true_x.append(curr_true_x)
        true_y.append(curr_true_y)
        
        curr_fp32_x += (base_dx + pred_dx_fp32)
        curr_fp32_y += (base_dy + pred_dy_fp32)
        fp32_x.append(curr_fp32_x)
        fp32_y.append(curr_fp32_y)
        
        curr_int8_x += (base_dx + pred_dx_int8)
        curr_int8_y += (base_dy + pred_dy_int8)
        int8_x.append(curr_int8_x)
        int8_y.append(curr_int8_y)
        
        # --- 误差统计 ---
        err_fp32_list.append(np.sqrt((curr_true_x - curr_fp32_x)**2 + (curr_true_y - curr_fp32_y)**2))
        err_int8_list.append(np.sqrt((curr_true_x - curr_int8_x)**2 + (curr_true_y - curr_int8_y)**2))

    # 6. 画图与分析
    final_err_fp32 = err_fp32_list[-1]
    final_err_int8 = err_int8_list[-1]
    
    print("\n[4/4] 驶出隧道！精度对决结果：")
    print(f"  [FP32] 原始模型最终误差: {final_err_fp32:.2f} 米")
    print(f"  [INT8] 量化模型最终误差: {final_err_int8:.2f} 米")
    print(f"  [GAP] 量化掉件 (Quantization Loss): {abs(final_err_int8 - final_err_fp32):.2f} 米\n")
    
    # 画图
    fig, (ax_traj, ax_err) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'FP32 vs INT8 Precision Comparison (Tunnel: {tunnel_length/10}s)', fontsize=16, fontweight='bold')
    
    # 轨迹对比图
    ax_traj.plot(true_x, true_y, 'g-', label='True GPS Trajectory', linewidth=3)
    ax_traj.plot(fp32_x, fp32_y, color='darkorange', linestyle='-.', label=f'FP32 Keras (Err: {final_err_fp32:.2f}m)', linewidth=2.5)
    ax_traj.plot(int8_x, int8_y, 'b--', label=f'INT8 TFLite (Err: {final_err_int8:.2f}m)', linewidth=2.5)
    
    ax_traj.plot(true_x[0], true_y[0], 'ko', markersize=8, label='Start')
    ax_traj.plot(true_x[-1], true_y[-1], 'g*', markersize=12)
    ax_traj.plot(fp32_x[-1], fp32_y[-1], color='darkorange', marker='X', markersize=10)
    ax_traj.plot(int8_x[-1], int8_y[-1], 'bs', markersize=10)
    
    ax_traj.set_title('Top-Down Trajectory Comparison')
    ax_traj.set_xlabel('East (meters)')
    ax_traj.set_ylabel('North (meters)')
    ax_traj.legend()
    ax_traj.grid(True, alpha=0.4)
    ax_traj.axis('equal')
    
    # 误差增长对比图
    ax_err.plot(time_axis_full, err_fp32_list, color='darkorange', linestyle='-.', label='FP32 Cumulative Error', linewidth=2.5)
    ax_err.plot(time_axis_full, err_int8_list, 'b-', label='INT8 Cumulative Error', linewidth=2.5)
    
    ax_err.fill_between(time_axis_full, err_fp32_list, err_int8_list, color='gray', alpha=0.2, label='Quantization Gap')
    
    ax_err.set_title('Cumulative Drift Error over Time')
    ax_err.set_xlabel('Time (seconds)')
    ax_err.set_ylabel('Drift Error (meters)')
    ax_err.legend()
    ax_err.grid(True, alpha=0.4)
    
    plt.tight_layout()
    output_path = Path("trained_models/tunnel_trajectory.png")
    plt.savefig(str(output_path), dpi=150)
    print(f"\n[OK] 轨迹图已保存: {output_path}")

if __name__ == "__main__":
    simulate_tunnel_compare()