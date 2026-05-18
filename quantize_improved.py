"""
增强的量化管道 - 完整版本
包含改进的错误处理、参数验证、INT8兼容性检查
"""
import tensorflow as tf
import numpy as np
from pathlib import Path
import json

# ============================================================================
# 配置与常量
# ============================================================================

PREPROCESSED_DATA_DIR = Path(__file__).parent / "preprocessed_data"
QUANTIZATION_PARAMS = {
    "num_samples": 500,  # 增加代表集大小以提高精度
    "representative_batch_size": 1,
}

# ============================================================================
# 预检查函数
# ============================================================================

def verify_int8_compatibility(keras_model_path):
    """
    检查模型是否满足INT8量化要求
    
    INT8 兼容性检查清单:
    [OK] Conv1D, Dense, Add - 支持
    [FAIL] Dropout - 不支持，训练推理不一致
    [FAIL] BatchNormalization - INT8动态范围问题
    """
    print("[Compatibility Check] 验证INT8兼容性...")
    
    model = tf.keras.models.load_model(keras_model_path)
    
    # 获取所有层名称
    layer_names = [layer.name for layer in model.layers]
    
    forbidden_layers = {
        'dropout': 'Dropout层违反INT8兼容性（训练推理不一致）',
        'batch_normalization': 'BatchNormalization不支持INT8量化',
    }
    
    issues = []
    for layer_name in layer_names:
        for keyword, issue in forbidden_layers.items():
            if keyword in layer_name.lower():
                issues.append(f"  [WARN] {layer_name}: {issue}")
    
    if issues:
        print("  发现不兼容层:")
        for issue in issues:
            print(issue)
        return False
    
    print("  [OK] 模型满足INT8兼容性要求")
    return True


def validate_input_data():
    """验证预处理数据是否存在且有效"""
    print("[Data Validation] 验证输入数据...")
    
    required_files = {
        'X_train.npy': 'X_train 数据集',
        'Y_train.npy': 'Y_train 目标集',
        'normalization_stats.json': '归一化统计参数',
    }
    
    for filename, desc in required_files.items():
        filepath = PREPROCESSED_DATA_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(f"缺少文件: {filepath} ({desc})")
        print(f"  [OK] {filename} ({filepath.stat().st_size / 1024:.1f} KB)")
    
    # 加载并验证数据
    try:
        X_train = np.load(PREPROCESSED_DATA_DIR / 'X_train.npy')
        Y_train = np.load(PREPROCESSED_DATA_DIR / 'Y_train.npy')
        
        assert X_train.shape[0] == Y_train.shape[0], "X和Y样本数不一致"
        assert X_train.shape[2] == 9, "输入特征维度应为9"
        assert Y_train.shape[1] == 2, "输出维度应为2 (dx, dy)"
        
        print(f"  [OK] 数据形状验证: X{X_train.shape} → Y{Y_train.shape}")
        return True
        
    except Exception as e:
        print(f"  [FAIL] 数据验证失败: {e}")
        return False


# ============================================================================
# 改进的量化函数
# ============================================================================

def create_quantization_dataset(num_samples=500):
    """
    创建量化代表集（更大的样本集提高精度）
    """
    print("[Dataset] 创建量化代表集...")
    
    X_train = np.load(PREPROCESSED_DATA_DIR / "X_train.npy")
    
    # 随机选择样本
    np.random.seed(42)
    indices = np.random.choice(
        len(X_train), 
        size=min(num_samples, len(X_train)), 
        replace=False
    )
    quant_samples = X_train[indices].astype(np.float32)
    
    print(f"  代表集大小: {len(quant_samples)} 样本")
    print(f"  样本形状: {quant_samples.shape}")
    
    def representative_dataset():
        """生成器 - 为量化提供代表性数据"""
        for i in range(len(quant_samples)):
            yield [quant_samples[i:i+1]]
    
    return representative_dataset


def quantize_model(keras_model_path, output_dir="quantized_models"):
    """
    改进的INT8量化函数 - 完整的错误处理
    """
    print("\n" + "="*80)
    print("[Quantization Pipeline] INT8量化转换")
    print("="*80)
    
    # 1. 验证输入
    model_path = Path(keras_model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    if not verify_int8_compatibility(keras_model_path):
        print("\n[WARN] 警告：模型可能不满足INT8要求，继续量化可能失败")
    
    # 2. 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    quantized_model_path = output_path / "model_int8.tflite"
    
    # 3. 加载模型并创建转换器
    try:
        print("\n[Loading] 加载Keras模型...")
        model = tf.keras.models.load_model(model_path)
        print(f"  [OK] 模型加载成功 (参数数: {model.count_params():,})")
        
        # 创建转换器 (移除已弃用参数)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
    except Exception as e:
        print(f"  [FAIL] 模型加载失败: {e}")
        return None
    
    # 4. 配置量化参数
    try:
        print("\n[Configuration] 配置INT8量化...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        
        # 创建代表集
        representative_dataset = create_quantization_dataset(
            num_samples=QUANTIZATION_PARAMS['num_samples']
        )
        
        converter.representative_dataset = representative_dataset
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        print("  [OK] 量化配置完成")
        
    except Exception as e:
        print(f"  [FAIL] 配置失败: {e}")
        return None
    
    # 5. 执行转换
    try:
        print("\n[Conversion] 执行转换...")
        quantized_tflite = converter.convert()
        
        with open(quantized_model_path, 'wb') as f:
            f.write(quantized_tflite)
        
        model_size_kb = quantized_model_path.stat().st_size / 1024
        print(f"  [OK] 转换成功")
        print(f"  保存路径: {quantized_model_path}")
        print(f"  模型大小: {model_size_kb:.1f} KB")
        
    except Exception as e:
        print(f"  [FAIL] 转换失败: {e}")
        return None
    
    # 6. 验证量化模型
    try:
        print("\n[Verification] 验证量化模型...")
        X_test = np.load(PREPROCESSED_DATA_DIR / "X_train.npy")[:10].astype(np.float32)
        
        interpreter = tf.lite.Interpreter(model_path=str(quantized_model_path))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"  输入形状: {input_details[0]['shape']}")
        print(f"  输出形状: {output_details[0]['shape']}")
        print(f"  输入量化: scale={input_details[0]['quantization'][0]:.6f}, "
              f"zero_point={input_details[0]['quantization'][1]}")
        print(f"  输出量化: scale={output_details[0]['quantization'][0]:.6f}, "
              f"zero_point={output_details[0]['quantization'][1]}")
        
        # 测试推理
        test_input = X_test[0:1]
        in_scale, in_zp = input_details[0]['quantization']
        q_input = (test_input / in_scale + in_zp).astype(np.int8)
        
        interpreter.set_tensor(input_details[0]['index'], q_input)
        interpreter.invoke()
        
        q_output = interpreter.get_tensor(output_details[0]['index'])
        out_scale, out_zp = output_details[0]['quantization']
        f_output = (q_output.astype(np.float32) - out_zp) * out_scale
        
        print(f"  [OK] 推理验证成功")
        print(f"  样本输出: [dx={f_output[0, 0]:.6f}, dy={f_output[0, 1]:.6f}]")
        
    except Exception as e:
        print(f"  [FAIL] 验证失败: {e}")
        return None
    
    # 7. 保存量化信息
    try:
        quantization_info = {
            'model_type': 'INT8_TFLite',
            'input_shape': input_details[0]['shape'].tolist(),
            'output_shape': output_details[0]['shape'].tolist(),
            'input_quantization': {
                'scale': float(input_details[0]['quantization'][0]),
                'zero_point': int(input_details[0]['quantization'][1]),
            },
            'output_quantization': {
                'scale': float(output_details[0]['quantization'][0]),
                'zero_point': int(output_details[0]['quantization'][1]),
            },
            'model_size_kb': model_size_kb,
            'keras_model_params': int(model.count_params()),
        }
        
        info_path = output_path / "quantization_info.json"
        with open(info_path, 'w') as f:
            json.dump(quantization_info, f, indent=2)
        
        print(f"\n  [OK] 量化信息已保存: {info_path}")
        
    except Exception as e:
        print(f"  [FAIL] 保存信息失败: {e}")
    
    return str(quantized_model_path)


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("INT8 TFLite 量化管道")
    print("="*80)
    
    try:
        # 验证预处理数据
        if not validate_input_data():
            print("\n[FAIL] 数据验证失败，无法继续")
            exit(1)
        
        # 执行量化 (假设已有已训练的模型)
        trained_model = Path("trained_models") / "best_model.keras"
        if trained_model.exists():
            quantize_model(str(trained_model))
        else:
            print(f"\n[WARN] 未找到已训练模型: {trained_model}")
            print("请先运行 train_tcn_model.py 来训练模型")
    
    except Exception as e:
        print(f"\n[FAIL] 处理失败: {e}")
        import traceback
        traceback.print_exc()
