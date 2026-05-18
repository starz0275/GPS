"""
1D-TCN 模型训练脚本 (自适应形状版)
用于GPS缺失场景下的位移增量残差预测
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 配置
# ============================================================================

OUTPUT_DIR = Path(__file__).parent / "preprocessed_data"
MODEL_DIR = Path(__file__).parent / "trained_models"
MODEL_DIR.mkdir(exist_ok=True)

# 训练参数
BATCH_SIZE = 16  # 减小批次以获得更频繁的梯度更新
EPOCHS = 150  # 增加轮数
VAL_SPLIT = 0.2
RANDOM_SEED = 42

# 学习率参数
INITIAL_LR = 0.001  # 标准学习率
LR_PATIENCE = 5  # 中等耐心
LR_FACTOR = 0.5

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================================
# TCN 模型构建 (支持动态输入形状)
# ============================================================================

def build_tcn_model(input_shape):
    """
    构建增强版1D-TCN模型 - 6层深度残差塔 + Dropout正则化
    """
    def res_block(x, filters, dilation, block_id):
        """S32K5 优化的残差块 (INT8量化友好)"""
        shortcut = x

        # 升维 -> 扩张卷积 -> 投影 (无Dropout - INT8兼容)
        x = layers.Conv1D(filters * 2, 1, padding='same', activation='relu',
                         name=f'res{block_id}_expand')(x)

        x = layers.Conv1D(filters, 3, dilation_rate=dilation, padding='causal',
                         activation='relu', name=f'res{block_id}_dilated')(x)

        # 确保维度一致 (关键！)
        if shortcut.shape[-1] != x.shape[-1]:
            shortcut = layers.Conv1D(filters, 1, padding='same',
                                    name=f'res{block_id}_shortcut')(shortcut)

        x = layers.Add(name=f'res{block_id}_add')([shortcut, x])
        return x

    inputs = layers.Input(shape=input_shape, name="tcn_input")

    # 初始投影：特征维 -> 64维
    x = layers.Conv1D(64, 3, padding='causal', activation='relu',
                     name='init_conv')(inputs)

    # 深度残差塔（6层）- dilation rates: 1,2,4,8,16,32 (超长感受野)
    x = res_block(x, 64, dilation=1,  block_id=1)
    x = res_block(x, 64, dilation=2,  block_id=2)
    x = res_block(x, 64, dilation=4,  block_id=3)
    x = res_block(x, 64, dilation=8,  block_id=4)
    x = res_block(x, 64, dilation=16, block_id=5)
    x = res_block(x, 64, dilation=32, block_id=6)

    # 时间融合：取最后时间步
    x = layers.Lambda(lambda x: x[:, -1, :], name='temporal_last_step')(x)

    # 多级密集融合 (无Dropout - INT8兼容)
    x = layers.Dense(128, activation='relu', name='dense_1')(x)

    x = layers.Dense(64, activation='relu', name='dense_2')(x)

    x = layers.Dense(32, activation='relu', name='dense_3')(x)

    # 输出层（无激活函数，用于回归）
    outputs = layers.Dense(2, activation='linear', name='residual_output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='TCN_ResidualPredictor_Enhanced')
    return model


# ============================================================================
# 数据加载与划分
# ============================================================================

def load_and_split_data(val_split=0.2, random_seed=42):
    """加载预处理数据"""
    print("[Data Loading] 读取预处理数据...")

    X = np.load(OUTPUT_DIR / "X_train.npy")
    Y = np.load(OUTPUT_DIR / "Y_train.npy")

    with open(OUTPUT_DIR / "normalization_stats.json") as f:
        config = json.load(f)

    print(f"   数据集大小: {len(X)} 样本")
    print(f"   检测到输入形状: {X.shape} (样本数, 时间步, 特征数)")

    # 随机打乱
    np.random.seed(random_seed)
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]

    # 划分
    split_idx = int(len(X) * (1 - val_split))
    X_train, X_val = X_shuffled[:split_idx], X_shuffled[split_idx:]
    Y_train, Y_val = Y_shuffled[:split_idx], Y_shuffled[split_idx:]

    return X_train, Y_train, X_val, Y_val, config


# ============================================================================
# 自定义指标
# ============================================================================

def rmse_metric(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))


# ============================================================================
# 主训练函数 (自动提取形状)
# ============================================================================

def train_model():
    print("="*80)
    print("1D-TCN 模型训练 - 动态适配版")
    print("="*80)

    # 1. 加载数据
    X_train, Y_train, X_val, Y_val, config = load_and_split_data(
        val_split=VAL_SPLIT,
        random_seed=RANDOM_SEED
    )

    # 2. 核心：自动获取输入形状 (例如获取 50, 9)
    auto_input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"\n[Shape Adapt] 模型输入自动设置为: {auto_input_shape}")

    # 3. 构建模型
    print("\n[Model Building]")
    model = build_tcn_model(input_shape=auto_input_shape)
    model.summary()

    # 4. 编译模型
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR),
        loss='mse',
        metrics=['mae', rmse_metric]
    )

    # 5. 回调函数
    checkpoint = ModelCheckpoint(
        filepath=str(MODEL_DIR / "best_model.keras"),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=1e-6,
        verbose=1
    )

    # 6. 开始训练
    print("\n开始训练...")
    history = model.fit(
        X_train, Y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, Y_val),
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )

    # 7. 评估与保存
    val_metrics = model.evaluate(X_val, Y_val, verbose=0)
    print(f"\n[验证集最终性能] Loss: {val_metrics[0]:.6f}, MAE: {val_metrics[1]:.6f}, RMSE: {val_metrics[2]:.6f}")

    Y_pred = model.predict(X_val, verbose=0)
    dx_mae = np.mean(np.abs(Y_val[:, 0] - Y_pred[:, 0]))
    dx_rmse = np.sqrt(np.mean(np.square(Y_val[:, 0] - Y_pred[:, 0])))
    dy_mae = np.mean(np.abs(Y_val[:, 1] - Y_pred[:, 1]))
    dy_rmse = np.sqrt(np.mean(np.square(Y_val[:, 1] - Y_pred[:, 1])))
    dx_range = Y_val[:, 0].max() - Y_val[:, 0].min() + 0.001
    dy_range = Y_val[:, 1].max() - Y_val[:, 1].min() + 0.001
    print(f"  dx - MAE: {dx_mae:.6f}, RMSE: {dx_rmse:.6f}, RelErr: {dx_rmse/dx_range*100:.1f}%")
    print(f"  dy - MAE: {dy_mae:.6f}, RMSE: {dy_rmse:.6f}, RelErr: {dy_rmse/dy_range*100:.1f}%")

    epochs_trained = len(history.history['loss'])
    info = {
        'epochs_trained': epochs_trained,
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'best_val_loss': float(min(history.history['val_loss'])),
        'val_metrics': {
            'loss': float(val_metrics[0]),
            'mae': float(val_metrics[1]),
            'rmse': float(val_metrics[2])
        },
        'dx_analysis': {
            'mae': float(dx_mae),
            'rmse': float(dx_rmse),
            'relative_error_percent': float(dx_rmse/dx_range*100)
        },
        'dy_analysis': {
            'mae': float(dy_mae),
            'rmse': float(dy_rmse),
            'relative_error_percent': float(dy_rmse/dy_range*100)
        },
    }
    with open(MODEL_DIR / "training_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    print(f"  [OK] training_info.json 已保存")

    return model, history


if __name__ == "__main__":
    model, history = train_model()
    # 保存训练曲线图
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(str(MODEL_DIR / "training_history.png"), dpi=100)
    print(f"[OK] training_history.png 已保存")
