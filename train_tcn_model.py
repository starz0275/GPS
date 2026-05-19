"""
1D-TCN 模型训练脚本 V2
========================
改进点：
  1. Huber 损失（对残余标签异常值更鲁棒，替代 MSE）
  2. 训练完成后自动保存完整 training_info.json（含当前 run 结果）
  3. 每 epoch 打印更丰富的指标
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

OUTPUT_DIR  = Path(__file__).parent / "preprocessed_data"
MODEL_DIR   = Path(__file__).parent / "trained_models"
MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE   = 16
EPOCHS       = 150
VAL_SPLIT    = 0.2
RANDOM_SEED  = 42

INITIAL_LR   = 0.001
LR_PATIENCE  = 5
LR_FACTOR    = 0.5
ES_PATIENCE  = 15       # 早停耐心（略放宽，给 Huber 更多时间）
HUBER_DELTA  = 0.1      # Huber 损失折点（m 量级）

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================================
# 自定义损失 / 指标
# ============================================================================

def huber_loss(delta=HUBER_DELTA):
    """Huber 损失：|e|<delta 时平方，否则线性（对异常值鲁棒）"""
    def loss_fn(y_true, y_pred):
        err = y_true - y_pred
        abs_err = tf.abs(err)
        quadratic = tf.minimum(abs_err, delta)
        linear    = abs_err - quadratic
        return tf.reduce_mean(0.5 * quadratic**2 + delta * linear)
    loss_fn.__name__ = 'huber_loss'
    return loss_fn


def rmse_metric(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))


# ============================================================================
# TCN 模型（与 V1 相同结构，INT8 量化友好）
# ============================================================================

def build_tcn_model(input_shape):
    def res_block(x, filters, dilation, block_id):
        shortcut = x
        x = layers.Conv1D(filters * 2, 1, padding='same', activation='relu',
                          name=f'res{block_id}_expand')(x)
        x = layers.Conv1D(filters, 3, dilation_rate=dilation, padding='causal',
                          activation='relu', name=f'res{block_id}_dilated')(x)
        if shortcut.shape[-1] != x.shape[-1]:
            shortcut = layers.Conv1D(filters, 1, padding='same',
                                     name=f'res{block_id}_shortcut')(shortcut)
        x = layers.Add(name=f'res{block_id}_add')([shortcut, x])
        return x

    inputs = layers.Input(shape=input_shape, name="tcn_input")
    x = layers.Conv1D(64, 3, padding='causal', activation='relu',
                      name='init_conv')(inputs)

    for i, d in enumerate([1, 2, 4, 8, 16, 32], start=1):
        x = res_block(x, 64, dilation=d, block_id=i)

    x = layers.Lambda(lambda t: t[:, -1, :], name='temporal_last_step')(x)
    x = layers.Dense(128, activation='relu', name='dense_1')(x)
    x = layers.Dense(64,  activation='relu', name='dense_2')(x)
    x = layers.Dense(32,  activation='relu', name='dense_3')(x)
    outputs = layers.Dense(2, activation='linear', name='residual_output')(x)

    return Model(inputs, outputs, name='TCN_ResidualPredictor_V2')


# ============================================================================
# 数据加载
# ============================================================================

def load_and_split_data():
    print("[Data] 读取预处理数据...")
    X_tr = np.load(OUTPUT_DIR / "X_train.npy")
    Y_tr = np.load(OUTPUT_DIR / "Y_train.npy")
    val_path = OUTPUT_DIR / "X_val.npy"

    with open(OUTPUT_DIR / "normalization_stats.json") as f:
        config = json.load(f)

    if val_path.exists():
        X_val = np.load(val_path)
        Y_val = np.load(OUTPUT_DIR / "Y_val.npy")
        print(f"  训练: {len(X_tr)} 窗  验证: {len(X_val)} 窗 (held-out)")
        return X_tr, Y_tr, X_val, Y_val, config

    print(f"  训练样本: {len(X_tr)}（无 X_val.npy，回退随机 {VAL_SPLIT:.0%} 划分）")
    np.random.seed(RANDOM_SEED)
    idx = np.random.permutation(len(X_tr))
    X_tr, Y_tr = X_tr[idx], Y_tr[idx]
    split = int(len(X_tr) * (1 - VAL_SPLIT))
    return X_tr[:split], Y_tr[:split], X_tr[split:], Y_tr[split:], config


# ============================================================================
# 训练主函数
# ============================================================================

def train_model():
    print("=" * 70)
    print("1D-TCN 训练 V2（多段训练 / held-out 验证，见 dataset_split.json）")
    print("=" * 70)

    X_tr, Y_tr, X_val, Y_val, config = load_and_split_data()
    input_shape = (X_tr.shape[1], X_tr.shape[2])
    print(f"\n[Model] 输入形状: {input_shape}")

    model = build_tcn_model(input_shape)
    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR),
        loss=huber_loss(HUBER_DELTA),
        metrics=['mae', rmse_metric]
    )

    checkpoint = ModelCheckpoint(
        filepath=str(MODEL_DIR / "best_model.keras"),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=ES_PATIENCE,
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

    print("\n开始训练...")
    history = model.fit(
        X_tr, Y_tr,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, Y_val),
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )

    # ---- 评估 ----
    tr_metrics  = model.evaluate(X_tr,  Y_tr,  verbose=0)
    val_metrics = model.evaluate(X_val, Y_val, verbose=0)

    # 反归一化预测（计算真实误差）
    with open(OUTPUT_DIR / "normalization_stats.json") as f:
        stats_json = json.load(f)

    Y_pred_val = model.predict(X_val, verbose=0)

    dx_mae  = float(np.mean(np.abs(Y_pred_val[:, 0] - Y_val[:, 0])))
    dy_mae  = float(np.mean(np.abs(Y_pred_val[:, 1] - Y_val[:, 1])))
    dx_rmse = float(np.sqrt(np.mean((Y_pred_val[:, 0] - Y_val[:, 0])**2)))
    dy_rmse = float(np.sqrt(np.mean((Y_pred_val[:, 1] - Y_val[:, 1])**2)))

    # 相对误差（相对于标签标准差）
    dx_std = float(np.std(Y_val[:, 0])) + 1e-8
    dy_std = float(np.std(Y_val[:, 1])) + 1e-8

    epochs_run = len(history.history['loss'])
    best_val   = float(min(history.history['val_loss']))

    print(f"\n{'='*70}")
    print(f"训练完成  epochs={epochs_run}  best_val_loss={best_val:.6f}")
    print(f"dx: MAE={dx_mae:.4f}m  RMSE={dx_rmse:.4f}m  relative={dx_mae/dx_std*100:.1f}%")
    print(f"dy: MAE={dy_mae:.4f}m  RMSE={dy_rmse:.4f}m  relative={dy_mae/dy_std*100:.1f}%")
    print(f"{'='*70}")

    # ---- 保存训练信息 ----
    info = {
        "epochs_trained":    epochs_run,
        "best_val_loss":     best_val,
        "final_train_loss":  float(history.history['loss'][-1]),
        "final_val_loss":    float(history.history['val_loss'][-1]),
        "huber_delta":       HUBER_DELTA,
        "train_metrics": {
            "loss": float(tr_metrics[0]),
            "mae":  float(tr_metrics[1]),
            "rmse": float(tr_metrics[2])
        },
        "val_metrics": {
            "loss": float(val_metrics[0]),
            "mae":  float(val_metrics[1]),
            "rmse": float(val_metrics[2])
        },
        "dx_analysis": {
            "mae":                    dx_mae,
            "rmse":                   dx_rmse,
            "relative_error_percent": dx_mae / dx_std * 100
        },
        "dy_analysis": {
            "mae":                    dy_mae,
            "rmse":                   dy_rmse,
            "relative_error_percent": dy_mae / dy_std * 100
        }
    }
    with open(MODEL_DIR / "training_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    print("  [OK] training_info.json 已保存")

    # ---- Loss 曲线 ----
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'],     label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Huber Loss'); plt.xlabel('Epoch'); plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'],     label='train MAE')
    plt.plot(history.history['val_mae'], label='val MAE')
    plt.title('MAE (m)'); plt.xlabel('Epoch'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(MODEL_DIR / "training_history.png", dpi=120)
    print("  [OK] training_history.png 已保存")

    return model, history


if __name__ == "__main__":
    train_model()
