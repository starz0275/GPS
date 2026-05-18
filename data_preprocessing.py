"""
数据预处理脚本：IMU + 车速 + GPS 多模态融合
核心功能：
  1. 时间戳对齐到10Hz
  2. 物理航位推算 (Dead Reckoning)
  3. GPS → ENU坐标转换
  4. 构建残差学习目标
  5. 滑动窗口切分
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 配置与常量
# ============================================================================

DATA_DIR = Path(__file__).parent / "标定实车数据"
OUTPUT_DIR = Path(__file__).parent / "preprocessed_data"
OUTPUT_DIR.mkdir(exist_ok=True)

# 地球参数 (WGS84)
EARTH_RADIUS = 6371000.0  # 地球半径 (m)
DEGREES_TO_RADIANS = np.pi / 180.0
RADIANS_TO_DEGREES = 180.0 / np.pi

# 数据处理参数
TARGET_FREQ = 10  # Hz
TARGET_DT = 1.0 / TARGET_FREQ  # 0.1 s
WINDOW_SIZE = 20  # 时间步 (50 * 0.1s = 5.0s)
WINDOW_STRIDE = 1  # 滑动步长


def clean_column_names(columns):
    """清理列名中的BOM和空格"""
    cleaned = []
    for col in columns:
        # 移除BOM标记 'ï»¿' 和其他尾随/前导空格
        col_cleaned = col.replace('ï»¿', '').strip()
        cleaned.append(col_cleaned)
    return cleaned


# ============================================================================
# 工具函数：坐标系转换
# ============================================================================

def wgs84_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    """
    将WGS84经纬度坐标转换为局部ENU坐标系
    
    参数：
        lat, lon, alt: 待转换点的纬度(°)、经度(°)、高度(m)
        ref_lat, ref_lon, ref_alt: 参考点的纬度(°)、经度(°)、高度(m)
    
    返回：
        east, north, up: ENU坐标 (单位: 米)
    """
    lat_rad = lat * DEGREES_TO_RADIANS
    lon_rad = lon * DEGREES_TO_RADIANS
    ref_lat_rad = ref_lat * DEGREES_TO_RADIANS
    ref_lon_rad = ref_lon * DEGREES_TO_RADIANS
    
    # WGS84椭球体参数
    a = 6378137.0  # 长半轴
    e2 = 0.00669437999014132  # 第一偏心率平方
    
    # 计算参考点的曲率半径
    N_ref = a / np.sqrt(1 - e2 * np.sin(ref_lat_rad)**2)
    
    # ECEF坐标（相对于椭球中心）
    X = (N_ref + ref_alt) * np.cos(ref_lat_rad) * np.cos(ref_lon_rad)
    Y = (N_ref + ref_alt) * np.cos(ref_lat_rad) * np.sin(ref_lon_rad)
    Z = (N_ref * (1 - e2) + ref_alt) * np.sin(ref_lat_rad)
    
    x = (EARTH_RADIUS + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (EARTH_RADIUS + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (EARTH_RADIUS * (1 - e2) + alt) * np.sin(lat_rad)
    
    dx = x - X
    dy = y - Y
    dz = z - Z
    
    # 旋转矩阵：从ECEF转到ENU
    sin_lat = np.sin(ref_lat_rad)
    cos_lat = np.cos(ref_lat_rad)
    sin_lon = np.sin(ref_lon_rad)
    cos_lon = np.cos(ref_lon_rad)
    
    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    
    return east, north, up


def compute_heading_from_gyro(gyro_z, dt):
    """
    通过Z轴角速度积分计算航向角
    
    参数：
        gyro_z: Z轴角速度 (°/s), numpy数组
        dt: 时间间隔 (s)
    
    返回：
        heading: 累积航向角 (°), numpy数组
    """
    # 积分：heading = ∑ gyro_z * dt
    heading = np.cumsum(gyro_z * dt)
    return heading


# ============================================================================
# 数据加载与对齐
# ============================================================================

def load_and_align_data(data_dir, target_freq=10):
    """
    加载三个传感器数据文件，对齐到统一的时间戳
    
    参数：
        data_dir: 数据目录路径
        target_freq: 目标频率 (Hz)
    
    返回：
        aligned_df: 对齐后的DataFrame
    """
    print("[Loading] 读取传感器数据...")
    
    # 加载IMU数据
    imu_file = data_dir / "Data01_IMU.txt"
    imu_df = pd.read_csv(imu_file, sep='\t', skipinitialspace=True, encoding='utf-8-sig')
    imu_df.columns = clean_column_names(imu_df.columns)
    
    # 加载车速数据
    speed_file = data_dir / "Data01_VehicleSpeed.txt"
    speed_df = pd.read_csv(speed_file, sep='\t', skipinitialspace=True, encoding='utf-8-sig')
    speed_df.columns = clean_column_names(speed_df.columns)
    
    # 加载GPS数据
    gps_file = data_dir / "Data01_GNSS.txt"
    gps_df = pd.read_csv(gps_file, sep='\t', skipinitialspace=True, encoding='utf-8-sig')
    gps_df.columns = clean_column_names(gps_df.columns)
    
    print(f"  IMU: {len(imu_df)} 条记录, Time range: {imu_df['Time_s'].min():.3f} - {imu_df['Time_s'].max():.3f}")
    print(f"  Speed: {len(speed_df)} 条记录, Time range: {speed_df['Time_s'].min():.3f} - {speed_df['Time_s'].max():.3f}")
    print(f"  GPS: {len(gps_df)} 条记录, Time range: {gps_df['Time_s'].min():.3f} - {gps_df['Time_s'].max():.3f}")
    
    # 确定统一的时间范围
    t_start = max(imu_df['Time_s'].min(), speed_df['Time_s'].min(), gps_df['Time_s'].min())
    t_end = min(imu_df['Time_s'].max(), speed_df['Time_s'].max(), gps_df['Time_s'].max())
    
    print(f"[Aligning] 统一时间范围: {t_start:.3f} - {t_end:.3f} s")
    
    # 生成目标时间戳 (10 Hz)
    target_dt = 1.0 / target_freq
    t_aligned = np.arange(t_start, t_end, target_dt)
    
    print(f"[Resampling] 目标频率: {target_freq} Hz, 共 {len(t_aligned)} 个时间步")
    
    # 插值各传感器数据
    imu_interp = {}
    for col in ['AccX_g', 'AccY_g', 'AccZ_g', 'GyroX_degs', 'GyroY_degs', 'GyroZ_degs']:
        f_interp = interp1d(imu_df['Time_s'], imu_df[col], kind='cubic', fill_value='extrapolate')
        imu_interp[col] = f_interp(t_aligned)
    
    speed_interp = {}
    for col in ['VehicleSpeed_kmh']:
        f_interp = interp1d(speed_df['Time_s'], speed_df[col], kind='cubic', fill_value='extrapolate')
        speed_interp[col] = f_interp(t_aligned)
    
    gps_interp = {}
    for col in ['Latitude_deg', 'Longitude_deg', 'Height_m']:
        f_interp = interp1d(gps_df['Time_s'], gps_df[col], kind='cubic', fill_value='extrapolate')
        gps_interp[col] = f_interp(t_aligned)
    
    # 构造对齐的DataFrame
    aligned_df = pd.DataFrame({
        'Time_s': t_aligned,
        'AccX_g': imu_interp['AccX_g'],
        'AccY_g': imu_interp['AccY_g'],
        'AccZ_g': imu_interp['AccZ_g'],
        'GyroX_degs': imu_interp['GyroX_degs'],
        'GyroY_degs': imu_interp['GyroY_degs'],
        'GyroZ_degs': imu_interp['GyroZ_degs'],
        'VehicleSpeed_kmh': speed_interp['VehicleSpeed_kmh'],
        'Latitude_deg': gps_interp['Latitude_deg'],
        'Longitude_deg': gps_interp['Longitude_deg'],
        'Height_m': gps_interp['Height_m'],
    })
    
    print(f"[OK] 数据对齐完成: {len(aligned_df)} 个时间步")
    return aligned_df


# ============================================================================
# 物理航位推算 (Dead Reckoning)
# ============================================================================

def dead_reckoning(df, dt):
    """
    基于车速和陀螺仪角速度计算理论位移增量
    
    参数：
        df: 对齐后的DataFrame
        dt: 时间间隔 (s)
    
    返回：
        df: 添加了Base_dx, Base_dy列的DataFrame
    """
    print("[Dead Reckoning] 计算物理航位推算...")
    
    # 1. 车速转换: km/h → m/s
    v_ms = df['VehicleSpeed_kmh'].values / 3.6
    
    # 2. 航向角：通过Z轴陀螺积分
    heading_rad = np.cumsum(df['GyroZ_degs'].values * DEGREES_TO_RADIANS * dt)
    heading_rad = np.insert(heading_rad[:-1], 0, 0)  # 初始航向为0
    
    # 3. 位移增量 (每个时间步)
    dx = v_ms * np.cos(heading_rad) * dt
    dy = v_ms * np.sin(heading_rad) * dt
    
    df['Heading_rad'] = heading_rad
    df['Heading_deg'] = heading_rad * RADIANS_TO_DEGREES
    df['VehicleSpeed_ms'] = v_ms
    df['Base_dx'] = dx
    df['Base_dy'] = dy
    
    print(f"  航向角范围: {heading_rad.min()*RADIANS_TO_DEGREES:.2f}° ~ {heading_rad.max()*RADIANS_TO_DEGREES:.2f}°")
    print(f"  速度范围: {v_ms.min():.3f} ~ {v_ms.max():.3f} m/s")
    print(f"  单步位移范围 (dx): {dx.min():.6f} ~ {dx.max():.6f} m")
    print(f"  单步位移范围 (dy): {dy.min():.6f} ~ {dy.max():.6f} m")
    
    return df


# ============================================================================
# GPS → ENU坐标转换 & 真实标签构造
# ============================================================================

def gps_to_enu_trajectory(df):
    """
    将GPS经纬度转换为ENU坐标，计算真实位移增量
    
    参数：
        df: 包含GPS数据的DataFrame
    
    返回：
        df: 添加了ENU_x, ENU_y, True_dx, True_dy列的DataFrame
    """
    print("[GPS to ENU] 坐标系转换...")
    
    # 选择第一个有效GPS点作为参考点
    ref_lat = df['Latitude_deg'].iloc[0]
    ref_lon = df['Longitude_deg'].iloc[0]
    ref_alt = df['Height_m'].iloc[0]
    
    print(f"  参考点: Lat={ref_lat:.6f}°, Lon={ref_lon:.6f}°, Alt={ref_alt:.2f}m")
    
    # 批量转换所有点
    enu_x = np.zeros(len(df))
    enu_y = np.zeros(len(df))
    
    for i in range(len(df)):
        east, north, up = wgs84_to_enu(
            df['Latitude_deg'].iloc[i],
            df['Longitude_deg'].iloc[i],
            df['Height_m'].iloc[i],
            ref_lat, ref_lon, ref_alt
        )
        enu_x[i] = east
        enu_y[i] = north
    
    df['ENU_x'] = enu_x
    df['ENU_y'] = enu_y
    
    # 计算真实位移增量
    true_dx = np.diff(enu_x, prepend=enu_x[0])
    true_dy = np.diff(enu_y, prepend=enu_y[0])
    
    df['True_dx'] = true_dx
    df['True_dy'] = true_dy
    
    print(f"  ENU坐标范围 X: {enu_x.min():.2f} ~ {enu_x.max():.2f} m")
    print(f"  ENU坐标范围 Y: {enu_y.min():.2f} ~ {enu_y.max():.2f} m")
    print(f"  真实位移范围 (dx): {true_dx.min():.6f} ~ {true_dx.max():.6f} m")
    print(f"  真实位移范围 (dy): {true_dy.min():.6f} ~ {true_dy.max():.6f} m")
    
    return df


# ============================================================================
# 残差目标构造
# ============================================================================

def compute_residual_targets(df):
    """
    计算模型的学习目标：残差 = 真实值 - 推算值
    
    参数：
        df: 包含True/Base位移的DataFrame
    
    返回：
        df: 添加了Target_dx, Target_dy列的DataFrame
    """
    print("[Residual] 构建残差学习目标...")
    
    df['Target_dx'] = df['True_dx'] - df['Base_dx']
    df['Target_dy'] = df['True_dy'] - df['Base_dy']
    
    print(f"  残差 (dx) 范围: {df['Target_dx'].min():.6f} ~ {df['Target_dx'].max():.6f} m")
    print(f"  残差 (dy) 范围: {df['Target_dy'].min():.6f} ~ {df['Target_dy'].max():.6f} m")
    print(f"  残差 (dx) 统计: μ={df['Target_dx'].mean():.6f}, σ={df['Target_dx'].std():.6f}")
    print(f"  残差 (dy) 统计: μ={df['Target_dy'].mean():.6f}, σ={df['Target_dy'].std():.6f}")
    
    return df


# ============================================================================
# 特征归一化
# ============================================================================

def normalize_features(df):
    """
    对输入特征进行Z-score标准化
    
    参数：
        df: DataFrame
    
    返回：
        df: 包含归一化特征的DataFrame
        stats: 归一化统计信息 (用于推理时逆标准化)
    """
    print("[Normalization] 特征归一化...")
    
    feature_cols = [
        'AccX_g', 'AccY_g', 'AccZ_g',
        'GyroX_degs', 'GyroY_degs', 'GyroZ_degs',
        'VehicleSpeed_ms',
        'Base_dx', 'Base_dy'
    ]
    
    stats = {}
    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std() + 1e-6  # 避免除零
        stats[col] = {'mean': mean, 'std': std}
        df[f'{col}_norm'] = (df[col] - mean) / std
    
    for col in feature_cols:
        print(f"  {col}: μ={stats[col]['mean']:.6f}, σ={stats[col]['std']:.6f}")
    
    return df, stats


# ============================================================================
# 滑动窗口切分
# ============================================================================

def create_windowed_dataset(df, window_size, stride):
    """
    从时间序列数据创建滑动窗口样本集
    
    参数：
        df: DataFrame
        window_size: 窗口大小 (时间步数)
        stride: 滑动步长
    
    返回：
        X: 输入特征 (N, window_size, num_features)
        Y: 目标残差 (N, 2) - [Target_dx, Target_dy]
        timestamps: 每个样本的对应时间戳
    """
    print(f"[Windowing] 创建滑动窗口数据集 (window_size={window_size}, stride={stride})...")
    
    feature_cols_norm = [
        'AccX_g_norm', 'AccY_g_norm', 'AccZ_g_norm',
        'GyroX_degs_norm', 'GyroY_degs_norm', 'GyroZ_degs_norm',
        'VehicleSpeed_ms_norm',
        'Base_dx_norm', 'Base_dy_norm'
    ]
    
    X_list = []
    Y_list = []
    timestamps = []
    
    num_steps = len(df)
    
    for i in range(0, num_steps - window_size, stride):
        window_data = df[feature_cols_norm].iloc[i:i+window_size].values
        
        # 目标：窗口最后一个时间步的残差
        target_idx = i + window_size - 1
        target = df[['Target_dx', 'Target_dy']].iloc[target_idx].values
        
        X_list.append(window_data)
        Y_list.append(target)
        timestamps.append(df['Time_s'].iloc[target_idx])
    
    X = np.array(X_list, dtype=np.float32)  # (N, window_size, num_features)
    Y = np.array(Y_list, dtype=np.float32)  # (N, 2)
    timestamps = np.array(timestamps, dtype=np.float32)
    
    print(f"  创建样本数: {len(X)}")
    print(f"  X形状: {X.shape}")
    print(f"  Y形状: {Y.shape}")
    print(f"  特征列数: {len(feature_cols_norm)}")
    print(f"  特征名称: {feature_cols_norm}")
    
    return X, Y, timestamps, feature_cols_norm


# ============================================================================
# 多数据集加载 (Data01 + Data02)
# ============================================================================

def load_multiple_datasets(data_dir, target_freq=10):
    """
    加载多个数据集（Data01 和 Data02），合并成一个大数据集
    
    参数：
        data_dir: 数据目录路径
        target_freq: 目标频率 (Hz)
    
    返回：
        combined_df: 合并后的对齐DataFrame
    """
    print("\n" + "="*80)
    print("[Multi-Dataset] 正在加载多个数据集...")
    print("="*80)
    
    all_dataframes = []
    
    # 尝试加载 Data01 和 Data02
    for dataset_id in ["Data01", "Data02"]:
        print(f"\n[{dataset_id}] 处理中...")
        
        imu_file = data_dir / f"{dataset_id}_IMU.txt"
        speed_file = data_dir / f"{dataset_id}_VehicleSpeed.txt"
        gps_file = data_dir / f"{dataset_id}_GNSS.txt"
        
        # 检查文件是否存在
        if not imu_file.exists():
            print(f"  [WARN] {dataset_id}_IMU.txt 不存在，跳过此数据集")
            continue
        
        try:
            # 临时修改全局变量来区分数据集
            # 加载单个数据集
            
            # 加载IMU数据
            imu_df = pd.read_csv(imu_file, sep='\t', skipinitialspace=True, encoding='utf-8-sig')
            imu_df.columns = clean_column_names(imu_df.columns)
            
            # 加载车速数据
            speed_df = pd.read_csv(speed_file, sep='\t', skipinitialspace=True, encoding='utf-8-sig')
            speed_df.columns = clean_column_names(speed_df.columns)
            
            # 加载GPS数据
            gps_df = pd.read_csv(gps_file, sep='\t', skipinitialspace=True, encoding='utf-8-sig')
            gps_df.columns = clean_column_names(gps_df.columns)
            
            print(f"  [OK] IMU: {len(imu_df)} 条")
            print(f"  [OK] Speed: {len(speed_df)} 条")
            print(f"  [OK] GPS: {len(gps_df)} 条")
            
            # 确定统一的时间范围
            t_start = max(imu_df['Time_s'].min(), speed_df['Time_s'].min(), gps_df['Time_s'].min())
            t_end = min(imu_df['Time_s'].max(), speed_df['Time_s'].max(), gps_df['Time_s'].max())
            
            # 生成目标时间戳
            target_dt = 1.0 / target_freq
            t_aligned = np.arange(t_start, t_end, target_dt)
            
            # 插值各传感器数据
            imu_interp = {}
            for col in ['AccX_g', 'AccY_g', 'AccZ_g', 'GyroX_degs', 'GyroY_degs', 'GyroZ_degs']:
                f_interp = interp1d(imu_df['Time_s'], imu_df[col], kind='cubic', fill_value='extrapolate')
                imu_interp[col] = f_interp(t_aligned)
            
            speed_interp = {}
            for col in ['VehicleSpeed_kmh']:
                f_interp = interp1d(speed_df['Time_s'], speed_df[col], kind='cubic', fill_value='extrapolate')
                speed_interp[col] = f_interp(t_aligned)
            
            gps_interp = {}
            for col in ['Latitude_deg', 'Longitude_deg', 'Height_m']:
                f_interp = interp1d(gps_df['Time_s'], gps_df[col], kind='cubic', fill_value='extrapolate')
                gps_interp[col] = f_interp(t_aligned)
            
            # 构建对齐后的DataFrame
            aligned_df = pd.DataFrame({
                'Time_s': t_aligned,
                **imu_interp,
                **speed_interp,
                **gps_interp
            })
            
            print(f"  [OK] 对齐完成: {len(aligned_df)} 条记录")
            all_dataframes.append(aligned_df)
            
        except Exception as e:
            print(f"  [FAIL] {dataset_id} 处理失败: {str(e)}")
            continue
    
    # 合并所有数据集
    if len(all_dataframes) == 0:
        raise RuntimeError("[FAIL] 没有成功加载任何数据集！请检查文件路径。")
    
    if len(all_dataframes) == 1:
        combined_df = all_dataframes[0]
        print(f"\n[OK] 仅加载了1个数据集")
    else:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"\n[OK] 成功合并 {len(all_dataframes)} 个数据集")
    
    print(f"  总记录数: {len(combined_df)}")
    print(f"  时间跨度: {combined_df['Time_s'].max() - combined_df['Time_s'].min():.1f} 秒")
    print(f"  内存占用: {combined_df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    return combined_df


# ============================================================================
# 主流程
# ============================================================================

def main():
    print("="*80)
    print("IMU + GPS 多模态数据预处理流程 (支持多数据集)")
    print("="*80)
    
    # 1. 加载并对齐数据（支持Data01 + Data02）
    df = load_multiple_datasets(DATA_DIR, target_freq=TARGET_FREQ)
    print()
    
    # 2. 物理航位推算
    df = dead_reckoning(df, TARGET_DT)
    print()
    
    # 3. GPS转ENU & 真实标签
    df = gps_to_enu_trajectory(df)
    print()
    
    # 4. 残差目标
    df = compute_residual_targets(df)
    print()
    
    # 5. 特征归一化
    df, stats = normalize_features(df)
    print()
    
    # 6. 滑动窗口切分
    X, Y, timestamps, feature_names = create_windowed_dataset(
        df, 
        window_size=WINDOW_SIZE,
        stride=WINDOW_STRIDE
    )
    print()
    
    # 7. 保存预处理结果
    print("[Saving] 保存预处理数据...")
    np.save(OUTPUT_DIR / "X_train.npy", X)
    np.save(OUTPUT_DIR / "Y_train.npy", Y)
    np.save(OUTPUT_DIR / "timestamps.npy", timestamps)
    
    # 保存完整DataFrame（用于分析）
    df.to_csv(OUTPUT_DIR / "aligned_data.csv", index=False)
    
    # 保存统计信息
    import json
    with open(OUTPUT_DIR / "normalization_stats.json", 'w') as f:
        # 转换为可序列化的格式
        stats_serializable = {k: {'mean': float(v['mean']), 'std': float(v['std'])} 
                             for k, v in stats.items()}
        json.dump({
            'stats': stats_serializable,
            'feature_names': feature_names,
            'window_size': WINDOW_SIZE,
            'target_freq': TARGET_FREQ,
        }, f, indent=2)
    
    print(f"  [OK] X_train.npy: {X.shape}")
    print(f"  [OK] Y_train.npy: {Y.shape}")
    print(f"  [OK] timestamps.npy: {timestamps.shape}")
    print(f"  [OK] aligned_data.csv")
    print(f"  [OK] normalization_stats.json")
    
    print()
    print("="*80)
    print("数据预处理完成！")
    print("="*80)
    
    # 打印数据统计
    print("\n[Summary] 数据集统计:")
    print(f"  样本数: {len(X)}")
    print(f"  输入形状: {X.shape} (Batch, Timesteps, Features)")
    print(f"  目标形状: {Y.shape} (Batch, 2)")
    print(f"  时间覆盖: {timestamps[0]:.3f} ~ {timestamps[-1]:.3f} s")
    print(f"  目标残差统计:")
    print(f"    dx: μ={Y[:, 0].mean():.6f}, σ={Y[:, 0].std():.6f}, "
          f"min={Y[:, 0].min():.6f}, max={Y[:, 0].max():.6f}")
    print(f"    dy: μ={Y[:, 1].mean():.6f}, σ={Y[:, 1].std():.6f}, "
          f"min={Y[:, 1].min():.6f}, max={Y[:, 1].max():.6f}")
    
    return X, Y, timestamps, df, stats, feature_names


if __name__ == "__main__":
    X, Y, timestamps, df, stats, feature_names = main()
