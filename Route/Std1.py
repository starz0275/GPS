import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_gps_trajectory(file_path):
    """
    读取GNSS数据文件并绘制轨迹图
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 - {file_path}")
        return

    try:
        # --- 关键步骤：根据实际文件格式调整读取参数 ---
        # 假设1: 文件是空格或Tab分隔，没有表头，第1列是经度，第2列是纬度
        # 假设2: 如果有表头，设置 header=0
        # 假设3: 如果是逗号分隔，设置 sep=','
        
        # 这里尝试自动推断，通常 GNSS 原始数据可能是空格分隔
        # 请根据实际数据修改 sep 和 header 参数
        df = pd.read_csv(file_path, sep='\s+', header=None, engine='python')
        
        # 假设前两个数值列分别是经度和纬度
        # 注意：GNSS数据中通常顺序是 [时间, 纬度, 经度, ...] 或 [经度, 纬度, ...]
        # 请务必确认哪一列是经度(Lon)，哪一列是纬度(Lat)
        # 下面假设第0列是经度，第1列是纬度。如果不对，请交换 df[0] 和 df[1]
        lon_col = 0 
        lat_col = 1
        
        # 如果数据中有非数值行（如表头或异常值），强制转换并丢弃错误
        df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
        df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
        
        # 删除包含 NaN 的行
        df_clean = df.dropna(subset=[lon_col, lat_col])
        
        if df_clean.empty:
            print("错误: 未找到有效的经纬度数据。请检查文件格式和列索引。")
            return

        lons = df_clean[lon_col]
        lats = df_clean[lat_col]

        # 绘图
        plt.figure(figsize=(12, 8))
        plt.plot(lons, lats, 'b-', linewidth=1, label='GPS Trajectory')
        
        # 标记起点和终点
        plt.plot(lons.iloc[0], lats.iloc[0], 'go', markersize=8, label='Start')
        plt.plot(lons.iloc[-1], lats.iloc[-1], 'ro', markersize=8, label='End')
        
        plt.title('GPS Trajectory from Data01_GNSS.txt')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 保持纵横比一致，避免轨迹变形
        plt.axis('equal')
        
        plt.tight_layout()
        plt.show()
        
        print(f"成功绘制轨迹，共 {len(df_clean)} 个点。")

    except Exception as e:
        print(f"发生错误: {e}")
        print("提示: 请检查文件分隔符、列顺序以及是否包含表头。")

# 使用你的文件路径
file_path = r'C:\Users\nxj\Desktop\GPS\标定实车数据\Data01_GNSS.txt'
plot_gps_trajectory(file_path)