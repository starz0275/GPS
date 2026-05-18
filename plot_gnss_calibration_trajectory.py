"""画标定实车数据 Data01_GNSS 的经纬度轨迹"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

DATA_FILE = Path(r'C:\Users\nxj\Desktop\GPS\标定实车数据\Data02_GNSS.txt')
OUT_DIR = Path(__file__).parent / 'trained_models'
OUT_PNG = OUT_DIR / 'gnss_calibration_trajectory2.png'


def main():
    # 读取 tab 分隔的数据，跳过 BOM
    data = np.loadtxt(DATA_FILE, delimiter='\t', skiprows=1, encoding='utf-8-sig')
    lon = data[:, 2]  # Longitude_deg
    lat = data[:, 1]  # Latitude_deg

    # 过滤掉 GPS 速度为 0 的静止点
    gps_spd = data[:, 11]
    moving = gps_spd > 0.01

    print(f'总点数: {len(lat)}, 运动中点数: {moving.sum()}')

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- 左侧：全部轨迹 ---
    ax = axes[0]
    ax.plot(lat, lon, 'b.-', ms=1, lw=0.8, label=f'All points ({len(lat)})')
    ax.set_xlabel('Latitude (deg)')
    ax.set_ylabel('Longitude (deg)')
    ax.set_title('GNSS Trajectory — All Points')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)
    ax.set_aspect(1.0 / np.cos(np.radians(np.mean(lat))))  # 等比例

    # --- 右侧：仅运动轨迹 ---
    ax = axes[1]
    if moving.any():
        ax.plot(lat[moving], lon[moving], 'r.-', ms=1.5, lw=0.8,
                label=f'Moving ({moving.sum()} pts)')
        # 起终点标记
        move_idx = np.where(moving)[0]
        ax.scatter(lat[move_idx[0]], lon[move_idx[0]], c='green', s=80,
                   marker='o', zorder=5, label='Start', edgecolors='black', linewidth=0.5)
        ax.scatter(lat[move_idx[-1]], lon[move_idx[-1]], c='purple', s=80,
                   marker='s', zorder=5, label='End', edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Latitude (deg)')
    ax.set_ylabel('Longitude (deg)')
    ax.set_title('GNSS Trajectory — Moving Only')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)
    ax.set_aspect(1.0 / np.cos(np.radians(np.mean(lat))))

    fig.suptitle('Data01 GNSS Calibration Trajectory', fontsize=14, fontweight='bold')
    plt.tight_layout()

    OUT_DIR.mkdir(exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'\n[OK] {OUT_PNG}')

    # 统计
    lon_range = lon.max() - lon.min()
    lat_range = lat.max() - lat.min()
    mid_lat = np.mean(lat)
    dx = lon_range * 111320 * np.cos(np.radians(mid_lat))
    dy = lat_range * 111320
    print(f'  经度范围: {lon_range:.6f} deg (~{dx:.1f} m)')
    print(f'  纬度范围: {lat_range:.6f} deg (~{dy:.1f} m)')
    print(f'  总跨度: ~{np.hypot(dx, dy):.1f} m')


if __name__ == '__main__':
    main()
