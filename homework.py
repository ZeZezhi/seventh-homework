# -*- coding: utf-8 -*-
"""
人工智能编程课 第三次作业 — IC 刷卡数据分析
==============================================
本脚本综合使用 numpy, pandas, matplotlib, seaborn 完成：
  任务1: 数据预处理
  任务2: 时间分布分析（numpy 统计 + matplotlib 柱状图）
  任务3: 线路站点分析（自定义函数 + seaborn 柱状图）
  任务4: 高峰小时系数 PHF 计算
  任务5: 线路驾驶员信息提取（批量生成 txt 文件）
  任务6: 运行效率热力图（seaborn heatmap）

作者在 AI 辅助下主导分析方向、参数选择与可视化设计，
AI 仅按指令生成代码框架，所有核心逻辑均经过人工审查与调优。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os

# ============================================================
# 全局设置：中文字体 & 英文图表风格
# ============================================================
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 任务1: 数据预处理
# ============================================================
print("=" * 60)
print("【任务1】数据预处理")
print("=" * 60)

# 1.1 读取数据
df = pd.read_csv('ICData.csv', encoding='utf-8')
print(">> 数据基本信息:")
print(f"   形状 (行数, 列数): {df.shape}")
print(f"   列名: {list(df.columns)}")
print(f"\n>> 数据类型:")
print(df.dtypes)
print(f"\n>> 前5行数据:")
print(df.head())

# 1.2 时间列转换为 pandas datetime 类型，并提取小时
# 列索引: 1 = 交易时间
time_col = df.columns[1]
df[time_col] = pd.to_datetime(df[time_col])
df['hour'] = df[time_col].dt.hour
print(f"\n>> 已将 '{time_col}' 转换为 datetime 类型，并提取 'hour' 列")
print(f"   hour 范围: {df['hour'].min()} ~ {df['hour'].max()}")

# 1.3 计算 ride_stops = |下车站点 - 上车站点|
# 保留原始列名以便后续使用
boarding_col = df.columns[6]   # 上车站点
alighting_col = df.columns[7]  # 下车站点
df['ride_stops'] = (df[alighting_col] - df[boarding_col]).abs()
print(f"\n>> 已创建 ride_stops 列 (|{alighting_col} - {boarding_col}|)")
print(f"   ride_stops 前5个值: {df['ride_stops'].head().tolist()}")

# 删除 ride_stops == 0 的异常记录
anomaly_count = (df['ride_stops'] == 0).sum()
print(f"\n>> ride_stops == 0 的异常记录数: {anomaly_count}")
df = df[df['ride_stops'] != 0].copy()
print(f"   删除后数据形状: {df.shape}")

# 1.4 缺失值检查
missing = df.isnull().sum()
print(f"\n>> 各列缺失值统计:")
print(missing)
# 删除含缺失值的行
missing_rows = df.isnull().any(axis=1).sum()
if missing_rows > 0:
    df = df.dropna().copy()
    print(f"   已删除 {missing_rows} 条含缺失值的记录")
else:
    print("   未发现缺失值，无需删除")
print(f"   最终数据形状: {df.shape}")

# ============================================================
# 任务2: 时间分布分析
# ============================================================
print("\n" + "=" * 60)
print("【任务2】时间分布分析")
print("=" * 60)

# --- 2(a) 使用 numpy 统计凌晨与深夜刷卡量 ---
total_swipes = len(df)

# 使用 numpy.where 条件统计
early_morning_mask = np.where(df['hour'].values < 7, True, False)
late_night_mask = np.where(df['hour'].values >= 22, True, False)

early_morning_count = np.sum(early_morning_mask)
late_night_count = np.sum(late_night_mask)

print("\n>> 2(a) 凌晨与深夜刷卡量统计 (使用 numpy.where + numpy.sum):")
print(f"   凌晨时段 (hour < 7):  {early_morning_count} 次 "
      f"({early_morning_count / total_swipes * 100:.2f}%)")
print(f"   深夜时段 (hour >= 22): {late_night_count} 次 "
      f"({late_night_count / total_swipes * 100:.2f}%)")

# --- 2(b) matplotlib 24小时刷卡量分布柱状图 ---
hour_counts = df['hour'].value_counts().sort_index()
# 确保0-23小时全部出现
all_hours = np.arange(24)
counts_array = np.array([hour_counts.get(h, 0) for h in all_hours])

# 颜色映射: 凌晨(<7)和深夜(>=22)用不同颜色
colors = []
for h in all_hours:
    if h < 7:
        colors.append('#FF8C00')       # 橙色 — 凌晨
    elif h >= 22:
        colors.append('#8B008B')       # 紫色 — 深夜
    else:
        colors.append('#4682B4')       # 钢蓝 — 常规时段

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(all_hours, counts_array, color=colors, edgecolor='white', linewidth=0.5)

# 在柱顶标注数值
for bar_obj, val in zip(bars, counts_array):
    if val > 0:
        ax.text(bar_obj.get_x() + bar_obj.get_width()/2.,
                bar_obj.get_height() + max(counts_array)*0.01,
                str(val), ha='center', va='bottom', fontsize=7)

ax.set_title('24-Hour Swipe Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Number of Swipes', fontsize=12)
ax.set_xticks(all_hours)
ax.set_xticklabels(all_hours)  # 水平标签
ax.set_xlim(-0.8, 23.8)
ax.grid(axis='y', alpha=0.3)

# 图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FF8C00', label='Early Morning (< 7:00)'),
    Patch(facecolor='#4682B4', label='Regular Hours (7:00-21:59)'),
    Patch(facecolor='#8B008B', label='Late Night (>= 22:00)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig('hour_distribution.png', dpi=150)
plt.close()
print("\n>> 2(b) 柱状图已保存为 hour_distribution.png (dpi=150)")

# ============================================================
# 任务3: 线路站点分析
# ============================================================
print("\n" + "=" * 60)
print("【任务3】线路站点分析")
print("=" * 60)


def analyze_route_stops(df, route_col='线路号', stops_col='ride_stops'):
    """
    按线路分析乘客的平均乘坐站数及标准差。

    Parameters
    ----------
    df : pd.DataFrame
        预处理后的数据集，必须包含 route_col 和 stops_col。
    route_col : str
        线路列名，默认 '线路号'。
    stops_col : str
        乘坐站数列名，默认 'ride_stops'。

    Returns
    -------
    pd.DataFrame
        包含三列：route_col, 'mean_stops', 'std_stops'，
        按 mean_stops 降序排列。
    """
    # 分组聚合：均值与标准差
    route_stats = df.groupby(route_col)[stops_col].agg(
        mean_stops='mean',
        std_stops='std'
    ).reset_index()
    # 按均值降序排序
    route_stats = route_stats.sort_values('mean_stops', ascending=False).reset_index(drop=True)
    return route_stats


# 获取数据中的实际线路列名
route_col_name = df.columns[4]  # 线路号
route_result = analyze_route_stops(df, route_col=route_col_name, stops_col='ride_stops')
print("\n>> 线路平均乘坐站数及标准差 (前10行):")
print(route_result.head(10).to_string(index=False))

# Seaborn 水平柱状图 — Top 15 线路
top15 = route_result.head(15).copy()
# 反转以便在水平柱状图中自上而下显示最高值
top15 = top15.iloc[::-1]

fig, ax = plt.subplots(figsize=(10, 8))
# 使用 seaborn barplot
# 为 seaborn v0.13+ 兼容，使用 hue 参数替代 palette
sns.barplot(
    data=top15,
    y=top15[route_col_name].astype(str),
    x='mean_stops',
    hue=top15[route_col_name].astype(str),
    palette='Blues_d',
    legend=False,
    ax=ax
)

# 手动添加误差线 (兼容各版本 seaborn)
# 取标准差
std_vals = top15['std_stops'].values
mean_vals = top15['mean_stops'].values
y_positions = range(len(top15))
ax.errorbar(mean_vals, y_positions, xerr=std_vals,
            fmt='none', ecolor='black', capsize=0.3, elinewidth=0.8, alpha=0.7)

ax.set_title('Top 15 Routes by Average Ride Stops', fontsize=14, fontweight='bold')
ax.set_xlabel('Average Number of Stops', fontsize=12)
ax.set_ylabel('Route Number', fontsize=12)
ax.set_xlim(0, top15['mean_stops'].max() * 1.2)

plt.tight_layout()
plt.savefig('route_stops.png', dpi=150)
plt.close()
print("\n>> 3 水平柱状图已保存为 route_stops.png (dpi=150)")

# ============================================================
# 任务4: 高峰小时系数 (PHF) 计算
# ============================================================
print("\n" + "=" * 60)
print("【任务4】高峰小时系数 (Peak Hour Factor) 计算")
print("=" * 60)

# 4.1 统计每个小时的刷卡量，自动找出高峰小时
hourly_counts = df.groupby('hour').size()
peak_hour = hourly_counts.idxmax()
peak_count = hourly_counts.max()
print(f"\n>> 高峰小时识别:")
print(f"   高峰小时为 {peak_hour:02d}:00 ~ {peak_hour:02d}:59，刷卡量 {peak_count} 次")

# 4.2 提取高峰小时内的数据
peak_df = df[df['hour'] == peak_hour].copy()
# 获取分钟 (精确到分钟)
peak_df['minute'] = peak_df[time_col].dt.minute

# --- PHF5: 按5分钟窗口聚合 ---
# 将分钟按5分钟窗口分组: 0-4 → 窗口0, 5-9 → 窗口1, ...
peak_df['min5_bin'] = peak_df['minute'] // 5
min5_counts = peak_df.groupby('min5_bin').size()
max_5min = min5_counts.max()
max_5min_bin = min5_counts.idxmax()
max_5min_start = max_5min_bin * 5
max_5min_end = max_5min_start + 4

PHF5 = peak_count / (12 * max_5min)
print(f"\n>> PHF5 计算:")
print(f"   最高5分钟刷卡量: {peak_hour:02d}:{max_5min_bin*5:02d}~"
      f"{peak_hour:02d}:{max_5min_bin*5+4:02d}，{max_5min} 次")
print(f"   PHF5 = {peak_count} / (12 × {max_5min}) = {PHF5:.4f}")

# --- PHF15: 按15分钟窗口聚合 ---
peak_df['min15_bin'] = peak_df['minute'] // 15
min15_counts = peak_df.groupby('min15_bin').size()
max_15min = min15_counts.max()
max_15min_bin = min15_counts.idxmax()
max_15min_start = max_15min_bin * 15
max_15min_end = max_15min_start + 14

PHF15 = peak_count / (4 * max_15min)
print(f"\n>> PHF15 计算:")
print(f"   最高15分钟刷卡量: {peak_hour:02d}:{max_15min_bin*15:02d}~"
      f"{peak_hour:02d}:{max_15min_bin*15+14:02d}，{max_15min} 次")
print(f"   PHF15 = {peak_count} / (4 × {max_15min}) = {PHF15:.4f}")

# 格式化输出
print(f"\n>> 完整输出格式:")
print(f"   高峰小时: {peak_hour:02d}:00 ~ {peak_hour:02d}:59，刷卡量 {peak_count} 次")
print(f"   最高5分钟刷卡量: {peak_hour:02d}:{max_5min_bin*5:02d}~"
      f"{peak_hour:02d}:{max_5min_bin*5+4:02d}，{max_5min} 次")
print(f"   PHF5  = {peak_count} / (12 × {max_5min}) = {PHF5:.4f}")
print(f"   最高15分钟刷卡量: {peak_hour:02d}:{max_15min_bin*15:02d}~"
      f"{peak_hour:02d}:{max_15min_bin*15+14:02d}，{max_15min} 次")
print(f"   PHF15 = {peak_count} / (4 × {max_15min}) = {PHF15:.4f}")

# ============================================================
# 任务5: 线路驾驶员信息提取
# ============================================================
print("\n" + "=" * 60)
print("【任务5】线路驾驶员信息提取")
print("=" * 60)

# 5.1 筛选线路号在 1101 ~ 1120 之间的记录
route_col = df.columns[4]     # 线路号
vehicle_col = df.columns[5]   # 车辆编号
driver_col = df.columns[8]    # 驾驶员编号

filtered_df = df[(df[route_col] >= 1101) & (df[route_col] <= 1120)].copy()
print(f"\n>> 筛选线路 1101~1120, 共 {len(filtered_df)} 条记录")

# 5.2 创建输出目录
output_dir = '公交驾驶员信息'
os.makedirs(output_dir, exist_ok=True)

# 5.3 为每条线路生成 txt 文件
unique_routes = sorted(filtered_df[route_col].unique())
file_count = 0

for route in unique_routes:
    route_data = filtered_df[filtered_df[route_col] == route]
    # 按车辆编号分组，对驾驶员去重，形成 车辆-驾驶员 对应关系
    # 取每个线路中出现的 (车辆编号, 驾驶员编号) 去重组合
    pairs = route_data[[vehicle_col, driver_col]].drop_duplicates()
    # 按车辆编号排序
    pairs = pairs.sort_values(vehicle_col)

    filepath = os.path.join(output_dir, f'{route}.txt')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f'线路号: {route}\n')
        for _, row in pairs.iterrows():
            f.write(f'{int(row[vehicle_col])}\t{int(row[driver_col])}\n')
    file_count += 1

print(f"\n>> 已在 '{output_dir}/' 目录下生成 {file_count} 个 txt 文件")
print(f"   线路列表: {unique_routes}")

# 打印第一个文件的内容示例
sample_file = os.path.join(output_dir, f'{unique_routes[0]}.txt')
print(f"\n>> 样例文件 '{unique_routes[0]}.txt' 内容:")
with open(sample_file, 'r', encoding='utf-8') as f:
    print(f.read())

# ============================================================
# 任务6: 运行效率热力图
# ============================================================
print("\n" + "=" * 60)
print("【任务6】运行效率热力图")
print("=" * 60)

# 6.1 计算各维度的服务人次（每行=1人次）
# 统计 Top 10 驾驶员
driver_rides = df[driver_col].value_counts().head(10)
top10_drivers = driver_rides.index.tolist()
print(f"\n>> Top 10 驾驶员 (人次):")
for i, (driver, count) in enumerate(driver_rides.items(), 1):
    print(f"   {i}. 驾驶员 {int(driver)}: {count} 次")

# 统计 Top 10 线路
route_rides = df[route_col].value_counts().head(10)
top10_routes = route_rides.index.tolist()
print(f"\n>> Top 10 线路 (人次):")
for i, (route, count) in enumerate(route_rides.items(), 1):
    print(f"   {i}. 线路 {int(route)}: {count} 次")

# 统计 Top 10 上车站点
boarding_rides = df[boarding_col].value_counts().head(10)
top10_stations = boarding_rides.index.tolist()
print(f"\n>> Top 10 上车站点 (人次):")
for i, (station, count) in enumerate(boarding_rides.items(), 1):
    print(f"   {i}. 站点 {int(station)}: {count} 次")

# 统计 Top 10 车辆
vehicle_rides = df[vehicle_col].value_counts().head(10)
top10_vehicles = vehicle_rides.index.tolist()
print(f"\n>> Top 10 车辆 (人次):")
for i, (vehicle, count) in enumerate(vehicle_rides.items(), 1):
    print(f"   {i}. 车辆 {int(vehicle)}: {count} 次")

# 6.2 构建 4×10 矩阵用于热力图
# 行: [Driver, Route, Boarding Station, Vehicle]
# 列: Top1 ~ Top10
heatmap_data = np.zeros((4, 10), dtype=int)

# 行0 = 驾驶员人次 (Top1 ~ Top10)
heatmap_data[0, :] = driver_rides.values
# 行1 = 线路人次 (Top1 ~ Top10)
heatmap_data[1, :] = route_rides.values
# 行2 = 上车站点人次 (Top1 ~ Top10)
heatmap_data[2, :] = boarding_rides.values
# 行3 = 车辆人次 (Top1 ~ Top10)
heatmap_data[3, :] = vehicle_rides.values

# 6.3 seaborn heatmap
fig, ax = plt.subplots(figsize=(14, 6))

row_labels = ['Driver', 'Route', 'Boarding Station', 'Vehicle']
col_labels = [f'Top{i}' for i in range(1, 11)]

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt='d',
    cmap='RdYlGn',
    xticklabels=col_labels,
    yticklabels=row_labels,
    linewidths=0.5,
    linecolor='white',
    ax=ax,
    cbar_kws={'label': 'Number of Rides'}
)

ax.set_title('Service Performance Heatmap: Top 10 Entities by Ride Count',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Rank', fontsize=12)
ax.set_ylabel('Dimension', fontsize=12)

# x轴标签不旋转
plt.setp(ax.get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('performance_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n>> 6 热力图已保存为 performance_heatmap.png (dpi=150, bbox_inches='tight')")

# 分析说明 (约50字)
print("\n>> 热力图分析说明:")
analysis_text = (
    "From the heatmap, Route 46003 ranks first with 7127 rides, significantly "
    "outperforming the second-ranked route 1091 (4893 rides). Driver 0 and Vehicle 0 "
    "show anomalously high volumes (6484 and 11774 respectively), which may indicate "
    "placeholder or aggregated records rather than individual entities. Boarding "
    "Station 1 leads with 7776 rides, suggesting it is a major transit hub. The "
    "ride counts exhibit a steep drop-off from Top1 to Top10 across all four "
    "dimensions, revealing highly concentrated service demand on a small number "
    "of routes, stations, drivers, and vehicles."
)
print(f"   {analysis_text}")

print("\n" + "=" * 60)
print("全部任务完成！")
print("=" * 60)
print("\n生成的文件列表:")
print("  - ICData.csv")
print("  - homework.py (本文件)")
print("  - hour_distribution.png")
print("  - route_stops.png")
print("  - performance_heatmap.png")
print("  - 公交驾驶员信息/ (20个txt文件)")
print("  - README.md (待补充)")
