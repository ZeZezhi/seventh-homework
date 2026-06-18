# 刘桢-25361124-人工智能编程课作业

**仓库地址**: https://github.com/ZeZezhi/seventh-homework

---

## 1. 任务分解与 AI 协作方式

本次作业共包含 6 项数据分析任务，涉及公交 IC 卡刷卡数据的预处理、统计分析与可视化。我在任务执行过程中扮演**主导者与决策者**的角色，AI（Claude Code）仅作为**代码生成与排错的辅助工具**。具体分工如下：

| 阶段 | 我的主导工作 | AI 的辅助工作 |
|------|-------------|--------------|
| **数据理解** | 阅读作业要求，理解 ICData 各字段含义，确定 ride_stops 计算方式和异常记录判定标准 | 按指令读取数据、打印字段信息 |
| **任务1 预处理** | 决定删除 ride_stops=0 的记录（40886条）、确认无缺失值后不执行删除操作 | 生成 pandas 读取、to_datetime 转换、abs() 计算等代码 |
| **任务2 时间分析** | 明确要求使用 `numpy.where` + `numpy.sum` 实现条件统计（不可用 pandas 替代），指定凌晨/深夜的颜色映射方案 | 生成 numpy 条件统计代码和 matplotlib 柱状图代码 |
| **任务3 线路分析** | 设计 `analyze_route_stops` 函数的参数接口、返回格式和 docstring；决定取 Top 15 并使用水平柱状图 | 生成函数框架、seaborn barplot 代码，我手动调整 errorbar 实现 |
| **任务4 PHF 计算** | 核心难点——设计"自动识别高峰小时 → 5分钟/15分钟窗口聚合 → PHF 计算"的完整算法流程；验证公式正确性 | 生成聚合和计算代码，我逐行审查并修正时间窗口的格式输出 |
| **任务5 驾驶员信息** | 确定筛选范围 1101~1120、定义输出格式（线路号首行 + 车辆-驾驶员对应表） | 生成批量文件写入代码 |
| **任务6 热力图** | 确定 4 个分析维度（驾驶员/线路/上车站点/车辆）和热力图配色方案（RdYlGn），分析并撰写 50 字观察结论 | 生成 Top-N 统计和 seaborn heatmap 代码 |

**总结**：所有分析方向、算法设计、参数选择均由我独立决策；AI 仅负责将我的指令转化为 Python 代码，且所有 AI 生成代码均经过我逐行审查和调优。

---

## 2. 关键 Prompt 调优记录

以下展示任务4（PHF 计算）中一次关键的 Prompt 调优过程：

### 初始 Prompt（我给 AI 的第一版指令）

> "计算 PHF5 和 PHF15，高峰小时就是刷卡量最大的小时。"

### AI 初始生成的代码（存在问题）

```python
# AI 初始版本：未处理分钟级窗口聚合，直接按小时统计
peak_hour = df['hour'].value_counts().idxmax()
peak_count = df['hour'].value_counts().max()
PHF5 = peak_count / (12 * peak_count)  # 公式理解错误
PHF15 = peak_count / (4 * peak_count)
```

**问题**：
1. PHF 的分母应为"高峰小时内最高 N 分钟窗口的刷卡量"，而非总刷卡量
2. 未对分钟级别的时间数据进行 5 分钟/15 分钟窗口聚合
3. 公式分母使用了 `peak_count` 而非实际窗口最大值

### 优化后的 Prompt

> "计算 PHF5 和 PHF15。首先统计全天每小时的刷卡量，自动找出刷卡量最大的小时作为高峰小时。然后在高峰小时内，按5分钟（分钟//5）和15分钟（分钟//15）进行窗口分组统计，分别找到各窗口内的最大刷卡量。PHF5 = 高峰小时总量 / (12 × 最高5分钟窗口量)，PHF15 = 高峰小时总量 / (4 × 最高15分钟窗口量)。输出格式参考：
> 高峰小时：08:00 ~ 09:00，刷卡量 XXXX 次
> 最高5分钟刷卡量：08:15~08:20，XX 次
> PHF5 = XXXX / (12 × XX) = X.XXXX
> 最高15分钟刷卡量：08:00~08:15，XX 次
> PHF15 = XXXX / (4 × XX) = X.XXXX"

### AI 最终生成（经我审核后）

```python
# 4.1 自动识别高峰小时
hourly_counts = df.groupby('hour').size()
peak_hour = hourly_counts.idxmax()
peak_count = hourly_counts.max()

# 4.2 提取高峰小时数据，获取分钟
peak_df = df[df['hour'] == peak_hour].copy()
peak_df['minute'] = peak_df[time_col].dt.minute

# 4.3 PHF5: 按5分钟窗口聚合
peak_df['min5_bin'] = peak_df['minute'] // 5     # 0-4→0, 5-9→1, ...
min5_counts = peak_df.groupby('min5_bin').size()
max_5min = min5_counts.max()
PHF5 = peak_count / (12 * max_5min)

# 4.4 PHF15: 按15分钟窗口聚合
peak_df['min15_bin'] = peak_df['minute'] // 15    # 0-14→0, 15-29→1, ...
min15_counts = peak_df.groupby('min15_bin').size()
max_15min = min15_counts.max()
PHF15 = peak_count / (4 * max_15min)
```

**调优效果**：准确实现了 PHF 计算逻辑，输出格式完全满足作业要求。

---

## 3. Debug 记录

### 问题描述
在任务5生成线路 txt 文件时，代码报错：
```
AttributeError: 'list' object has no attribute 'tolist'
```

### 发现过程
运行 `homework.py` 时，任务5在完成 20 个 txt 文件的写入后，执行 `print(f"线路列表: {unique_routes.tolist()}")` 时崩溃。

### 排查
检查代码发现，`unique_routes` 来自 `sorted(filtered_df[route_col].unique())`，而 `pandas.Series.unique()` 返回的是 **numpy array**，但经过 `sorted()` 内置函数处理后，返回值变成了 **Python 原生 list**。Python list 没有 `.tolist()` 方法，因此报错。

### 解决方案
将 `.tolist()` 调用移除，直接使用 `unique_routes`（它已经是 list）：
```python
# 修改前（报错）
print(f"   线路列表: {unique_routes.tolist()}")

# 修改后（正确）
print(f"   线路列表: {unique_routes}")
```

### 经验总结
在使用 pandas/numpy 数据结构的返回值时，应注意 `sorted()`、`list()` 等 Python 内置函数会改变返回类型，后续调用需适配新的数据类型。本次调试花费约 2 分钟定位到根因并修复。

---

## 4. 人工审核标注（代码注释）

以下为任务4 PHF 计算核心代码，由我逐行审核并添加注释（注释以 `# [人工审核]` 标注）：

```python
# ============================================================
# 任务4: 高峰小时系数 (PHF) 计算
# ============================================================

# [人工审核] 第一步：按小时聚合刷卡量，自动识别高峰小时
# groupby('hour').size() 对 0~23 每个小时计算刷卡记录数
hourly_counts = df.groupby('hour').size()
# [人工审核] idxmax() 返回刷卡量最大的小时（如 9 代表 09:00-09:59）
peak_hour = hourly_counts.idxmax()
peak_count = hourly_counts.max()   # 高峰小时总刷卡量
print(f"高峰小时为 {peak_hour:02d}:00 ~ {peak_hour:02d}:59，刷卡量 {peak_count} 次")

# [人工审核] 第二步：提取高峰小时内的子集，提取分钟字段
# 用 .copy() 避免 SettingWithCopyWarning
peak_df = df[df['hour'] == peak_hour].copy()
peak_df['minute'] = peak_df[time_col].dt.minute  # 交易时间的分钟部分 (0~59)

# ── PHF5: 以 5 分钟为窗口 ──
# [人工审核] 关键操作：整数除法将分钟映射到 5 分钟窗口
# 0~4 → bin 0, 5~9 → bin 1, ..., 55~59 → bin 11（共12个窗口）
peak_df['min5_bin'] = peak_df['minute'] // 5
# [人工审核] 按窗口分组，统计每个窗口的刷卡量
min5_counts = peak_df.groupby('min5_bin').size()
max_5min = min5_counts.max()        # 取 12 个窗口中刷卡量最高的值
max_5min_bin = min5_counts.idxmax() # 最高窗口的编号
# [人工审核] PHF5 = 高峰小时总量 / (12 × 最高5分钟窗口量)
# 分母中的 12 代表 1 小时 = 12 个 5 分钟窗口
PHF5 = peak_count / (12 * max_5min)

# ── PHF15: 以 15 分钟为窗口 ──
# [人工审核] 15 分钟窗口：0~14 → bin 0, 15~29 → bin 1,
# 30~44 → bin 2, 45~59 → bin 3（共4个窗口）
peak_df['min15_bin'] = peak_df['minute'] // 15
min15_counts = peak_df.groupby('min15_bin').size()
max_15min = min15_counts.max()
max_15min_bin = min15_counts.idxmax()
# [人工审核] PHF15 = 高峰小时总量 / (4 × 最高15分钟窗口量)
# 分母中的 4 代表 1 小时 = 4 个 15 分钟窗口
PHF15 = peak_count / (4 * max_15min)

# [人工审核] 格式化输出（时间窗口换算回分钟显示）
print(f"PHF5 = {peak_count} / (12 × {max_5min}) = {PHF5:.4f}")
print(f"PHF15 = {peak_count} / (4 × {max_15min}) = {PHF15:.4f}")
```

### 审核确认：
- ✅ 公式 `PHF = 高峰小时总量 / (窗口数 × 最高窗口刷卡量)` 符合交通工程学定义
- ✅ `minute // 5` 将 0~59 分钟正确映射到 12 个窗口
- ✅ `minute // 15` 将 0~59 分钟正确映射到 4 个窗口
- ✅ 使用 `.copy()` 避免链式赋值警告
- ✅ 输出保留 4 位小数，格式符合要求

---

## 附录：提交文件清单

| 文件 | 说明 |
|------|------|
| `ICData.csv` | 原始数据集（CSV 格式） |
| `homework.py` | 完整 Python 代码（可直接运行） |
| `hour_distribution.png` | 24小时刷卡分布柱状图（dpi=150） |
| `route_stops.png` | Top 15 线路平均站数水平柱状图（dpi=150） |
| `performance_heatmap.png` | 4 维度 × Top 10 服务效率热力图（dpi=150） |
| `公交驾驶员信息/` | 线路 1101~1120 的驾驶员-车辆对应表（20 个 txt） |
| `README.md` | 本文件——人机协同报告 |
