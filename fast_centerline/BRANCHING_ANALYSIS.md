# 中线分叉问题：四种解决方案详细分析报告

## 目录

1. [问题描述](#1-问题描述)
2. [当前代码的根本原因分析](#2-当前代码的根本原因分析)
3. [方案 A：Steiner 树](#3-方案-a-steiner-树)
4. [方案 B：先剪枝再取全骨架](#4-方案-b-先剪枝再取全骨架)
5. [方案 C：度数感知路径提取](#5-方案-c-度数感知路径提取)
6. [方案 D：自动阈值剪枝](#6-方案-d-自动阈值剪枝)
7. [对比总结表](#7-对比总结表)
8. [建议](#8-建议)

---

## 1. 问题描述

当输入 polygon 呈 Y 形、T 形或有多个分叉时，ArcGIS 内置工具会生成**带分叉的中线**（紫色线），
而我们的工具只生成**一条从端到端的最长路径**（红色线），忽略了所有分支。

```
ArcGIS 效果（紫色）：          我们的工具（红色）：

     ╱  ╲                          │
    ╱    ╲                         │
   ╱      ╲                        │
   ────────                    ────────
       │                           │
       │                           │
```

## 2. 当前代码的根本原因分析

### 2.1 执行流程

```
polygon_to_centerline_wkt(wkt, single_line=True)
  └─ _centerline_voronoi_fast(exterior, holes, ...)
       ├─ Stage 0: 边界加密 (_densify_fast)
       ├─ Stage 1: Voronoi 细分 (scipy.spatial.Voronoi)
       ├─ Stage 2: 过滤无效脊线 (距离过滤 + PIP + 交叉检测)
       ├─ Stage 3: 构建 NetworkX 图 (G)
       ├─ [可选] _prune_branches(G, threshold)
       └─ ★ _extract_longest_path(G)  ← 问题在此
```

### 2.2 `_extract_longest_path` 的工作方式

```python
def _extract_longest_path(G):
    # 1. 如果图不连通，仅取最大连通分量
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    # 2. 找所有叶节点（度数=1的节点）
    leaves = [n for n in G.nodes() if G.degree(n) == 1]

    # 3. 两次 Dijkstra：找图直径（两个最远叶节点之间的路径）
    dist1, _ = nx.single_source_dijkstra(G, leaves[0])
    u = max(leaves, key=lambda n: dist1.get(n, 0.0))
    dist2, path2 = nx.single_source_dijkstra(G, u)
    v = max(leaves, key=lambda n: dist2.get(n, 0.0))
    return path2[v]  # 返回一条路径
```

**关键问题**：这个算法永远只返回**一条路径**——即图中两个最远叶节点之间的最短路径。
对于 Y 形 polygon，图中有 3 个叶节点和 1 个分叉点（度数=3 的节点），
但算法只选择最远的两个叶节点，第三个分支被完全忽略。

### 2.3 `single_line=False` 的问题

将 `single_line` 设为 `False` 会返回全部骨架边，但这包含大量噪声末端短边
（Voronoi 图在 polygon 凸角/凹角处产生的毛刺），效果杂乱不实用。

### 2.4 实测数据

对一个 Y 形测试 polygon 的 Voronoi 骨架图分析：

| 指标 | 值 |
|------|-----|
| 图节点数 | 78 |
| 图边数 | 72 |
| 叶节点 (度=1) | 13 |
| 分叉点 (度≥3) | 1 |
| 连通分量数 | 6 |

大部分叶节点是 Voronoi 噪声产生的短末端，
真正有意义的分叉结构只有 1 个分叉点连接 3 个主要分支。

---

## 3. 方案 A：Steiner 树

### 3.1 工作原理

Steiner 树是图论中连接一组**指定终端节点**的**最小权重子树**。

```
输入：骨架图 G + 终端节点集合 T = {所有叶节点}
输出：G 的一棵子树，包含 T 中所有节点，且总边权最小

步骤：
1. 从骨架图 G 中提取所有叶节点（度=1）作为终端节点 T
2. 调用 nx.approximation.steiner_tree(G, T, weight='weight')
3. 返回 Steiner 树的所有边作为 MULTILINESTRING
```

### 3.2 算法详解

NetworkX 的 `steiner_tree` 使用的是 Kou-Markowsky-Berman 近似算法：

```
1. 计算所有终端节点对之间的最短路径距离
2. 以这些距离构建完全图 G'（节点=终端节点）
3. 在 G' 上求最小生成树 MST
4. 将 MST 中的边映射回原图 G 中的最短路径
5. 合并这些路径得到子图 H
6. 在 H 上再求一次最小生成树
7. 移除度=1 的非终端节点（修剪多余分支）
```

### 3.3 优点

- **效果最接近 ArcGIS**：自然地保留所有有意义的分支，连接所有末端
- **理论保证**：在近似比为 2 的范围内找到最优连接方案
- **无需用户调参**：不需要设置阈值参数
- **自动忽略噪声**：短小毛刺作为叶节点不会影响树结构

### 3.4 缺点

- **精确 Steiner 树是 NP-hard**：需要使用近似算法
- **复杂度较高**：O(|T|² × V × log V)，其中 T 是终端数量，V 是图节点数
- **噪声叶节点问题**：Voronoi 噪声产生的短末端叶节点也会被当作终端节点，
  需要先进行小分支剪枝才能得到干净结果
- **需要与剪枝配合**：如果不先清理短毛刺，Steiner 树可能包含不必要的分支
- **可能引入冗余路径**：极端情况下，近似算法可能产生非最优的连接方式

### 3.5 伪代码

```python
def _extract_steiner_tree(G, prune_first=True, min_branch_len=0):
    # 1. 可选：先剪除短毛刺
    if prune_first and min_branch_len > 0:
        G = _prune_branches(G, min_branch_len)

    # 2. 取最大连通分量
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    # 3. 收集终端节点
    terminals = [n for n in G.nodes() if G.degree(n) == 1]
    if len(terminals) < 2:
        return _extract_longest_path(G)  # fallback

    # 4. 计算 Steiner 树
    st = nx.approximation.steiner_tree(G, terminals, weight='weight')

    # 5. 返回所有边
    return [(u, v) for u, v in st.edges()]
```

### 3.6 实现复杂度

**中等**。核心逻辑约 20 行代码，但需要：
- 修改 `_centerline_voronoi_fast` 的返回逻辑
- 修改 `polygon_to_centerline_wkt` 的输出格式处理
- 为 Steiner 树设计合理的叶节点选取策略

---

## 4. 方案 B：先剪枝再取全骨架

### 4.1 工作原理

利用已有的 `_prune_branches` 函数清除短分支后，返回完整的骨架图（所有剩余边）。

```
步骤：
1. 构建 Voronoi 骨架图 G
2. 调用 _prune_branches(G, threshold) 移除短于阈值的末端分支
3. 返回 G 中所有剩余边作为 MULTILINESTRING
```

### 4.2 算法详解

`_prune_branches` 的工作流程：

```
重复以下步骤直到无变化：
  1. 找所有叶节点（度=1 的节点）
  2. 对每个叶节点：
     a. 从叶节点出发，沿唯一路径向内行走
     b. 累加边权（路径长度）
     c. 直到遇到分叉点（度≥3）或另一个叶节点时停止
     d. 如果累积长度 < threshold → 移除整段路径
     e. 如果累积长度 ≥ threshold → 保留

示例（threshold = 10）：
  移除前：  A──3──B──5──C──20──D──4──E
                                      │
                                      8
                                      │
                                      F
  A-B-C: 长度 = 3+5 = 8 < 10 → 移除 A,B
  E:     长度 = 4 < 10 → 移除 E
  移除后：  C──20──D
                   │
                   8
                   │
                   F
```

### 4.3 优点

- **实现最简单**：代码改动极小，仅需修改 `single_line=True` 的分支逻辑
- **现有代码复用**：`_prune_branches` 已经实现并测试过
- **用户可控**：通过 `prune_threshold` 参数精细控制保留哪些分支
- **执行速度最快**：剪枝操作本身很快，不增加算法复杂度

### 4.4 缺点

- **需要手动调参**：用户必须选择合适的 `prune_threshold` 值
  - 太小 → 毛刺太多（保留噪声短边）
  - 太大 → 有意义的短分支也被删除
- **不同 polygon 需要不同阈值**：一个固定阈值不能适应所有几何形状
- **无法区分"噪声毛刺"和"有意义短分支"**：纯粹基于长度判断
- **结果可能不连通**：过度剪枝可能导致图断裂

### 4.5 伪代码

```python
# 在 _centerline_voronoi_fast 中修改：
if single_line:
    if prune_threshold > 0:
        G = _prune_branches(G, prune_threshold)
    # 改为返回完整骨架而非最长路径
    return [(u, v) for u, v in G.edges()]
```

### 4.6 实现复杂度

**低**。仅需修改约 5 行代码。但实用性高度依赖用户选择合适的阈值。

---

## 5. 方案 C：度数感知路径提取

### 5.1 工作原理

分析骨架图的拓扑结构，识别出分叉点（度≥3 的节点），
然后将骨架分解为"分叉点到叶节点"和"分叉点到分叉点"的路径段，
再智能地组装这些路径段为一棵树。

```
步骤：
1. 构建 Voronoi 骨架图 G
2. 识别所有分叉点（度≥3 的节点）和叶节点（度=1 的节点）
3. 将图分解为路径段：
   - 叶节点 → 分叉点 的路径（末端分支）
   - 分叉点 → 分叉点 的路径（内部连接）
4. 根据长度过滤短路径段（噪声）
5. 组装剩余路径段为树形结构
6. 返回 MULTILINESTRING
```

### 5.2 算法详解

```
图分解过程：

原始骨架图：
  L1──s1──B1──s2──J1──s3──B2──s4──L2
                   │
                  s5
                   │
                  B3
                   │
                  s6
                   │
                  L3

  L = 叶节点 (degree 1)
  J = 分叉点 (degree ≥ 3)
  B = 过渡节点 (degree 2)
  s = 路径段

分解为路径段：
  段 1: L1 → J1  (经过 B1)      长度 = s1 + s2
  段 2: J1 → L2  (经过 B2)      长度 = s3 + s4
  段 3: J1 → L3  (经过 B3)      长度 = s5 + s6

过滤规则：
  - 保留所有长度 > min_branch_length 的段
  - min_branch_length 可以基于平均段长度自动计算

重组为树：
  将保留的路径段连接起来，形成分叉树结构
```

### 5.3 优点

- **精确的拓扑控制**：基于图结构而非纯长度判断
- **自然的分支识别**：分叉点是天然的分支连接处
- **保留骨架路径的连续性**：每段路径是从图中原始提取的，坐标精确
- **可以智能过滤**：基于段的长度/角度/宽度等多种标准

### 5.4 缺点

- **实现最复杂**：需要自己编写图分解和重组逻辑
- **边界情况多**：
  - 多个连通分量的处理
  - 纯环图（无叶节点无分叉点）的处理
  - 级联分叉点的处理（分叉点直接相邻）
  - 多分叉点之间的路径选择
- **过滤标准需要设计**：如何判断一个路径段是"有意义的分支"还是"噪声"
- **维护成本高**：逻辑较复杂，后续修改/调试困难

### 5.5 伪代码

```python
def _extract_branching_skeleton(G, min_branch_ratio=0.1):
    """提取分叉骨架。"""
    # 1. 识别关键节点
    leaves = [n for n in G.nodes() if G.degree(n) == 1]
    junctions = [n for n in G.nodes() if G.degree(n) >= 3]
    key_nodes = set(leaves + junctions)

    # 2. 分解为路径段
    segments = []
    visited_edges = set()
    for start in key_nodes:
        for neighbor in G.neighbors(start):
            edge = frozenset({start, neighbor})
            if edge in visited_edges:
                continue
            # 沿 degree-2 节点行走直到遇到另一个关键节点
            path = [start]
            current, prev = neighbor, start
            path_length = G[prev][current]['weight']
            while current not in key_nodes:
                visited_edges.add(frozenset({prev, current}))
                path.append(current)
                nxt = next(n for n in G.neighbors(current) if n != prev)
                path_length += G[current][nxt]['weight']
                prev = current
                current = nxt
            path.append(current)
            visited_edges.add(frozenset({prev, current}))
            segments.append({
                'path': path,
                'length': path_length,
                'start': start,
                'end': current,
            })

    # 3. 计算过滤阈值
    if segments:
        max_len = max(s['length'] for s in segments)
        min_length = max_len * min_branch_ratio

    # 4. 过滤短段（但保留连接分叉点的内部段）
    kept = []
    for seg in segments:
        is_internal = (seg['start'] in junctions and seg['end'] in junctions)
        if is_internal or seg['length'] >= min_length:
            kept.append(seg)

    # 5. 返回所有保留路径段的边
    edges = []
    for seg in kept:
        path = seg['path']
        for i in range(len(path) - 1):
            edges.append((path[i], path[i+1]))
    return edges
```

### 5.6 实现复杂度

**中高**。核心逻辑约 60-80 行代码，需要处理多种边界情况。

---

## 6. 方案 D：自动阈值剪枝

### 6.1 工作原理

自动根据 polygon 的几何特征（面积、周长、骨架长度等）计算合适的剪枝阈值，
然后用方案 B 的方式返回剪枝后的全骨架。用户无需手动调参。

```
步骤：
1. 构建 Voronoi 骨架图 G
2. 计算 polygon 的几何特征指标
3. 从指标推导合适的 prune_threshold
4. 调用 _prune_branches(G, auto_threshold)
5. 返回所有剩余边作为 MULTILINESTRING
```

### 6.2 算法详解

**自动阈值计算方式（多种策略）：**

#### 策略 1：基于 polygon 宽度

```
典型宽度 = 面积 / 骨架总长度
auto_threshold = 典型宽度 × 系数 (如 0.5 ~ 1.0)

原理：polygon 的"宽度"代表了有意义分支的最小尺度。
      短于宽度的分支通常是 Voronoi 噪声。
```

#### 策略 2：基于密度距离

```
auto_threshold = densify_distance × 系数 (如 3.0 ~ 5.0)

原理：densify_distance 定义了边界采样精度。
      短于几倍 densify_distance 的分支通常是采样噪声。
```

#### 策略 3：基于统计分布

```
计算所有末端分支的长度分布
auto_threshold = 中位数(branch_lengths) 或 均值 - 标准差

原理：基于实际分支长度的统计特征，自动找到"噪声"和"信号"的分界线。
```

#### 策略 4：基于面积比

```
polygon_area = polygon 面积
auto_threshold = sqrt(polygon_area) × 系数 (如 0.05 ~ 0.1)

原理：polygon 面积的平方根近似其"特征尺度"。
      短于此尺度的分支不太可能是有意义的结构。
```

### 6.3 优点

- **用户无需调参**：自动计算合适的阈值
- **自适应**：不同大小/形状的 polygon 会得到不同阈值
- **实现难度中等**：核心是选择合适的自动阈值公式
- **复用现有代码**：`_prune_branches` 已实现
- **可以提供 fallback**：如果自动阈值效果不好，用户仍可手动覆盖

### 6.4 缺点

- **阈值公式需要调优**：没有通用的完美公式，不同场景可能需要不同系数
- **可能过度/不足剪枝**：自动计算的阈值可能不适合极端形状
- **无法区分结构差异**：同一 polygon 内部可能有不同尺度的分支
  （例如一个大分叉 + 一个小分叉），统一阈值可能不合适
- **依赖几何指标的准确性**：面积/周长计算本身可能受坐标系影响

### 6.5 伪代码

```python
def _auto_prune_threshold(G, exterior, holes, densify_distance):
    """自动计算剪枝阈值。"""
    # 策略 1：基于 polygon 宽度
    area = _polygon_area(exterior, holes)
    total_skeleton_length = sum(
        d['weight'] for _, _, d in G.edges(data=True)
    )
    if total_skeleton_length > 0:
        typical_width = area / total_skeleton_length
        threshold = typical_width * 0.5
    else:
        threshold = densify_distance * 3.0

    return max(threshold, densify_distance * 3.0)  # 保证最小值

def _polygon_area(exterior, holes):
    """用 Shoelace 公式计算面积。"""
    def ring_area(ring):
        x, y = ring[:, 0], ring[:, 1]
        return 0.5 * abs(np.sum(x[:-1]*y[1:] - x[1:]*y[:-1]))
    area = ring_area(exterior)
    for hole in holes:
        area -= ring_area(hole)
    return area
```

### 6.6 实现复杂度

**中等**。核心逻辑约 30 行代码，但需要测试和调优系数。

---

## 7. 对比总结表

| 维度 | 方案 A: Steiner 树 | 方案 B: 剪枝+全骨架 | 方案 C: 度数感知 | 方案 D: 自动阈值 |
|------|:-:|:-:|:-:|:-:|
| **效果接近 ArcGIS** | ★★★★★ | ★★★☆☆ | ★★★★☆ | ★★★★☆ |
| **实现难度** | 中等 (~20行) | 低 (~5行) | 高 (~80行) | 中等 (~30行) |
| **是否需要用户调参** | 否（但建议配合剪枝） | **是**（必须设阈值） | 否（内置比例过滤） | 否（自动计算） |
| **运行速度** | 较慢（近似 Steiner） | 最快 | 快 | 快 |
| **鲁棒性** | 高 | 中（依赖阈值） | 中（边界情况多） | 中（公式依赖） |
| **代码维护性** | 高（依赖 networkx） | 最高 | 低（逻辑复杂） | 高 |
| **噪声处理能力** | 需要配合预剪枝 | 取决于阈值选择 | 内置比例过滤 | 自动 |
| **适应不同形状** | 好 | 差（固定阈值） | 好 | 中等 |

### 各方案输出对比（Y 形 polygon）

```
方案 A (Steiner 树):        方案 B (剪枝+全骨架):
     ╱  ╲                       ╱  ╲
    ╱    ╲                     ╱    ╲
   ╱      ╲                  ╱      ╲
   ────┬───                  ────┬───── + 可能的毛刺
       │                        │
       │                        │

方案 C (度数感知):          方案 D (自动阈值):
     ╱  ╲                       ╱  ╲
    ╱    ╲                     ╱    ╲
   ╱      ╲                  ╱      ╲
   ────┬───                  ────┬───
       │                        │
       │                        │
```

---

## 8. 建议

### 推荐实施顺序

1. **首选方案 A (Steiner 树)**：效果最接近 ArcGIS，无需用户调参。
   配合先执行一轮基于 `densify_distance` 的小分支预剪枝（移除短于 `3 × densify_distance` 的叶节点分支），
   即可获得干净的分叉中线。NetworkX 已内置 `steiner_tree` 近似算法，
   实现代价可控。

2. **备选方案 D (自动阈值)**：如果 Steiner 树性能不满足需求，
   方案 D 是很好的退路——复用现有 `_prune_branches`，
   只需添加自动阈值计算逻辑。

3. **方案 B** 可作为最快速的临时方案（用户自己设 `prune_threshold`），
   但长期不推荐作为默认行为。

4. **方案 C** 虽然理论上最精确，但实现和维护成本最高，不建议优先选择。

### 兼容性建议

无论选择哪个方案，都应保留现有的 `single_line` 参数：
- `single_line=True` → 新的分叉中线提取（MULTILINESTRING）
- `single_line=False` → 完整骨架（保持现有行为）
- 可考虑新增 `single_line="longest"` 选项保留原始最长路径行为
