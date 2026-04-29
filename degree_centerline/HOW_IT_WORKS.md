# Degree-Aware Branching Centerline 工具运行原理详解

> 本文档面向希望完全理解本工具**每一步工作原理**和**代码结构**的读者。
> 所有环节均采用"**先用类比通俗解释，再用精确技术语言补充**"的双层结构。

---

## 目录

1. [总览：这个工具在做什么？](#1-总览这个工具在做什么)
2. [整体流水线：从输入到输出](#2-整体流水线从输入到输出)
3. [第一步：读懂地图——WKT 解析](#3-第一步读懂地图wkt-解析)
4. [第二步：给边界"补牙"——向量化加密](#4-第二步给边界补牙向量化加密)
5. [第三步：画"势力范围"——Voronoi 图构建](#5-第三步画势力范围voronoi-图构建)
6. [第四步：去伪存真——脊线过滤](#6-第四步去伪存真脊线过滤)
7. [第五步：搭建骨架——构建 NetworkX 图](#7-第五步搭建骨架构建-networkx-图)
8. [第六步：智能修剪——度数感知分支提取（核心算法）](#8-第六步智能修剪度数感知分支提取核心算法)
9. [第七步：交付成果——输出 WKT](#9-第七步交付成果输出-wkt)
10. [代码结构总览表](#10-代码结构总览表)
11. [四大加速技术详解](#11-四大加速技术详解)
12. [与 Steiner 方法对比](#12-与-steiner-方法对比)
13. [参数说明与调优指南](#13-参数说明与调优指南)

---

## 1. 总览：这个工具在做什么？

### 🎯 通俗解释

想象你手里有一张**不规则形状的纸条**（比如一条弯弯曲曲的河流），
你想找到它的"**中线**"——也就是沿着纸条中央画的那条线。

如果纸条像字母 **Y** 一样分叉了呢？那中线也应该分叉，形成一棵"**中线树**"。

这个工具做的就是：

```
输入：一个 polygon（多边形）的形状描述
       ↓
处理：自动找到中线，包括所有分叉
       ↓
输出：中线的形状描述
```

### 🔬 精确描述

本工具接收 **WKT (Well-Known Text)** 格式的 `POLYGON` 或 `MULTIPOLYGON` 字符串，
通过 **Voronoi 骨架提取** 算法计算其**中轴线 (medial axis)**，
然后使用**度数感知拓扑分解 (degree-aware topological decomposition)** 算法
保留所有有意义的分支，过滤 Voronoi 噪声毛刺，
最终输出 `MULTILINESTRING` WKT 字符串。

核心算法复杂度为 **O(V + E)**（线性时间），其中 V 是图节点数，E 是图边数。

---

## 2. 整体流水线：从输入到输出

### 🎯 通俗解释

想象一条**流水线工厂**，你的 polygon 就是原材料，经过 7 道工序后变成精美的中线产品：

```
        ┌─────────────────────────────────────────────────────────────┐
        │                    完 整 流 水 线                            │
        │                                                             │
        │  📄 WKT文本  ──→  解析  ──→  加密  ──→  Voronoi  ──→      │
        │                                                             │
        │  ──→  过滤  ──→  建图  ──→  🌳 度数感知提取  ──→  📄 WKT   │
        └─────────────────────────────────────────────────────────────┘
```

每道工序的目的：

| 工序 | 类比 | 做什么 |
|------|------|--------|
| ① WKT 解析 | 读懂设计图纸 | 把文本变成坐标数组 |
| ② 边界加密 | 给粗糙的轮廓补上更多点 | 确保边界足够细腻 |
| ③ Voronoi 构建 | 画"势力范围"分界线 | 生成初始骨架候选线 |
| ④ 脊线过滤 | 只保留在形状内部的线 | 去除超出 polygon 的无效线段 |
| ⑤ 构建图 | 把线段连成网络 | 用图数据结构表示骨架 |
| ⑥ 度数感知提取 | 🌟 智能剪枝，保留分叉 | 去除噪声，保留有意义的分支 |
| ⑦ 输出 WKT | 把结果写成标准格式 | 生成最终的中线文本 |

### 🔬 精确描述

```
polygon_to_centerline_wkt(wkt)                    # 公开 API 入口
  │
  ├─ _parse_wkt_polygon(wkt)                       # ① 解析 WKT → [(exterior, holes)]
  │
  └─ _centerline_voronoi_fast(exterior, holes, ...)  # 主管道
       │
       ├─ _compute_perimeter() + 自适应调密度       # 计算周长，必要时自动放宽密度
       ├─ _densify_fast()                           # ② 向量化加密边界
       │
       ├─ Voronoi(pts)                              # ③ scipy Voronoi 图
       ├─ 无限脊线 + 生成距离过滤                    # ③ 初步清理
       │
       ├─ _segments_in_polygon_batch()              # ④ 批量 PIP + 交叉检测
       │     ├─ _pip_polygon_batch()                #    点在多边形内判断
       │     └─ _segments_cross_ring_batch()        #    线段与边界交叉检测
       │
       ├─ nx.Graph() + add_edges_from()             # ⑤ 构建 NetworkX 图
       ├─ _prune_branches()                         # ⑤ 可选：用户指定阈值剪枝
       │
       ├─ _extract_branching_skeleton()             # ⑥ 🌟 核心：度数感知提取
       │     ├─ 预剪枝短毛刺
       │     ├─ 识别关键节点（叶+分叉点）
       │     ├─ 分解为路径段
       │     ├─ 按比例过滤短末端段
       │     └─ 重建连通边集
       │
       └─ _edges_to_multilinestring_wkt()           # ⑦ 输出 WKT
```

**代码位置**：`centerline_degree.py`，公开 API 函数在第 **1205–1281** 行。

---

## 3. 第一步：读懂地图——WKT 解析

### 🎯 通俗解释

WKT 就像一种**坐标语言**，它用文本描述形状。比如一个矩形可以写成：

```
POLYGON ((0 0, 100 0, 100 20, 0 20, 0 0))
```

意思是：从原点 (0,0) 出发，走到 (100,0)，再到 (100,20)，再到 (0,20)，回到 (0,0)。

如果 polygon 有**洞**（比如甜甜圈形状），就多加一组坐标：

```
POLYGON ((0 0, 100 0, 100 100, 0 100, 0 0), (30 30, 70 30, 70 70, 30 70, 30 30))
         └──────── 外边界 ────────────────┘  └────────── 内部的洞 ──────────────┘
```

这一步的工作：把这些文字**拆成数字数组**，后面的步骤才能做数学运算。

### 🔬 精确描述

解析由 4 个函数完成，调用关系如下：

```
_parse_wkt_polygon(wkt)
  ├─ 检测 "MULTIPOLYGON" vs "POLYGON" 前缀
  ├─ _split_at_depth(body, 0)     # 按逗号在特定括号深度拆分
  └─ _parse_polygon_body(ps)       # 提取外环 + 内环列表
       └─ _parse_ring(ring_str)    # 每个环 → float64 ndarray (N, 2)
```

| 函数 | 代码行 | 输入 | 输出 |
|------|--------|------|------|
| `_split_at_depth()` | 192–211 | 字符串, 深度 | 按指定括号深度的逗号分割的字符串列表 |
| `_parse_ring()` | 214–227 | 坐标字符串 `"x1 y1, x2 y2, ..."` | `np.ndarray (N, 2)` float64 |
| `_parse_polygon_body()` | 230–237 | POLYGON 括号内容 | `(exterior_ndarray, [hole_ndarrays])` |
| `_parse_wkt_polygon()` | 240–271 | 完整 WKT 字符串 | `[(exterior, holes), ...]` 列表 |

**关键细节**：外环至少需要 4 个顶点（首尾重合的闭合环）。

---

## 4. 第二步：给边界"补牙"——向量化加密

### 🎯 通俗解释

想象你有一个粗略勾勒的 polygon，只有 4 个角点。
Voronoi 算法需要边界上有**足够多的点**才能生成精细的骨架——
就像一把好梳子需要很多梳齿才能梳理得均匀。

**加密 (Densification)** 就是在原有顶点之间**均匀插入新的点**：

```
   加密前（4个点）：              加密后（很多点）：

   ●─────────────────●           ●──●──●──●──●──●──●
   │                 │           ●                  ●
   │                 │           ●                  ●
   │                 │           ●                  ●
   ●─────────────────●           ●──●──●──●──●──●──●
```

**加速秘诀**：传统做法是一个一个点算（像用手一个个摆牙签），
本工具用 NumPy 的 `np.repeat` 一次性算出所有点（像用模具一次性冲压）。

### 🔬 精确描述

**函数**：`_densify_fast(exterior, holes, max_distance)` — 第 334–412 行

**算法**：

1. 对每条边计算长度：`lengths = np.hypot(diffs[:, 0], diffs[:, 1])`
2. 每条边需要插入的点数：`n_inserts = ceil(length / max_distance)`
3. 用 `np.repeat` + cumsum 技巧一次性生成所有插值点：

```python
# 关键代码（简化版）：

# 复制每条边的起点，每条边重复 n_inserts 次
starts_rep = np.repeat(ring[:-1], n_inserts, axis=0)  # (total_pts, 2)
diffs_rep  = np.repeat(diffs,     n_inserts, axis=0)  # (total_pts, 2)

# 用 cumsum 技巧计算每个点在其所属边上的位置参数 t ∈ [0, 1)
seg_start_in_flat = np.cumsum(n_inserts)   # 每条边在总数组中的起始位置
k = arange(total_pts) - seg_start[seg_idx] # 局部偏移量
t = k / n_inserts_per_point                # 参数 t ∈ [0, 1)

# 一行 numpy 完成所有点的插值
pts = starts_rep + t[:, None] * diffs_rep   # (total_pts, 2)
```

**输出**：
- `pts`：所有加密后的点 `(N, 2) float64`
- `ring_ids`：每个点属于哪个环（0=外环，≥1=第几个洞）

**自适应密度（Plan D）**：如果加密后的点数超过 `_MAX_DENSIFY_POINTS`（默认 10,000），
自动放大 `densify_distance`，避免内存爆炸。（代码第 887–907 行）

```
           ┌────────────────────────────────────────────────┐
           │  自适应密度调整示例                              │
           │                                                │
           │  周长 = 10,000     densify_distance = 1.0      │
           │  预估点数 = 10,000 / 1.0 = 10,000              │
           │  10,000 ≤ MAX(10,000) → 不调整 ✅               │
           │                                                │
           │  周长 = 50,000     densify_distance = 1.0      │
           │  预估点数 = 50,000 / 1.0 = 50,000              │
           │  50,000 > MAX(10,000) → 自动调整为 5.0 ⚠️       │
           └────────────────────────────────────────────────┘
```

---

## 5. 第三步：画"势力范围"——Voronoi 图构建

### 🎯 通俗解释

想象边界上的每个点都是一个**小城市**，每个城市都有自己的**势力范围**
（谁离这个城市最近就归它管）。

所有势力范围的**分界线**就叫 **Voronoi 图**。
这些分界线天然就在形状的"中间"——因为它们是两个边界点等距的地方！

```
    边界点 ● 的"势力范围"分界线就是中线！

    ●──●──●──●──●──●──●
    │  ╱  ╱  │  ╲  ╲  │      ← Voronoi 分界线（粗线 = 中线候选）
    │ ╱  ╱   │   ╲  ╲ │
    │╱  ╱    │    ╲  ╲│
    ●  ╱     │     ╲  ●
    │ ╱      │      ╲ │
    ●──●──●──●──●──●──●
```

但是！Voronoi 图会产生很多**伸到 polygon 外面**的无用线段，
需要后续步骤去掉。

### 🔬 精确描述

**代码位置**：`_centerline_voronoi_fast()` 第 915–944 行

**步骤**：

1. **构建 Voronoi 图**：`vor = Voronoi(pts)` — 调用 SciPy 的 Voronoi 算法
2. **提取脊线数据**：
   - `vor.ridge_vertices`：每条脊线的两个端点索引 `(R, 2)`
   - `vor.ridge_points`：每条脊线对应的两个生成点索引 `(R, 2)`
   - `vor.vertices`：所有 Voronoi 顶点的坐标 `(V, 2)`

3. **第一道过滤——去除无限脊线** (第 925–928 行)：
   ```python
   finite_mask = (rv[:, 0] >= 0) & (rv[:, 1] >= 0)
   ```
   顶点索引为 -1 的脊线延伸到无穷远，直接丢弃。

4. **第二道过滤——生成距离过滤** (第 933–941 行)：
   - 如果一条脊线的两个生成点在**同一个环**上且**距离太近**，
     说明这条脊线是边界凹凸产生的噪声，丢弃。
   - 阈值 = `3 × densify_distance`

```
   ┌──────────────────────────────────────────────┐
   │          Voronoi 过滤流程                      │
   │                                               │
   │  原始脊线数 R                                  │
   │       │                                       │
   │       ▼                                       │
   │  过滤1：去除无限脊线（端点索引=-1）             │
   │       │                                       │
   │       ▼                                       │
   │  过滤2：去除同环近距离脊线（噪声毛刺）          │
   │       │                                       │
   │       ▼                                       │
   │  剩余脊线 → 进入下一步 PIP 检测               │
   └──────────────────────────────────────────────┘
```

---

## 6. 第四步：去伪存真——脊线过滤

### 🎯 通俗解释

经过第三步，我们有了一堆 Voronoi 脊线。但很多脊线**跑到 polygon 外面去了**
（就像骨头长出了皮肤外面）。

这一步做两件事：
1. **点检测**：脊线的中点在 polygon 里面吗？（"你站在里面还是外面？"）
2. **线检测**：脊线是否穿过了 polygon 的边界？（"你有没有钻墙？"）

只有两个检测都通过的脊线才保留。

```
   ● = polygon 边界        ─── = Voronoi 脊线

   ●──────────────────●
   │  ──✓──  ──✓──   │    ← 在里面且不穿墙 → 保留 ✅
   │     ───✗────── ─│──  ← 穿过了边界 → 丢弃 ❌
   │  ──✓──          │
   ●──────────────────●
        ──✗──              ← 完全在外面 → 丢弃 ❌
```

### 🔬 精确描述

**函数**：`_segments_in_polygon_batch(v0s, v1s, exterior, holes)` — 第 571–609 行

**两阶段流水线**：

**Stage 1：批量点在多边形内 (PIP) 检测**

```python
midpoints = (v0s + v1s) * 0.5                         # 每段脊线的中点
pip_ok = _pip_polygon_batch(midpoints, exterior, holes) # 批量检测
```

> **PIP 算法**（射线投影法）：从测试点向右发射一条水平射线，
> 数它穿过 polygon 边界的次数——奇数次=在内部，偶数次=在外部。

`_pip_polygon_batch` (第 458–477 行) 使用 **matplotlib.path.Path** 的 C 扩展实现，
一次检测所有 M 个中点。若 matplotlib 不可用，退回到 `_pip_ring_batch` (第 420–455 行)，
用 NumPy 向量化的射线投影法——对每个环顶点做一次 NumPy 操作（处理所有 M 个点）。

**Stage 2：批量边界交叉检测**

```python
ok_idx = np.where(pip_ok)[0]           # 只检测通过了 PIP 的脊线
for ring in [exterior] + holes:
    crosses = _segments_cross_ring_batch(v0_ok, v1_ok, ring)
    no_cross &= ~crosses               # 交叉 → 排除
```

`_segments_cross_ring_batch` (第 485–563 行) 使用**四次叉积法**判断两条线段是否相交：

```
   线段 AB 与 线段 CD 相交的条件：
   
   d1 = cross(CD, CA)    ←  A 在 CD 的哪一侧？
   d2 = cross(CD, CB)    ←  B 在 CD 的哪一侧？
   d3 = cross(AB, AC)    ←  C 在 AB 的哪一侧？
   d4 = cross(AB, AD)    ←  D 在 AB 的哪一侧？
   
   相交条件：d1 和 d2 异号 AND d3 和 d4 异号
```

**双维度分块**（代码第 519–559 行）：
- **M 维度**（脊线）：每 `_CHUNK_SIZE = 2048` 条一批
- **K 维度**（环边）：每 `_RING_CHUNK_SIZE = 4096` 条一批
- 峰值内存约 2048 × 4096 × 48 字节 ≈ **384 MB**

---

## 7. 第五步：搭建骨架——构建 NetworkX 图

### 🎯 通俗解释

经过过滤，我们有了一堆"好的"线段。现在要把它们**连成一个网络**，
就像用积木搭一个骨架模型——每个线段是一根积木棒，交叉点是连接头。

同时，我们给每根棒标上**长度**，这样后面才知道哪些是"长骨头"（有意义的分支），
哪些是"短刺"（噪声）。

```
   过滤后的线段集合              →         NetworkX 图
   
   ─── ───                                ●───●
    ╲ ╱                                    │ ╲ │
     ●                         →           ●  ●
    ╱ ╲                                    │ ╱ │
   ─── ───                                ●───●
                                           
   (松散线段)                    (连通的带权图)
```

### 🔬 精确描述

**代码位置**：`_centerline_voronoi_fast()` 第 960–988 行

```python
G = nx.Graph()

# 批量计算所有边的长度
lengths = np.hypot(v1s[:, 0] - v0s[:, 0], v1s[:, 1] - v0s[:, 1])

# 构建边列表——每条边附带 weight 属性（欧氏距离）
edge_list = [
    (tuple(p), tuple(q), {"weight": float(w)})
    for p, q, w in zip(v0s, v1s, lengths)
]
G.add_edges_from(edge_list)
```

**图的特征**：
- **节点**：Voronoi 顶点坐标 `(x, y)` 元组
- **边**：相邻 Voronoi 顶点之间的连线，权重 = 欧氏距离
- **节点度数**：
  - **度数 1** = 叶节点（末端）
  - **度数 2** = 链条节点（中间过渡）
  - **度数 ≥ 3** = 分叉点（路口）

可选的用户级剪枝：
```python
if prune_threshold > 0:
    G = _prune_branches(G, prune_threshold)  # 移除短于阈值的末端分支
```

`_prune_branches` (第 617–647 行) 迭代地找到所有叶节点，沿 degree-2 链走到分叉点，
如果总长度 < 阈值就删除整段。重复直到无变化。

---

## 8. 第六步：智能修剪——度数感知分支提取（核心算法）

### 🎯 通俗解释

这是本工具**最核心、最与众不同**的一步！

想象骨架图像一棵**杂草丛生的树**——主干和大分枝是"有意义的中线"，
但根部和末端长满了短小的"毛刺"（Voronoi 噪声）。

普通修剪（Steiner 树方法）的做法是：
先算出哪些叶子"重要"，再用复杂算法连接它们——就像请一个测量师量遍整棵树再决定剪哪些枝。

**度数感知方法**的做法更聪明：
直接看每个节点有几条路——像一个**交警在路口数车道**：

```
   路口类型            对应图概念           处理方式
   
   死胡同  ●──        度数 = 1（叶节点）   可能是噪声末端
   直路    ──●──      度数 = 2（链条节点）   中间过渡，不用管
   三岔路  ──●──      度数 = 3（分叉点）    重要！是真正的分叉
           │
   十字路  ──●──      度数 = 4              非常重要的交叉口
           │
```

算法把**叶节点**和**分叉点**当作"关键路口"，
在它们之间沿"直路"走过去，测量每段路的长度，
然后：
- ✅ **长路**（占最长段的 10% 以上）→ 保留
- ✅ **内部路**（分叉点到分叉点）→ **永远保留**
- ❌ **短路**（短小毛刺）→ 删除

```
   修剪前：                              修剪后：
   
       ╱short╲                              ╱     ╲
      ●       ●                            ●       ●
      │       │                            │       │
   ●──●───────●──●                      ●──●───────●──●
   │  │  long │  │ short                   │  long │
   ●  ●───────●  ●                         ●───────●
               │
               ●──● short
               ↓
             被删除
```

### 🔬 精确描述

**函数**：`_extract_branching_skeleton(G, min_branch_ratio=0.1, densify_distance=1.0)` — 第 702–845 行

#### Step 0：预剪枝噪声毛刺（第 736–741 行）

```python
pre_prune_threshold = 3.0 * densify_distance
G = _prune_branches(G.copy(), pre_prune_threshold)
```

移除长度 < 3 倍加密距离的末端分支——这些几乎都是 Voronoi 在 polygon 拐角处产生的伪影。

#### Step 1：取最大连通分量（第 743–749 行）

```python
if not nx.is_connected(G):
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
```

预剪枝可能使图断裂，只保留最大的连通部分。

#### Step 2：识别关键节点（第 751–763 行）

```python
leaves    = {n for n in G.nodes() if G.degree(n) == 1}   # 叶节点
junctions = {n for n in G.nodes() if G.degree(n) >= 3}   # 分叉点
key_nodes = leaves | junctions                            # 关键节点集合
```

**特殊情况**：
- 无分叉点 → 纯链条/纯路径 → 直接返回全部边
- 无关键节点 → 纯环 → 直接返回全部边

#### Step 3：分解为路径段（第 765–805 行）

这是核心步骤——沿 degree-2 链条"行走"，在关键节点之间划分路径段：

```
   图结构：
   
   L1 ── B1 ── B2 ── J1 ── B3 ── L2
                       │
                      B4
                       │
                      L3
   
   L = 叶节点(degree 1)   J = 分叉点(degree ≥ 3)   B = 链条节点(degree 2)
   
   分解结果：
   
   段 1: L1 → B1 → B2 → J1    长度 = w(L1,B1) + w(B1,B2) + w(B2,J1)
   段 2: J1 → B3 → L2          长度 = w(J1,B3) + w(B3,L2)
   段 3: J1 → B4 → L3          长度 = w(J1,B4) + w(B4,L3)
```

算法伪代码：
```python
for start in key_nodes:                     # 从每个关键节点出发
    for neighbor in G.neighbors(start):     # 对每条出边
        if edge already visited: skip
        
        path = [start]
        current = neighbor
        while current not in key_nodes:     # 沿 degree-2 链走
            path.append(current)
            next = G 中 current 的另一个邻居
            current = next
        path.append(current)                # 到达另一个关键节点
        
        segments.append({
            "path": path,
            "length": 累积边权,
            "start": path 的起点（关键节点）,
            "end": path 的终点（关键节点）
        })
```

使用 `visited_edges` 集合（frozenset）防止同一段路径被正反方向走两遍。

#### Step 4：比例过滤短末端段（第 810–823 行）

```python
max_len = max(seg["length"] for seg in segments)
min_length = max_len * min_branch_ratio    # 默认 10% × 最长段

kept = []
for seg in segments:
    is_internal = (seg["start"] in junctions and seg["end"] in junctions)
    if is_internal or seg["length"] >= min_length:
        kept.append(seg)
```

**核心规则**：

| 路径段类型 | 起点 → 终点 | 过滤规则 |
|-----------|------------|---------|
| 末端段 (terminal) | 叶节点 → 分叉点 | 长度 ≥ 10% × 最长段 才保留 |
| 内部段 (internal) | 分叉点 → 分叉点 | **永远保留**（不管多短） |
| 退化保护 | — | 如果全部被过滤，保留最长的那一段 |

**为什么"内部段永远保留"？** 因为两个分叉点之间的连接段是骨架结构的"脊柱"，
删除它会导致整个结构断裂。

```
   过滤示例（min_branch_ratio = 0.1，最长段 = 100）：
   
   段 A: L1 → J1, 长度 = 80  (80 ≥ 10 ✅ 保留)
   段 B: J1 → J2, 长度 = 5   (内部段 ✅ 永远保留)
   段 C: J2 → L2, 长度 = 60  (60 ≥ 10 ✅ 保留)
   段 D: J1 → L3, 长度 = 3   (3 < 10 ❌ 删除——噪声毛刺)
   段 E: J2 → L4, 长度 = 2   (2 < 10 ❌ 删除——噪声毛刺)
```

#### Step 5：重建边集并确保连通（第 825–845 行）

```python
# 用 frozenset 去重——(u,v) 和 (v,u) 视为同一条边
edge_set = set()
for seg in kept:
    for i in range(len(seg["path"]) - 1):
        edge_set.add(frozenset({path[i], path[i + 1]}))

# 从原图中提取子图
G_kept = G.edge_subgraph([...]).copy()

# 安全检查：如果过滤导致断裂，取最大连通分量
if not nx.is_connected(G_kept):
    G_kept = G_kept.subgraph(max(...)).copy()

return [(u, v) for u, v in G_kept.edges()]
```

---

## 9. 第七步：交付成果——输出 WKT

### 🎯 通俗解释

最后一步很简单：把保留的边（线段）按标准格式写成文本。

如果有多条线段（分叉中线），输出 `MULTILINESTRING`：

```
MULTILINESTRING ((0 0, 50 10), (50 10, 100 0), (50 10, 50 50))
                  └─ 段1 ──┘    └── 段2 ──┘    └── 段3 ──┘
```

### 🔬 精确描述

**函数**：`_edges_to_multilinestring_wkt(edge_pairs)` — 第 284–290 行

```python
def _edges_to_multilinestring_wkt(edge_pairs):
    parts = [
        "({} {}, {} {})".format(s[0], s[1], e[0], e[1])
        for s, e in edge_pairs
    ]
    return "MULTILINESTRING ({})".format(", ".join(parts))
```

每条边 `((x1, y1), (x2, y2))` 被格式化为 `(x1 y1, x2 y2)`，所有边用逗号连接后包裹在 `MULTILINESTRING (...)` 中。

---

## 10. 代码结构总览表

```
centerline_degree.py（1,282 行）
│
├── 模块文档字符串 + 四大加速技术说明 .............. 第 1–145 行
├── 导入 + 常量 ................................... 第 147–184 行
│     _CHUNK_SIZE = 2048
│     _MAX_DENSIFY_POINTS = 10,000
│     _RING_CHUNK_SIZE = 4096
│
├── WKT 解析 ...................................... 第 192–271 行
│     _split_at_depth()         # 按括号深度拆分
│     _parse_ring()             # 坐标串 → ndarray
│     _parse_polygon_body()     # 提取外环+内环
│     _parse_wkt_polygon()      # 解析完整 WKT
│
├── WKT 输出 ...................................... 第 279–302 行
│     _path_to_linestring_wkt()
│     _edges_to_multilinestring_wkt()
│     _paths_to_wkt()
│
├── 周长计算 ...................................... 第 310–326 行
│     _compute_perimeter()      # 用于自适应密度
│
├── 技术 1：向量化加密 ............................ 第 334–412 行
│     _densify_fast()           # np.repeat + cumsum
│
├── 技术 2a：批量点在多边形内判断 ................. 第 420–477 行
│     _pip_ring_batch()         # numpy 射线投影
│     _pip_polygon_batch()      # 外环-内洞 + matplotlib 加速
│
├── 技术 2b：批量线段交叉检测 ..................... 第 485–563 行
│     _segments_cross_ring_batch()  # 双维度分块 + 四叉积
│
├── 批量线段在多边形内判断 ........................ 第 571–609 行
│     _segments_in_polygon_batch()  # PIP + 交叉 两阶段管道
│
├── 图工具 ........................................ 第 617–694 行
│     _prune_branches()         # 迭代叶节点剪枝
│     _traverse_cycle()         # 纯环遍历
│     _extract_longest_path()   # 两次 Dijkstra 求直径
│
├── 🌟 核心：度数感知分支提取 ..................... 第 702–845 行
│     _extract_branching_skeleton()
│       Step 0: 预剪枝
│       Step 1: 最大连通分量
│       Step 2: 识别关键节点
│       Step 3: 段分解
│       Step 4: 比例过滤
│       Step 5: 重建边集
│
├── 方法 A：向量化 Voronoi 骨架管道 ............... 第 853–988 行
│     _centerline_voronoi_fast()
│       Stage 1/4: 加密边界
│       Stage 2/4: Voronoi + 初步过滤
│       Stage 3/4: 批量 PIP + 交叉
│       Stage 4/4: 建图 + 提取分支
│
├── 技术 3：向量化栅格化 .......................... 第 996–1026 行
│     _rasterize_polygon_fast()
│
├── 技术 4：向量化骨架图构建 ...................... 第 1034–1113 行
│     _build_skeleton_graph_fast()
│
├── 方法 B：栅格骨架管道 .......................... 第 1121–1197 行
│     _centerline_skeleton_fast()
│
└── 📢 公开 API ................................... 第 1205–1281 行
      polygon_to_centerline_wkt()
```

---

## 11. 四大加速技术详解

### 🎯 通俗解释

本工具在前辈（`pure_centerline`）的基础上做了**四处大提速**：

| # | 传统做法（像手工） | 本工具做法（像机器） | 提速效果 |
|---|------------------|-------------------|---------|
| 1 | 一个一个点插值 | `np.repeat` 一次性插值所有点 | **3–10×** |
| 2 | 一条一条脊线检查是否在内部 | NumPy 批量检查所有脊线 | **20–200×** |
| 3 | 一行一行地光栅化 | matplotlib C 扩展整体光栅化 | **10–100×** |
| 4 | 一个一个像素检查邻居 | NumPy 数组偏移批量找邻居 | **4–8×** |

### 🔬 精确描述

#### 技术 1：向量化加密 `_densify_fast()` (第 334–412 行)

传统方法的三重 Python 循环：
```python
for ring in rings:              # O(1–2)
    for segment in ring:        # O(S) 段数
        for k in range(n):      # O(n) 插入点数
            points.append(...)  # Python list.append
```
总计 O(N) 次 Python `append` 调用。

向量化方法：
```python
starts_rep = np.repeat(ring[:-1], n_inserts, axis=0)  # 一次分配所有内存
pts = starts_rep + t[:, None] * diffs_rep              # 一次 numpy 运算
```
O(1) 次 numpy 调用，内部走 C/Fortran 循环。

#### 技术 2：批量脊线过滤 (第 420–609 行)

传统方法：
```python
for ridge in all_ridges:                       # O(M) 脊线
    for vertex in ring:                        # O(V) 环顶点
        ... 判断 ...                            # Python 标量运算
```
总计 O(M × V) 次 Python 迭代。

向量化方法：
```python
pip_ok = _pip_polygon_batch(midpoints, exterior, holes)  # O(V) numpy ops, 每次处理 M 个
crosses = _segments_cross_ring_batch(v0s, v1s, ring)     # O(1) numpy broadcast per chunk
```
O(V) 次 numpy 调用，每次处理所有 M 个脊线。

#### 技术 3：向量化栅格化 `_rasterize_polygon_fast()` (第 996–1026 行)

传统方法：逐行 Python 循环 O(rows × V)

向量化方法：
```python
all_points = np.column_stack([xx.ravel(), yy.ravel()])  # rows×cols 个像素中心
inside = _MplPath(exterior).contains_points(all_points) # 一次 C 调用
```

#### 技术 4：向量化骨架图构建 `_build_skeleton_graph_fast()` (第 1034–1113 行)

传统方法：逐像素检查 8 邻居 → O(N × 8) Python 调用

向量化方法：
```python
for dr, dc in 8_directions:
    shifted = yx + [dr, dc]                              # 所有像素同时偏移
    nb_in_skel = skeleton[shifted[:, 0], shifted[:, 1]]  # 批量索引查找
    G.add_edges_from([...])                              # 批量添加边
```

---

## 12. 与 Steiner 方法对比

### 🎯 通俗解释

| 方面 | Steiner 方法 | 度数感知方法 |
|------|-------------|-------------|
| 比喻 | 请一个测量师量遍整棵树，计算"最优连接方案" | 交警在路口数车道，直接按长度决定保留哪些路 |
| 速度 | 较慢——需要算"所有叶子之间最短路" | ⚡ 快——只走一遍图 |
| 结果 | 理论上"最优"连接 | 实际效果相当，自动适应噪声 |

### 🔬 精确描述

| 维度 | Steiner 树 (`steiner_centerline`) | 度数感知 (`degree_centerline`) |
|------|----------------------------------|-------------------------------|
| 核心算法 | `nx.approximation.steiner_tree()` | `_extract_branching_skeleton()` |
| 瓶颈调用 | `centerline_steiner.py:747` | 无瓶颈——纯线性遍历 |
| 时间复杂度 | O(T² · V · log V)，T=终端数 | **O(V + E)** 线性 |
| 空间复杂度 | O(T × V) 路径矩阵 | O(V + E) 仅图本身 |
| 噪声处理 | 预剪枝 + Steiner 自动选择 | 预剪枝 + 比例阈值过滤 |
| 需要用户调参 | 否 | 否（默认 10% 比例阈值） |
| 分支质量 | 理论最优近似 | 实测等价——分支方向和数量一致 |
| 依赖 | networkx 的 Steiner 近似算法 | 仅 networkx 基础图操作 |

**复杂度分解**：

```
   Steiner 方法内部发生了什么：
   
   对每对终端节点 (T₁, T₂) 运行 Dijkstra  → O(T² × (V log V + E))
   构建完全图                              → O(T²)
   计算 MST                                → O(T² log T)
   映射回原图                              → O(T × V)
   ────────────────────────────────────────
   总计：O(T² · V · log V)

   度数感知方法内部发生了什么：
   
   预剪枝                   → O(V + E) 每轮一遍
   识别关键节点              → O(V)
   段分解（每条边走一次）    → O(V + E)
   比例过滤                  → O(段数) ≤ O(V)
   重建                      → O(V + E)
   ────────────────────────────────────────
   总计：O(V + E)
```

**瓶颈分析文档**：详见 `steiner_centerline/STEINER_BOTTLENECK_ANALYSIS.md`。

---

## 13. 参数说明与调优指南

### 公开 API 参数

```python
polygon_to_centerline_wkt(
    wkt,                          # 输入 WKT 字符串（POLYGON 或 MULTIPOLYGON）
    method="voronoi",             # "voronoi"（推荐）或 "skeleton"（需要 scikit-image）
    densify_distance=1.0,         # 加密间距（单位与坐标系一致）
    prune_threshold=0.0,          # 用户级剪枝阈值（0=不剪枝）
    smooth_sigma=0.0,             # 高斯平滑参数（仅 skeleton 方法）
    raster_resolution=None,       # 栅格分辨率（仅 skeleton 方法）
    single_line=True,             # True=度数感知提取，False=返回全部骨架
    progress_callback=None,       # 进度回调函数
    max_densify_points=10_000,    # 自适应密度上限
)
```

### 常见调优场景

| 场景 | 调什么 | 怎么调 |
|------|--------|--------|
| 结果太粗糙 | `densify_distance` | 调小（如 0.5），生成更多边界点 → 更精细的骨架 |
| 有短毛刺残留 | `prune_threshold` | 调大（如 10.0），移除更多短分支 |
| 有意义的短分支被删了 | `min_branch_ratio`* | 在代码内调小（如 0.05），降低过滤比例 |
| 内存不够 / polygon 太大 | `max_densify_points` | 调小（如 5000），自动放宽加密间距 |
| 想看完整骨架（含噪声） | `single_line` | 设为 False，跳过度数感知过滤 |

*`min_branch_ratio` 目前不作为 API 参数暴露，需要在代码第 702 行修改默认值。

---

## 附录：文件清单

| 文件 | 用途 | 行数 |
|------|------|------|
| `centerline_degree.py` | 核心算法（本文档所述所有内容） | 1,282 |
| `Degree_Centerline.pyt` | ArcGIS Python Toolbox 界面封装 | 539 |
| `README.md` | 快速使用说明 | ~113 |
| `HOW_IT_WORKS.md` | 本文档——详细原理说明 | — |
| `install_dependencies.bat` | Windows 依赖安装脚本 | 123 |
| `requirements.txt` | Python 依赖列表 | 3 |
