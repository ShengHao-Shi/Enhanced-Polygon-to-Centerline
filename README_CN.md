# 增强型多边形转中心线工具

> **🌐 语言 / Language**: [中文说明](#增强型多边形转中心线工具) · [English](README.md)

本工具集是一套 ArcGIS Python 工具箱，能够将细长多边形（河流、道路、走廊、水道）
转换为干净的、保留分叉结构的中心线折线。
核心算法采用**基于向量的 Voronoi 骨架**方法，仅需 ArcGIS Basic 许可，
无需 Spatial Analyst 扩展。

---

## 为什么不用 ArcGIS 原生方法？

ArcGIS 内置了两种近似提取中心线的方式：

| 原生方法 | 工作原理 | 局限性 |
|---|---|---|
| **Polygon to Centerline**（制图工具箱） | 内部使用 Thiessen 多边形，需要 **Advanced** 许可 | 仅限 Advanced 许可；无分支感知剪枝 |
| **最小代价路径（Least Cost Path）**（空间分析） | 将多边形栅格化，计算代价距离栅格，沿代价最小路径追踪线 | 栅格分辨率限制精度；只能追踪**单条路径**（无分叉）；需要 Spatial Analyst；结果高度依赖种子点位置 |

**栅格代价路径方法的核心缺陷（图解）：**

```
原始多边形边界（矢量）：           栅格化网格（单元大小 = d）：

  ╭──────────────────╮               ┌─┬─┬─┬─┬─┬─┬─┬─┐
  │                  │               │ │█│█│█│█│█│█│ │
  │                  │               ├─┼─┼─┼─┼─┼─┼─┼─┤
  │                  │               │ │█│█│█│█│█│█│ │
  ╰──────────────────╯               └─┴─┴─┴─┴─┴─┴─┴─┘

  • 形状精确                          • 边界呈锯齿状（误差 ≈ d）
  • 代价路径为阶梯线                   • 需要手动放置种子点
  • 分叉多边形 → 只能出一条路，         • 精度与单元大小成反比（越精细越慢）
    所有分支丢失
```

**本工具箱解决了上述所有问题**，完全在矢量域内通过 Voronoi 骨架算法工作：

| 特性 | 原生最小代价路径 | **本工具箱** |
|---|---|---|
| 处理矢量几何 | ✗（需栅格化） | ✅ |
| 所需许可 | Spatial Analyst | **无（Basic 即可）** |
| 处理分叉多边形 | ✗ 只有单路径 | ✅ 完整骨架 |
| 输出精度 | 受单元大小限制 | **亚顶点精度** |
| 大数据集速度 | 慢（栅格运算） | **O(V + E) 线性** |
| 需要种子点 | 是 | **否** |

---

## 仓库工具列表

| 工具 | 文件 | 用途 |
|---|---|---|
| **度数感知中心线** | `degree_centerline/Degree_Centerline.pyt` | 单次 Voronoi 骨架提取，含度数感知分支剪枝 |
| **分块中心线** | `degree_centerline/Split_and_Process.pyt` | 同一算法，但通过自适应四叉树分块处理大型/国家级数据集 |
| **比例缓冲区** | `proportional_buffer/arcpy_toolbox/Proportional_Buffer.pyt` | 沿已有中心线生成变宽度缓冲多边形（宽度 ∝ 局部河道宽度） |

---

## 快速下载

已打包为 zip 文件，可直接加载到 ArcGIS Pro：

| 下载 | 内容 |
|---|---|
| [**Degree_Centerline_Toolbox.zip**](downloads/Degree_Centerline_Toolbox.zip) | `Degree_Centerline.pyt` + `centerline_degree.py` + `Split_and_Process.pyt` + `split_and_process.py` + `install_dependencies.bat` |
| [**Proportional_Buffer_Toolbox.zip**](downloads/Proportional_Buffer_Toolbox.zip) | `Proportional_Buffer.pyt`（独立，无额外依赖） |

---

## 算法原理详解

### 第一步 · 边界加密（Densification）

多边形边界通常顶点过少，Voronoi 图会过于粗糙。
本步骤在相邻顶点之间均匀插入新点，间距为 `d`（**加密距离**）。

```
原始边界（4 个顶点）：                加密后（间距 = d）：

  A ──────────────── B                A ─·─·─·─·─·─·─·─ B
  │                  │                ·                  ·
  │                  │                ·                  ·
  D ──────────────── C                D ─·─·─·─·─·─·─·─ C

  4 点的 Voronoi → 粗糙骨架            多点的 Voronoi → 精细骨架，
  遗漏内部细节                          与形状高度吻合
```

> **实现**：使用 `numpy.repeat` + cumsum 全向量化，无逐点 Python 循环，
> 复杂度 O(N)。

---

### 第二步 · Voronoi 图构建（Voronoi Tessellation）

所有加密后的边界点输入 `scipy.spatial.Voronoi`。
每条 Voronoi **脊线**（相邻两个 Voronoi 单元之间的边）
都是候选中心线线段。

```
加密边界点（·）：                       Voronoi 图：

  · · · · · · · · ·                    · · · · · · · · ·
  ·               ·                    · ╲ │ │ │ │ │ ╱ ·
  ·               ·         →          ·  ╲│ │ │ │╱  ·
  ·               ·                    ·   ┼─┼─┼─┼   ·
  · · · · · · · · ·                    ·  ╱│ │ │ │╲  ·
                                        · ╱ │ │ │ │ ╲ ·
                                        · · · · · · · · ·

  每个边界点"拥有"一个单元格。
  单元格间的脊线到两侧边界点等距 → 自然描绘出中轴线。
```

---

### 第三步 · 内部脊线过滤（Interior Ridge Filtering）

并非所有 Voronoi 脊线都有意义，两个过滤器去除噪声：

1. **内部性检验**：脊线两端点必须都在多边形内部（批量射线投射法）。
2. **生成距离过滤器**：脊线的两个生成边界点距离极近（< 3×`d`）
   时，说明它们位于同侧，对应的脊线是凸角处的噪声毛刺，予以丢弃。

---

### 第四步 · 图结构构建（Graph Construction）

保留的脊线变为 `networkx.Graph` 中的边。
节点坐标为脊线端点，边权重为欧氏长度。

---

### 第五步 · 度数感知分支分解（Degree-Aware Branch Decomposition）

这是本工具与"最长路径"方法的核心区别所在。

**节点类型识别：**

```
  ● 度数 1 = 叶节点（分支末端）
  ◆ 度数 2 = 链节点（路过节点，无分叉）
  ★ 度数 ≥ 3 = 交叉节点（分叉点）
```

**路径段分解：**

```
  原始骨架图：

    L1 ──·──·──·── J1 ──·──·──·── L2
                    │
                   ·──·──·── L3

    分解为 3 个拓扑路径段：
      段 A：L1 → J1（长度 = 各边权重之和）
      段 B：J1 → L2
      段 C：J1 → L3
```

**噪声过滤规则：**

- 末端路径段（叶→交叉）长度 < `min_branch_ratio × 参考长度` → 作为噪声毛刺删除
- `参考长度 = max(最长路径段, 总骨架长度 × 0.5)`
  — 这个上下文感知的分母可防止在存在众多短分支时错误删除长分支
- 内部路径段（交叉→交叉）**始终保留**
- 朝向多边形钝角顶点（内角 > 150°）的末端分支额外删除，
  但若该分支长度 ≥ 总骨架长度的 20%，则作为结构性分支予以保留

```
  过滤后（min_branch_ratio = 0.10）：

    短噪声毛刺 L3 已删除（长度 < 参考长度的 10%）：

    L1 ─────────── J1 ─────────── L2
         （干净，无噪声毛刺）
```

**复杂度**：O(V + E)——关于骨架顶点数和边数线性，远快于 Steiner 树
或全对最短路径方法。

---

### 第六步 · 输出（Output）

保留的边组装为 `MULTILINESTRING` 几何并写入输出要素类，
同时完整保留输入多边形图层的所有属性字段。

---

## 安装说明

### 依赖包

| 包名 | 在 ArcGIS Pro 中的状态 | 备注 |
|---|---|---|
| `numpy` | 预装 | — |
| `scipy` | 通常预装 | — |
| `networkx` | **需要安装** | 一条命令（见下） |
| `matplotlib` | 通常预装 | 可选；加速栅格化 |

### 快速安装（推荐）

1. 打开 **ArcGIS Pro Python 命令提示符**。
2. `cd` 到 `degree_centerline/` 文件夹（或解压后的下载文件夹）。
3. 运行 `install_dependencies.bat`。

### 手动安装

```
conda create --name arcgispro-py3-degree --clone arcgispro-py3
activate arcgispro-py3-degree
conda install -c conda-forge -y networkx
```

然后在 ArcGIS Pro 中将 `arcgispro-py3-degree` 设置为活动环境
（**项目 → Python → Python 环境**），并重启 ArcGIS Pro。

> 比例缓冲区工具箱**无需额外依赖**——`numpy` 和 `scipy` 在默认
> ArcGIS Pro 环境中已预装。

---

## 使用指南

### 工具一 — 度数感知中心线（`Degree_Centerline.pyt`）

1. 解压 `Degree_Centerline_Toolbox.zip`，保持所有文件在**同一文件夹**内。
2. 在 ArcGIS Pro **目录**窗格 → 右键文件夹 → **添加工具箱** →
   选择 `Degree_Centerline.pyt`。
3. 运行 **Polygon to Centerline (Degree-Aware)**。

| 参数 | 说明 | 典型值 |
|---|---|---|
| 输入多边形要素 | 水道/道路/走廊多边形图层 | — |
| 输出中心线要素类 | 输出路径 | — |
| 加密距离 | 边界加密的顶点间距（坐标系单位） | `1.0` – `5.0` 米 |
| 最小分支比例 | 末端分支被剪枝的长度阈值（占参考长度的比例） | `0.05` – `0.15` |
| 剪枝阈值 | 噪声毛刺预剪枝的绝对长度阈值 | `0`（自动） |
| 返回原始完整骨架 | 勾选后跳过度数感知过滤 | 不勾选 |

**典型工作流程：**

```
输入多边形图层  →  度数感知中心线工具  →  中心线折线
```

---

### 工具二 — 大型数据集分块中心线（`Split_and_Process.pyt`）

对于国家级或超大型多边形数据集（每要素边界顶点 > 8 000 个），
单次处理算法可能不稳定。本工具通过自适应四叉树分块独立处理每个瓦片：

| 阶段 | 做什么 |
|---|---|
| **A** — 连通分量分割 | 每个不相连的多边形部分独立处理 |
| **B** — 四叉树细分 | 超过 `max_vertices` 的部分递归分割为四个象限瓦片 |
| **C** — 重叠缓冲提取 | 每个瓦片扩展 `buffer_factor × 加密距离`，防止瓦片缝处产生虚假分支 |

额外参数（除工具一参数外）：

| 参数 | 说明 | 典型值 |
|---|---|---|
| 每瓦片最大顶点数 | 阶段 B 分块阈值 | `8000` |
| 缓冲因子 | 重叠缓冲 = 因子 × 加密距离 | `5.0` |
| 最大四叉树深度 | 阶段 B 最大递归深度 | `5` |

---

### 工具三 — 比例缓冲区（`Proportional_Buffer.pyt`）

沿已有中心线生成变宽度缓冲多边形，
缓冲宽度 = `buffer_ratio × 2 × 局部半宽`，
局部半宽通过 KDTree 距离查询（O(N log M)）在每个采样点实时测量。

```
典型两步工作流程：

  第一步：运行度数感知中心线工具
             输入：水道多边形
             输出：中心线折线

  第二步：运行比例缓冲区工具
             输入：原始多边形 + 第一步输出的中心线
             输出：变宽度缓冲多边形
                   （例如：航行水道、洪泛区、河岸缓冲带）
```

| 参数 | 说明 | 默认值 |
|---|---|---|
| 输入多边形要素 | 源多边形图层 | — |
| 输入中心线要素 | 中心线折线图层 | — |
| 输出缓冲要素类 | 输出路径 | — |
| 缓冲比例 | 每侧局部半宽的占比（0 < r ≤ 1） | `0.5` |
| 采样距离 | 中心线采样间距（坐标系单位） | 自动 |
| 端盖样式 | `ROUND`（圆形）或 `FLAT`（平直） | `ROUND` |
| 裁剪至多边形 | 将输出裁剪至源多边形边界 | 勾选 |
| Chaikin 平滑迭代次数 | 0 = 不平滑 | `0` |

加载工具箱：**目录窗格 → 添加工具箱 → `Proportional_Buffer.pyt`**。
无需额外依赖，任何 ArcGIS Pro 安装均可直接使用。

---

## 仓库结构

```
Enhanced-Polygon-to-Centerline/
├── degree_centerline/          最新中心线提取工具箱（推荐使用）
│   ├── Degree_Centerline.pyt       单次处理 ArcGIS 工具箱
│   ├── centerline_degree.py        核心算法
│   ├── Split_and_Process.pyt       大型数据集分块处理工具箱
│   ├── split_and_process.py        分块包装器
│   ├── install_dependencies.bat    一键安装依赖
│   ├── requirements.txt
│   ├── HOW_IT_WORKS.md             详细算法说明（中文）
│   └── README.md
├── proportional_buffer/        变宽度缓冲区工具集（推荐使用）
│   ├── arcpy_toolbox/
│   │   └── Proportional_Buffer.pyt ArcGIS 工具箱（独立）
│   ├── proportional_buffer.py      核心算法（shapely/numpy/scipy）
│   ├── cli.py                      命令行接口
│   ├── requirements.txt
│   └── README.md
├── downloads/                  已打包 zip 文件（可直接下载）
│   ├── Degree_Centerline_Toolbox.zip
│   └── Proportional_Buffer_Toolbox.zip
├── archive/                    早期实验版本（仅供参考）
│   ├── arcpy_toolbox/
│   ├── auto_centerline/
│   ├── fast_centerline/
│   ├── gdal_centerline/
│   ├── pure_centerline/
│   └── steiner_centerline/
└── README.md                   英文主说明
```

---

## 示例效果

![中心线提取示例](https://github.com/user-attachments/assets/b360be4f-c1d0-4add-b5a0-dd552580c379)

*示例：从弯曲河流多边形中提取的度数感知分叉中心线。所有有意义的支流交汇点均被保留；短噪声毛刺被自动过滤。*

---

## 许可证

MIT License — 可自由使用、修改和分发。详见仓库根目录的 LICENSE 文件（如有）。
