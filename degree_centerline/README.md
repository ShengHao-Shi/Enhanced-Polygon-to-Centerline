# Degree-Aware Branching Centerline

**Approach C** from `fast_centerline/BRANCHING_ANALYSIS.md` — preserves ALL
meaningful branches using topological segment decomposition with O(V+E)
linear-time graph traversal.

## How it differs from other toolsets

| Feature | `fast_centerline` | `steiner_centerline` | **`degree_centerline`** |
|---|---|---|---|
| Extraction strategy | `_extract_longest_path` — ONE path between the two farthest leaf nodes | `_extract_steiner_tree` — Steiner tree connecting all leaves | **`_extract_branching_skeleton`** — degree-aware segment decomposition |
| Branch handling | All branches **discarded** | All meaningful branches **preserved** | All meaningful branches **preserved** |
| Output geometry | `LINESTRING` | `MULTILINESTRING` | `MULTILINESTRING` |
| Algorithm complexity | O(V log V) two-pass Dijkstra | O(T²·V·log V) Steiner approx. | **O(V + E) linear** |
| Noise filtering | User `prune_threshold` only | Pre-prune 3×densify + Steiner | **Pre-prune + ratio-based filter** |
| Speed | Fastest (single path) | Slowest (Steiner tree) | **Fast (linear traversal)** |

## Algorithm (`_extract_branching_skeleton`)

1. **Pre-prune** short noise spurs (length < 3 × `densify_distance`)
2. Take the **largest connected component**
3. Identify **key nodes**: leaves (degree 1) and junctions (degree ≥ 3)
4. **Decompose** the graph into segments — chains of degree-2 nodes
   connecting pairs of key nodes
5. **Filter** short terminal segments using a ratio threshold
   (`min_branch_ratio` × longest segment length, default 10%)
6. Always **keep** internal segments (junction → junction)
7. Return all surviving edges as `MULTILINESTRING`

```
Original skeleton graph:

  L1──s1──B1──s2──J1──s3──B2──s4──L2
                   │
                  s5
                   │
                  B3
                   │
                  s6
                   │
                  L3

  L = leaf (degree 1)    J = junction (degree ≥ 3)
  B = chain node (degree 2)    s = edge segment

Decomposed into segments:
  Segment 1: L1 → J1  (length = s1 + s2)
  Segment 2: J1 → L2  (length = s3 + s4)
  Segment 3: J1 → L3  (length = s5 + s6)

After filtering (if all segments are long enough):
  → All three branches preserved as MULTILINESTRING
```

## Dependencies

Same as `fast_centerline/`:

| Package | Required? | Notes |
|---|---|---|
| `numpy` ≥ 1.24 | **Yes** | Pre-installed in ArcGIS Pro |
| `scipy` ≥ 1.10 | **Yes** | Usually pre-installed in ArcGIS Pro |
| `networkx` ≥ 3.0 | **Yes** | Install via `conda install -c conda-forge networkx` |
| `matplotlib` ≥ 3.5 | Optional | Accelerates rasterisation; usually pre-installed |
| `scikit-image` ≥ 0.21 | Optional | Only needed for `method="skeleton"` |

## Installation

### Quick start

1. Open the **ArcGIS Pro Python Command Prompt**.
2. `cd` to the `degree_centerline/` folder.
3. Run `install_dependencies.bat`.

### Manual installation

```
conda create --name arcgispro-py3-degree --clone arcgispro-py3
activate arcgispro-py3-degree
conda install -c conda-forge -y networkx
```

Then set `arcgispro-py3-degree` as the active environment in ArcGIS Pro
(Project → Python → Python Environments).

## Loading the toolbox

In **ArcGIS Pro** (Catalog pane):

1. Right-click a folder → **Add Toolbox** → select `Degree_Centerline.pyt`.
2. Expand the toolbox and run **Polygon to Centerline (Degree-Aware)**.

> `centerline_degree.py` must be in the **same directory** as
> `Degree_Centerline.pyt`.

## Parameters

All parameters are identical to `fast_centerline/`, with one difference:

- **Return Full Raw Skeleton** — when checked, returns the complete raw
  skeleton graph (skipping degree-aware filtering), which may include
  all noise branches.  When unchecked (default), the degree-aware
  decomposition produces a clean branching centerline.

---

## Tiled extraction for large / complex datasets

For country-scale or otherwise very large polygon datasets (e.g. all Canadian
waterways) the single-pass algorithm can miss long narrow channels and run out
of practical memory.  The companion module `split_and_process.py` implements a
three-phase strategy to work around both limitations:

| Phase | What it does |
|---|---|
| **A** Connected-component split | Each MULTIPOLYGON part is processed independently; unconnected water bodies cannot affect each other's skeleton graph. |
| **B** Adaptive quad-tree subdivision | Polygon parts exceeding a vertex-count threshold are recursively split into four quadrant tiles until each tile is simple enough to process reliably. |
| **C** Overlap-buffer extraction | Each tile is expanded outward by `buffer_factor × densify_distance` before clipping; this prevents the Voronoi algorithm from producing spurious branches at tile seams.  The extracted centerline is clipped back to the original tile boundary before merging. |

### Python API

```python
from split_and_process import tile_and_extract_centerline

result_wkt = tile_and_extract_centerline(
    wkt,                  # WKT POLYGON or MULTIPOLYGON
    densify_distance=1.0,
    max_vertices=8000,    # Phase B: vertex threshold per tile
    max_bbox_area=None,   # Phase B: optional bbox area threshold
    buffer_factor=5.0,    # Phase C: buffer = factor × densify_distance
    max_depth=5,          # Phase B: max quad-tree depth
)
```

### ArcGIS toolbox

Load `Split_and_Process.pyt` in ArcGIS Pro (same steps as above) and run
**Polygon to Centerline (Tiled)**.  All tiling parameters are exposed as
tool parameters with descriptive help text.

---

## File listing

| File | Description |
|---|---|
| `Degree_Centerline.pyt` | ArcGIS Python Toolbox — single-pass extraction |
| `centerline_degree.py` | Degree-aware branching centerline algorithm |
| `Split_and_Process.pyt` | ArcGIS Python Toolbox — tiled extraction for large datasets |
| `split_and_process.py` | Three-phase tiling wrapper around `centerline_degree.py` |
| `install_dependencies.bat` | Windows helper — one conda command |
| `requirements.txt` | Dependency list |
| `README.md` | This file |
