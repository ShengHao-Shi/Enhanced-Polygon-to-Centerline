# Steiner Tree Branching Centerline

**Approach A** — Preserves ALL meaningful branches of the polygon's
medial-axis centerline using a Steiner tree approximation.

## How it differs from `fast_centerline/`

| Feature | `fast_centerline` | `steiner_centerline` |
|---|---|---|
| Extraction strategy | `_extract_longest_path` — returns ONE path between the two farthest leaf nodes | `_extract_steiner_tree` — connects ALL leaf nodes via a Steiner tree |
| Branch handling | All branches are **discarded** | All meaningful branches are **preserved** |
| Output geometry | `LINESTRING` (single path) | `MULTILINESTRING` (branching tree) |
| Noise filtering | User-specified `prune_threshold` only | Automatic pre-prune (3 × densify_distance) + user `prune_threshold` |

When `single_line=True` (the default), `fast_centerline` finds the two
farthest endpoints and returns the shortest weighted path between them —
every side-branch is lost.  `steiner_centerline` instead computes an
approximate Steiner tree that spans **all** leaf nodes, preserving the full
branching topology of the skeleton.

### Algorithm steps (`_extract_steiner_tree`)

1. **Pre-prune** short noise branches (length < 3 × `densify_distance`)
2. Take the **largest connected component**
3. Identify all **leaf nodes** (degree-1 nodes in the graph)
4. Compute an **approximate Steiner tree** connecting all leaves
   (`networkx.approximation.steiner_tree`)
5. Return all edges of the Steiner tree

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
2. `cd` to the `steiner_centerline/` folder.
3. Run `install_dependencies.bat`.

### Manual installation

```
conda create --name arcgispro-py3-steiner --clone arcgispro-py3
activate arcgispro-py3-steiner
conda install -c conda-forge -y networkx
```

Then set `arcgispro-py3-steiner` as the active environment in ArcGIS Pro
(Project → Python → Python Environments).

## Loading the toolbox

In **ArcGIS Pro** (Catalog pane):

1. Right-click a folder → **Add Toolbox** → select `Steiner_Centerline.pyt`.
2. Expand the toolbox and run **Polygon to Centerline (Steiner Tree)**.

> `centerline_steiner.py` must be in the **same directory** as
> `Steiner_Centerline.pyt`.

## Parameters

All parameters are identical to `fast_centerline/`, with one difference:

- **Return Full Raw Skeleton** — when checked, returns the complete raw
  skeleton graph (ignoring the Steiner tree filtering), which may include
  all noise branches.  When unchecked (default), the Steiner tree is used
  to produce a clean branching centerline.
