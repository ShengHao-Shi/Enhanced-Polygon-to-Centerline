# Auto-Threshold Branching Centerline — Approach D

This folder contains an **auto-threshold branching centerline** extraction
algorithm, packaged as an ArcGIS Python Toolbox (`.pyt`).

## What is this and why does it exist?

| Folder | Approach | Output |
|---|---|---|
| `fast_centerline/` | Longest single path | One LINESTRING per polygon |
| **`auto_centerline/`** | **Auto-threshold prune** | **Branching MULTILINESTRING** |

The `fast_centerline/` tool extracts only the single longest path through the
Voronoi/skeleton graph (`_extract_longest_path`).  This discards all branches,
which is fine for simple elongated shapes but loses important structural
information for polygons with meaningful branches (e.g. river deltas, road
junctions, T-shaped buildings).

**`auto_centerline/`** automatically computes a pruning threshold based on
the polygon's geometry, removes short noise branches (Voronoi artefacts near
corners), and returns the **full remaining skeleton** as a branching
MULTILINESTRING.

---

## How automatic threshold pruning works

The automatic threshold is computed as:

```
threshold = max(strategy_1, strategy_2)
```

Where:

- **Strategy 1** (primary — width-based):
  ```
  typical_width = polygon_area / total_skeleton_length
  threshold = typical_width × 0.5
  ```
  This adapts to the polygon's scale and shape.  Narrow polygons get a
  small threshold; wide polygons get a larger one.

- **Strategy 2** (floor — distance-based):
  ```
  threshold = densify_distance × 3.0
  ```
  This ensures a minimum pruning level regardless of polygon shape,
  removing the very short noise branches that arise from Voronoi
  tessellation near polygon corners.

The effective threshold is: `max(auto_threshold, user_prune_threshold)`.
If the user provides a non-zero prune threshold, it serves as a minimum —
the auto-computed value may be larger.

---

## How it differs from fast_centerline

| Feature | `fast_centerline/` | `auto_centerline/` |
|---|---|---|
| Default output | Single longest path (LINESTRING) | Branching skeleton (MULTILINESTRING) |
| Pruning | Manual threshold only | **Automatic** + optional manual minimum |
| Branch preservation | ❌ Discards all branches | ✅ Preserves meaningful branches |
| Noise removal | Manual or none | **Automatic** based on geometry |
| Speed | Same | Same (identical vectorised core) |
| Dependencies | Same | Same |

---

## Dependencies

| Package | Availability in ArcGIS Pro | Needed for |
|---|---|---|
| **numpy** | ✅ Pre-installed | All vectorised arithmetic |
| **scipy** | ✅ Usually pre-installed | Voronoi tessellation |
| **networkx** | ⬇ Install from conda-forge | Graph skeleton extraction |
| **matplotlib** | ✅ Usually pre-installed | Fastest rasterisation (optional) |
| **scikit-image** | ⬇ Optional, conda-forge | `method=skeleton` only |

Same dependencies as `fast_centerline/`.

---

## Installation

```
conda install -c conda-forge -y networkx
```

If the default environment is read-only:
```
conda create --name arcgispro-py3-auto --clone arcgispro-py3
activate arcgispro-py3-auto
conda install -c conda-forge -y networkx
```
Then in ArcGIS Pro: **Project → Python → Python Environments → arcgispro-py3-auto**.

Or double-click `install_dependencies.bat` for a guided installation.

---

## Loading in ArcGIS Pro

1. Open the **Catalog** pane.
2. Right-click a folder → **Add Toolbox**.
3. Select **`Auto_Centerline.pyt`** in this folder.
4. Expand the toolbox and run **Polygon to Centerline (Auto Branching)**.

Both `Auto_Centerline.pyt` and `centerline_auto.py` must be in the **same folder**.

---

## Parameters

| Parameter | Description |
|---|---|
| **Input Polygon Features** | Polygon feature layer or feature class |
| **Output Centerline Features** | Output polyline feature class path |
| **Method** | `voronoi` (default) or `skeleton` |
| **Densification Distance** | Max vertex spacing (CRS units) before Voronoi |
| **Min Branch Prune Threshold** | Minimum prune threshold (auto-computed if 0; CRS units) |
| **Gaussian Smooth Sigma** | Blur before skeletonisation (skeleton only) |
| **Raster Resolution Override** | Pixel size for skeleton method (blank = auto) |
| **Return Full Raw Skeleton** | Unchecked = auto-pruned branching; checked = all branches |

---

## File listing

| File | Description |
|---|---|
| `Auto_Centerline.pyt` | ArcGIS Python Toolbox |
| `centerline_auto.py` | Auto-threshold branching centerline algorithm |
| `install_dependencies.bat` | Windows helper — one conda command |
| `requirements.txt` | Dependency list |
| `README.md` | This file |
