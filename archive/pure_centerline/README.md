# Pure Centerline — Shapely-Free ArcGIS Toolbox

This folder contains a **shapely-free** reimplementation of the polygon-to-centerline algorithm, packaged as an ArcGIS Python Toolbox (`.pyt`).

## Why a separate toolbox?

The `gdal_centerline/GDAL_Centerline.pyt` toolbox depends on **shapely ≥ 2.0**, which — on some ArcGIS Pro installations — triggers licence-conflict errors in the conda environment that make ArcGIS Pro display **"Tool not licensed"** even when the user holds valid licences (Basic + Spatial Analyst, etc.).

This toolbox replaces every shapely call with pure-numpy arithmetic:

| shapely call | replacement |
|---|---|
| `polygon.exterior.coords` | numpy ring arrays |
| `polygon.interiors` | list of numpy ring arrays |
| `polygon.contains(segment)` | midpoint ray-casting + segment-intersection test |
| `polygon.contains(point)` | vectorised ray-casting (one raster row at a time) |
| `polygon.bounds` | `np.min / np.max` on coordinate arrays |
| `polygon.buffer(0)` | degenerate polygons silently skipped |
| `linemerge / unary_union` | WKT string construction |

The algorithms (Voronoi skeleton and raster skeletonisation) are identical to the original.

---

## Dependencies

| Package | Availability in ArcGIS Pro | Needed for |
|---|---|---|
| **numpy** | ✅ Pre-installed | All geometry arithmetic |
| **scipy** | ✅ Usually pre-installed | Voronoi tessellation |
| **networkx** | ⬇ Install from conda-forge | Graph skeleton extraction |
| **scikit-image** | ⬇ Optional, conda-forge | `method=skeleton` only |

**Only networkx typically needs to be installed.** This is a much lighter dependency footprint than the `gdal_centerline` toolbox (which also needed shapely, geopandas, and pandas).

---

## Installation

### Quickest path (one command)

Open the **ArcGIS Pro Python Command Prompt** and run:

```
conda install -c conda-forge -y networkx
```

If the default environment is read-only (ArcGIS installed in `C:\Program Files\`), clone it first:

```
conda create --name arcgispro-py3-pure --clone arcgispro-py3
activate arcgispro-py3-pure
conda install -c conda-forge -y networkx
```

Then in ArcGIS Pro: **Project → Python → Python Environments → arcgispro-py3-pure** and restart.

### Batch installer

Double-click `install_dependencies.bat` (or run it from the ArcGIS Pro Python Command Prompt) for a guided installation.

### Optional — Skeleton method

If you want to use `method=skeleton`, also install scikit-image:

```
conda install -c conda-forge -y scikit-image
```

---

## Loading the toolbox in ArcGIS Pro

1. Open the **Catalog** pane.
2. Right-click a folder → **Add Toolbox**.
3. Browse to this folder and select **`Pure_Centerline.pyt`**.
4. Expand the toolbox and run **Polygon to Centerline (Pure)**.

Both `Pure_Centerline.pyt` and `centerline_pure.py` must be in the **same folder**.

---

## Parameters

| Parameter | Description |
|---|---|
| **Input Polygon Features** | Polygon feature layer or feature class |
| **Output Centerline Features** | Output polyline feature class path |
| **Method** | `voronoi` (default) or `skeleton` |
| **Densification Distance** | Max vertex spacing (CRS units) before Voronoi; smaller = finer skeleton |
| **Branch Prune Threshold** | Remove dead-end branches shorter than this (0 = off; voronoi only) |
| **Gaussian Smooth Sigma** | Blur radius before skeletonisation (0 = off; skeleton only) |
| **Raster Resolution Override** | Pixel size for skeleton method; blank = auto (= densify distance) |
| **Return Full Skeleton** | Unchecked (default) = single trunk line; checked = all branches |

---

## Algorithms

### `voronoi` (default)

1. Densify polygon boundary so vertices are ≤ *densification distance* apart.
2. Compute the Voronoi tessellation of the boundary points (scipy).
3. Keep only Voronoi ridges that lie fully inside the polygon (pure-numpy containment test).
4. Build a NetworkX graph; iteratively prune dead-end branches.
5. Extract the single longest trunk path (two-pass Dijkstra diameter algorithm).

Works well for elongated polygons (rivers, roads, corridors). Handles polygons with holes (letter 'O', 'A' shapes) correctly.

### `skeleton`

1. Rasterise the polygon to a binary image using vectorised ray-casting (no shapely).
2. Optional Gaussian blur (`scipy.ndimage.gaussian_filter`).
3. Zhang–Suen morphological thinning (`skimage.morphology.skeletonize`).
4. Build a pixel graph; extract the longest trunk path.

Better for very jagged or organic shapes. Requires scikit-image.

---

## File listing

| File | Description |
|---|---|
| `Pure_Centerline.pyt` | ArcGIS Python Toolbox (tool dialog + execute logic) |
| `centerline_pure.py` | Shapely-free core algorithm (pure numpy / scipy / networkx) |
| `install_dependencies.bat` | Windows helper script to install networkx |
| `requirements.txt` | Dependency list (numpy, scipy, networkx) |
| `README.md` | This file |
