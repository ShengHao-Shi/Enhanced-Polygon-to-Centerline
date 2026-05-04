# gdal_centerline

An **open-source, ArcPy-free** Polygon-to-Centerline implementation using
standard geospatial and scientific Python libraries available in **Anaconda /
conda-forge**.

---

## Dependencies

| Library | Purpose | conda-forge |
|---|---|---|
| `numpy` | Array arithmetic | `conda install numpy` |
| `scipy` | Voronoi tessellation | `conda install scipy` |
| `shapely` | Geometry predicates & operations | `conda install shapely` |
| `geopandas` | Vector file I/O (Shapefile, GeoJSON, GPKG) | `conda install geopandas` |
| `scikit-image` | Morphological skeletonisation | `conda install scikit-image` |
| `networkx` | Graph-based branch pruning | `conda install networkx` |
| `matplotlib` | (Optional) visualisation | `conda install matplotlib` |

```bash
# Install everything at once
conda install -c conda-forge numpy scipy shapely geopandas scikit-image networkx matplotlib
# or
pip install -r requirements.txt
```

---

## Quick Start

### As a library

```python
import geopandas as gpd
from centerline import polygon_to_centerline

# Read any polygon vector file
gdf = gpd.read_file("my_polygons.geojson")

# Extract centerlines (Voronoi method — see algorithm below)
centerlines = polygon_to_centerline(gdf, method="voronoi", densify_distance=1.0)

# Save result
centerlines.to_file("my_centerlines.geojson", driver="GeoJSON")
```

### From the command line

```bash
# Voronoi method (default) — outputs a single non-branching LineString
python cli.py input.geojson output.geojson

# Voronoi with custom densification and pruning
python cli.py rivers.gpkg centerlines.gpkg --densify 0.5 --prune 5.0

# Raster skeleton method
python cli.py roads.shp roads_cl.shp --method skeleton --densify 1.0 --smooth 2.0

# Return the full skeleton (may contain forks) instead of a single line
python cli.py input.geojson output.geojson --multi-line

# All options
python cli.py --help
```

---

## Algorithm Explanation

Two independent methods are provided.  Choose based on your use case.

---

### Method A — Voronoi Skeleton (`method="voronoi"`)

This is a **vector-based** approach that mirrors the ArcPy Thiessen-polygon
workflow but uses only open-source tools.

#### Steps

```
Input polygon
      │
      ▼
① Densify boundary
      │  Insert extra vertices along the polygon outline until
      │  consecutive points are at most densify_distance apart.
      │  Purpose: more boundary points → finer Voronoi skeleton.
      │
      ▼
② Voronoi tessellation  (scipy.spatial.Voronoi)
      │  Compute the Voronoi diagram of all boundary points.
      │  Each Voronoi cell contains exactly one boundary point and
      │  encompasses the region of space closest to that point.
      │  The ridges (edges) between adjacent cells form the raw skeleton.
      │
      ▼
③ Filter interior ridges  (two-stage, ring-aware)
      │  Stage A — Generator-distance filter (same-ring artefacts only):
      │    Each Voronoi ridge is defined by two "generating" boundary
      │    points (one per cell).  If those two generators are closer than
      │    3 × densify_distance AND they lie on the SAME boundary ring
      │    (exterior↔exterior or hole↔hole), their ridge runs parallel to
      │    the boundary (an artefact) and is discarded.
      │    Ridges between the OUTER boundary and an INNER hole boundary
      │    are NOT distance-filtered; this preserves the centerline of
      │    narrow ring-shaped polygons (letter 'O') and thin-walled holes
      │    (letter 'A').
      │  Stage B — Full-segment containment:
      │    The entire ridge segment must lie inside the polygon
      │    (shapely.Polygon.contains).  This automatically excludes ridges
      │    that cross through any hole interior.
      │
      ▼
④ Build graph  (networkx.Graph)
      │  The surviving ridges become edges in a graph; each Voronoi vertex
      │  is a node.  Edge weights equal Euclidean length.
      │
      ▼
⑤ Prune dead-end branches  (optional)
      │  Paths starting at degree-1 leaves are walked toward the first
      │  junction; if the total path length < prune_threshold the entire
      │  branch is removed.  This eliminates end-cap spurs.
      │
      ▼
⑥ Extract single trunk  (networkx + two-pass Dijkstra / cycle traversal)
      │  For tree-shaped skeletons (open polygons):
      │  • Pass 1 — from an arbitrary leaf, find the farthest leaf u.
      │  • Pass 2 — from u, find the farthest leaf v.
      │  • The path u→v is the weighted diameter (the trunk).
      │  For cycle-shaped skeletons (ring polygons, e.g. letter 'O'):
      │  • All nodes have degree 2 → no leaves exist.
      │  • The full closed loop is traversed and returned as a closed
      │    LineString (first coordinate == last coordinate).
      │  Set single_line=False to skip this step and keep the full skeleton.
      │
      ▼
Output centerline (single LineString, no branches; closed loop for rings)
```

#### Why does the Voronoi diagram approximate the medial axis?

The **medial axis** of a shape is the locus of all points that are
equidistant from **two or more** boundary locations.  By definition, every
Voronoi ridge separating two cells is equidistant from the two generating
points (one per cell).  When those generating points are sampled densely and
uniformly from the polygon boundary, the set of interior Voronoi ridges
converges to the true medial axis of the polygon.

```
┌─────────────────────────────────────────┐
│  Polygon boundary (densified vertices •) │
│  ···•···•···•···•···•···•···•···•···•   │
│                                          │
│  Voronoi ridges inside polygon  ─────   │
│  (= approximate medial axis)            │
│                                          │
└─────────────────────────────────────────┘
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `densify_distance` | float | 1.0 | Max spacing between boundary vertices (CRS units). **Reduce** this for a more detailed / accurate skeleton. |
| `prune_threshold` | float | 0.0 | Remove branches shorter than this value. 0 = no pruning. |
| `single_line` | bool | `True` | When True (default), return a single non-branching LineString (longest trunk path). Set to False to return the full skeleton (may contain forks). |

#### Tips

* Start with `densify_distance ≈ width / 5` where *width* is the polygon's
  narrowest cross-section.
* If the skeleton looks spiky, increase `prune_threshold` to `width / 2`.
* For projected CRS (metres), values like `0.5` or `1.0` are typical.
* For geographic CRS (degrees), values like `0.00001` may be needed.

---

### Method B — Raster Skeleton (`method="skeleton"`)

This is a **raster-based** approach using morphological image-processing
techniques from `scikit-image`.

#### Steps

```
Input polygon
      │
      ▼
① Rasterise  (numpy grid)
      │  Convert the polygon to a binary 2-D numpy array at a given cell
      │  size (resolution).  Pixels whose centre lies inside the polygon
      │  are set to 1; all others are 0.
      │
      ▼
② Gaussian blur  (scipy.ndimage.gaussian_filter, optional)
      │  Smooth the binary image to reduce jagged boundary artefacts
      │  before thinning.  Controlled by smooth_sigma (in CRS units).
      │
      ▼
③ Morphological skeletonisation  (skimage.morphology.skeletonize)
      │  Apply Zhang–Suen thinning:  iteratively erode the binary image,
      │  removing foreground pixels from the border only if doing so does
      │  not break 8-connectivity.  Continues until no more pixels can be
      │  removed.  The result is a 1-pixel-wide skeleton.
      │
      ▼
④ Build weighted pixel graph  (networkx)
      │  Build a graph from the skeleton pixels (nodes) and their
      │  8-connected neighbours (edges).  Edge weights = Euclidean distance
      │  between pixel centres (1 for cardinal, √2 for diagonal).
      │
      ▼
⑤ Extract single trunk  (two-pass Dijkstra)
      │  Same diameter algorithm as the Voronoi method: find the two
      │  leaf pixels farthest apart by summed edge weight and return only
      │  the path between them as a single LineString.
      │  Set single_line=False to keep all pixel-pair edges.
      │
      ▼
Output centerline (single LineString, no branches)
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `densify_distance` | float | 1.0 | Used as raster cell size (when `raster_resolution` is not set). |
| `raster_resolution` | float | None | Explicit raster cell size override. |
| `smooth_sigma` | float | 0.0 | Gaussian smoothing σ (CRS units) applied before thinning. |
| `single_line` | bool | `True` | When True (default), return a single non-branching LineString. |

#### Tips

* Choose a cell size ≈ 1/10 of the polygon width for a good balance between
  accuracy and memory usage.
* Increase `smooth_sigma` to reduce noise from irregular boundaries.
* The output coordinates are pixel-centre coordinates — expect a slight
  quantisation effect at the chosen resolution.

---

## Complex Polygon Shapes

The algorithm handles non-trivial polygon geometries:

### Ring / Donut polygons ("letter O")

A ring polygon has an exterior ring *and* an interior hole.  The hole
represents empty space that the centerline must **not** cross.

```python
from shapely.geometry import Point
outer = Point(0, 0).buffer(30)
inner = Point(0, 0).buffer(20)
ring  = outer.difference(inner)          # donut, width = 10

gdf = gpd.GeoDataFrame(geometry=[ring])
cl  = polygon_to_centerline(gdf, densify_distance=1.0)
# → closed LineString at radius 25 (midpoint of 10…30)
```

**Behaviour:**
* Interior hole boundary points are densified and their ring index is
  tracked.
* The generator-distance artefact filter is applied only to **same-ring**
  pairs, so cross-ring ridges (outer↔inner) are preserved even for narrow
  rings (ring_width < 3 × densify_distance).
* Ridges that would cross through the hollow are rejected by
  `polygon.contains()` regardless.
* `single_line=True` detects the cycle-shaped skeleton and returns the
  **full closed ring** as a closed `LineString` (first == last coordinate).

### Polygons with internal holes ("letter A")

An 'A'-shaped polygon has a triangular hole in its interior.

```python
outer = Polygon([(-10,0),(10,0),(0,30)])
hole  = Polygon([(-4,10),(4,10),(0,18)])
A_poly = Polygon(outer.exterior, [hole.exterior])

gdf = gpd.GeoDataFrame(geometry=[A_poly])
cl_single = polygon_to_centerline(gdf, densify_distance=0.5)
# → single longest-path LineString, never crosses the hole
cl_full   = polygon_to_centerline(gdf, densify_distance=0.5, single_line=False)
# → MultiLineString with all skeleton branches (no holes crossed)
```

**Behaviour:**
* The hole boundary is densified alongside the exterior; Voronoi ridges in
  the wall between exterior and hole are kept (they ARE the centerline).
* Any ridge passing through the hole is removed by `polygon.contains()`.
* `single_line=True` returns the longest path, which represents the main
  spine.  Use `single_line=False` for a complete multi-branch skeleton that
  includes all structural arms.

---

## Comparison of the Two Methods

| Aspect | Voronoi (`"voronoi"`) | Skeleton (`"skeleton"`) |
|---|---|---|
| Data model | Pure vector | Raster → vector |
| CRS accuracy | Exact (floating point) | Quantised to cell size |
| Speed | Fast for simple polygons | Scales with raster size |
| Handles holes | Yes | Yes |
| Handles very jagged shapes | May need pruning | Gaussian smoothing helps |
| Output (default) | Single `LineString` (no forks) | Single `LineString` (no forks) |
| Best for | Road / river / building footprints | Complex organic shapes |
| Key parameters | `densify_distance`, `prune_threshold`, `single_line` | `raster_resolution`, `smooth_sigma`, `single_line` |

---

## File structure

```
gdal_centerline/
├── centerline.py                    Core algorithm (importable module)
├── cli.py                           Command-line interface
├── GDAL_Centerline.pyt              ArcGIS Python Toolbox (wraps centerline.py)
├── install_dependencies.bat         Helper script to install required packages (Windows)
├── requirements.txt                 pip-compatible dependency list
└── README.md                        This file
```

---

## ArcGIS Toolbox (`GDAL_Centerline.pyt`)

`GDAL_Centerline.pyt` is an **ArcGIS Python Toolbox** that provides a
graphical dialog interface for `centerline.py` inside ArcGIS Pro or ArcCatalog.

### Advantages over the ArcPy toolbox

| Feature | ArcPy toolbox | GDAL toolbox |
|---|---|---|
| Required ArcGIS licence | **Standard or Advanced** | **Basic (any level)** |
| Algorithm | Single Voronoi/Thiessen method | Voronoi *and* Skeleton |
| Handles holes / rings | Partial | Yes (letter 'O', 'A', …) |
| Dependencies | Only arcpy | geopandas, shapely, scipy, networkx |

### How to load

1. In **ArcGIS Pro** Catalog pane (or **ArcCatalog**), right-click the
   `gdal_centerline/` folder → **Add Toolbox** → select `GDAL_Centerline.pyt`.
2. Expand the toolbox and double-click **Polygon to Centerline (GDAL)**.

`centerline.py` **must be in the same directory** as `GDAL_Centerline.pyt`
(they are already co-located in this folder).

### Tool parameters

| # | Parameter | Default | Notes |
|---|---|---|---|
| 1 | Input Polygon Features | — | Any polygon layer / feature class |
| 2 | Output Centerline Features | — | New polyline feature class |
| 3 | Method | `voronoi` | `voronoi` or `skeleton` |
| 4 | Densification Distance | `1.0` | CRS units; smaller = finer detail |
| 5 | Branch Prune Threshold | `0.0` | Voronoi only; removes short stubs |
| 6 | Gaussian Smooth Sigma | `0.0` | Skeleton only; 0 = no smoothing |
| 7 | Raster Resolution Override | *(blank)* | Skeleton only; blank = auto |
| 8 | Return Full Skeleton | ☐ (off) | When on, returns branches/forks |

Parameters 5 (Prune Threshold) and 6–7 (Skeleton options) are automatically
**enabled or disabled** based on the selected Method.

### Output fields

The output feature class contains all attribute fields from the input polygon
plus one additional field:

| Field | Type | Description |
|---|---|---|
| `ORIG_FID` | Long | Object ID of the source polygon feature |

### Installing dependencies in ArcGIS Pro

> **Important:** ArcGIS Pro installs Python into a read-only conda environment
> (`arcgispro-py3`).  You must **clone** it before installing packages.
> The included `install_dependencies.bat` handles this for you.

#### Option A — Automatic (recommended)

1. Open **Start → ArcGIS → ArcGIS Pro Python Command Prompt**.
2. `cd` into the `gdal_centerline/` folder.
3. Run `install_dependencies.bat`.

The script detects whether your environment is writable, installs the packages,
and prints clear guidance if a clone is needed.

#### Option B — Manual (ArcGIS Pro Python Command Prompt)

```bat
REM Step 1 — Clone the default environment (only once)
conda create --name arcgispro-py3-gdal --clone arcgispro-py3

REM Step 2 — Install packages into the clone
activate arcgispro-py3-gdal
conda install -c conda-forge -y shapely geopandas scipy networkx scikit-image
```

Then in ArcGIS Pro:  
**Project → Python → Python Environments → arcgispro-py3-gdal → OK**  
Restart ArcGIS Pro.

#### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `CondaError: Read-only` | Default env is read-only | Clone it first (see Step 1 above) |
| `No module named 'shapely'` | Packages not installed, or wrong env is active | Complete Option A or B above |
| Install succeeds but tool still fails | ArcGIS Pro is using the old environment | Set the new env in **Project → Python** and restart |
