# Proportional Buffer Along Centerlines

A self-contained toolkit that generates **variable-width buffer polygons**
along pre-existing centerlines.  
The buffer width at every cross-section equals  
**`buffer_ratio × 2 × local_channel_width`**  
where the local channel width is measured at each sample point as the
distance to the nearest polygon boundary.

---

## Key design principles

| Principle | Detail |
|---|---|
| **Independent of centerline extraction** | Takes polygon + centerline as inputs; does not call any centerline-extraction code |
| **No metadata required** | Width is measured geometrically at runtime — no pre-computed attributes needed |
| **Two implementations** | Pure Python (shapely/geopandas) for CLI/scripting; ArcGIS Python Toolbox (arcpy + numpy/scipy) for ArcGIS Pro |
| **Scalable** | Boundary distances via `scipy.spatial.cKDTree` — O(N log M), handles national-scale datasets |

---

## Files

```
proportional_buffer/
├── proportional_buffer.py        # Core algorithm (shapely + numpy + scipy)
├── cli.py                        # Command-line interface
├── requirements.txt              # Python dependencies
├── arcpy_toolbox/
│   └── Proportional_Buffer.pyt  # ArcGIS Pro Python Toolbox
└── tests/
    └── test_proportional_buffer.py
```

---

## Algorithm

```
For each polygon / centerline pair:

  1. SAMPLE   Resample centerline at uniform intervals (sample_distance)
                → ordered point sequence [P₀, P₁, …, Pₙ]

  2. WIDTH    Build KDTree from densified polygon boundary.
              Query → local half-width hw_i at each Pᵢ

  3. NORMAL   Compute unit normal vectors nᵢ (⊥ to travel direction)
              Interior points: central differences
              Endpoints: one-sided differences

  4. OFFSET   radius rᵢ = buffer_ratio × hw_i   (clamped by width_min/max)
              Left side:  Lᵢ = Pᵢ + rᵢ × nᵢ
              Right side: Rᵢ = Pᵢ – rᵢ × nᵢ

  5. ASSEMBLE Forward left [L₀…Lₙ]
              → front end-cap (round or flat)
              → backward right [Rₙ…R₀]
              → back end-cap
              → close ring

  6. POST     buffer(0) to fix self-intersections
              Optional: clip to polygon, Chaikin smoothing
```

---

## Quick start — pure Python

### Install

```bash
pip install numpy scipy shapely geopandas
```

### Python API

```python
from shapely.geometry import Polygon, LineString
from proportional_buffer import compute_proportional_buffer

poly = Polygon([(-50, -5), (50, -5), (50, 5), (-50, 5)])
cl   = LineString([(-50, 0), (50, 0)])

buf = compute_proportional_buffer(
    polygon=poly,
    centerline=cl,
    buffer_ratio=0.5,       # 50% of local channel width
    sample_distance=1.0,    # 1 m sampling interval
    end_cap="round",
    clip_to_polygon=True,
)
print(buf.wkt)
```

### GeoDataFrame batch processing

```python
import geopandas as gpd
from proportional_buffer import process_geodataframes

poly_gdf = gpd.read_file("waterways.gpkg", layer="polygons")
cl_gdf   = gpd.read_file("waterways.gpkg", layer="centerlines")

out_gdf = process_geodataframes(
    poly_gdf,
    cl_gdf,
    buffer_ratio=0.5,
    sample_distance=5.0,    # 5 m intervals
    clip_to_polygon=True,
    smooth_tolerance=2,     # 2 Chaikin iterations
)
out_gdf.to_file("output.gpkg", layer="buffers")
```

### Command line

```bash
python cli.py \
  --polygon  waterways.gpkg --polygon-layer  polygons    \
  --centerline waterways.gpkg --centerline-layer centerlines \
  --output   buffers.gpkg   --output-layer   buffers     \
  --ratio 0.5 --end-cap round --smooth 2

# Full options
python cli.py --help
```

---

## Quick start — ArcGIS Pro

### Dependencies

Only `numpy` and `scipy` are required — both are pre-installed in ArcGIS Pro's
default Python environment.  No shapely or geopandas needed.

### Load the toolbox

1. In the **Catalog** pane, right-click a folder → **Add Toolbox**.  
2. Browse to `arcpy_toolbox/Proportional_Buffer.pyt` and click **OK**.  
3. Expand the toolbox and run **Proportional Buffer Along Centerlines**.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| Input Polygon Features | — | Waterway / channel polygon layer |
| Input Centerline Features | — | Pre-computed centerline polyline layer |
| Output Buffer Feature Class | — | Output path |
| Buffer Ratio | 0.5 | Fraction of local half-width used as buffer radius (0 < ratio ≤ 1) |
| Sample Distance | auto | Centerline sampling interval (blank = automatic) |
| End-Cap Style | ROUND | `ROUND` or `FLAT` |
| Clip Buffer to Polygon Boundary | True | Clip output to source polygon |
| Chaikin Smoothing Iterations | 0 | 0 = no smoothing |
| Minimum Buffer Width | 0 Meters | Prevents degenerate slivers |
| Maximum Buffer Width | — | Caps width at wide sections |

---

## Parameters reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `buffer_ratio` | float | 0.5 | Fraction of local half-width per side (0 < r ≤ 1) |
| `sample_distance` | float | auto | Centerline sample interval (CRS units) |
| `end_cap` | str | `"round"` | `"round"` or `"flat"` |
| `clip_to_polygon` | bool | `True` | Clip result to source polygon |
| `smooth_tolerance` | float | 0 | Chaikin iterations (rounded to int) |
| `width_min` | float | 0 | Minimum buffer width (CRS units) |
| `width_max` | float | ∞ | Maximum buffer width (CRS units) |

---

## Output attributes

| Field | Description |
|---|---|
| `src_id` | Source polygon identifier |
| `buffer_ratio` | Buffer ratio used |
| `width_min_m` | Minimum channel width along the centerline |
| `width_max_m` | Maximum channel width |
| `width_mean_m` | Mean channel width |

---

## Running tests

```bash
# From the repository root
pip install pytest numpy scipy shapely geopandas
pytest proportional_buffer/tests/ -v
```

---

## Performance notes

The main cost is the KDTree distance query in Step 2.  
The boundary tree is built **once per polygon** (O(M log M) for M boundary
vertices) and all sample points are queried in a single batch call
(O(N log M) for N sample points).  This is >100× faster than a brute-force
per-point distance loop.

For national-scale datasets with many polygons:
- Increase `sample_distance` to reduce the number of sample points.
- The `process_geodataframes()` function processes features sequentially;
  wrap it with `concurrent.futures.ProcessPoolExecutor` for parallel execution.

---

## Relation to other tools in this repository

This toolkit is **fully independent**.  It does not import or call any code
from `centerline_pure.py`, `arcpy_toolbox/`, or any other subfolder.

Typical workflow:

```
Step 1  Run any centerline-extraction tool (e.g. Pure_Centerline.pyt)
        → produces centerline polylines

Step 2  Run Proportional Buffer (this tool)
        Inputs : original polygon + centerlines from Step 1
        Output : variable-width buffer polygons
```
