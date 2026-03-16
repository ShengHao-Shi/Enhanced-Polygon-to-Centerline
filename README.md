# Polygon to Centerline

An ArcGIS Python Toolbox (`.pyt`) that converts elongated polygon features
into polyline features depicting the **centerline** (medial axis / skeleton)
of each input polygon.

---

## Requirements

| Requirement | Version |
|---|---|
| ArcGIS Pro | 2.x or later |
| ArcGIS Desktop (ArcMap) | 10.x with **Standard** or **Advanced** license |
| Python | 3.x (bundled with ArcGIS Pro) |

The following ArcGIS toolboxes / licenses are used internally:

| Tool | Required license |
|---|---|
| `Densify (Edit)` | Standard / Advanced |
| `Feature Vertices To Points` | Standard / Advanced |
| `Create Thiessen Polygons` | Standard / Advanced |
| `Clip` | Basic |
| `Polygon To Line` | Standard / Advanced |
| `Smooth Line (Cartography)` | Standard / Advanced |

---

## Installation

1. Download or clone this repository.
2. In **ArcGIS Pro** (or **ArcCatalog / ArcMap**), use the **Add Toolbox**
   command and browse to `Polygon_to_Centerline.pyt`.
3. The toolbox **Polygon to Centerline** will appear in the *Geoprocessing*
   pane, ready to use.

---

## Tool: Polygon to Centerline

### Parameters

| # | Parameter | Type | Required | Default | Description |
|---|---|---|---|---|---|
| 1 | **Input Polygon Features** | Feature Layer (Polygon) | Yes | — | The elongated polygon features whose centerlines you want to extract. |
| 2 | **Output Centerline Features** | Feature Class (Polyline) | Yes | — | Path for the output polyline feature class. |
| 3 | **Densification Distance** | Linear Unit | No | `1 Meters` | Distance between added vertices along the polygon boundary. Smaller values produce a more detailed skeleton at the cost of longer processing time. |
| 4 | **Smoothing Tolerance** | Linear Unit | No | `0 Meters` | PAEK smoothing tolerance applied to the centerline. Set to `0` to skip smoothing. |

### Typical workflow

```
Input polygon feature class (roads, rivers, parcels …)
        │
        ▼
[Polygon to Centerline]
  ├─ Densification Distance : 1 Meters
  └─ Smoothing Tolerance    : 5 Meters
        │
        ▼
Output centerline polyline feature class
```

---

## Algorithm

The tool implements the **Voronoi / Thiessen skeleton** approach:

1. **Densify** — Additional vertices are inserted along the polygon boundary
   at the specified *Densification Distance* to ensure a dense set of sample
   points.
2. **Extract vertices** — All boundary vertices are converted to point
   features.
3. **Thiessen polygons** — ArcGIS `Create Thiessen Polygons` tessellates the
   study area such that every location is assigned to its nearest boundary
   point. The edges between Thiessen cells whose generating points both lie
   on the *same* polygon boundary approximate the **medial axis** of that
   polygon.
4. **Clip** — The Thiessen polygons are clipped to the footprint of the
   original polygon to discard cells that fall outside.
5. **Interior edges** — The clipped Thiessen polygons are converted to line
   features. Only edges shared by **two** Thiessen cells (`LEFT_FID ≥ 0 AND
   RIGHT_FID ≥ 0`) are retained — these are the skeleton / centerline edges.
6. **Smooth** *(optional)* — If a smoothing tolerance > 0 is specified, the
   PAEK algorithm is applied to produce a smoother curve.

### Tips for best results

* Choose a **Densification Distance** that is small relative to the width of
  the polygon (e.g. ≤ 1/5 of the narrowest cross-section).
* For very long or complex polygons, reduce the densification distance
  gradually until the skeleton is satisfactory.
* Apply **Smooth Line** or **Simplify Line** as a post-processing step if the
  raw skeleton is too jagged.
* The tool handles **multiple polygon features** in one run; each polygon is
  treated independently because the Thiessen polygons are clipped to the
  original polygon footprint.

---

## Example

```python
import arcpy
from Polygon_to_Centerline import _extract_centerline

arcpy.env.workspace = r"C:\data\my_project.gdb"

_extract_centerline(
    in_features      = "roads_polygon",
    out_features     = "roads_centerline",
    densify_distance = "0.5 Meters",
    smooth_tolerance = "3 Meters",
)
```

---

## License

This project is released under the [MIT License](LICENSE).
