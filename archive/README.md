# Archive

This folder contains earlier experimental versions of the centerline extraction
algorithm, kept for reference and development history.

**These tools are superseded by `degree_centerline/` and are not recommended
for production use.**

| Folder | Description |
|---|---|
| `arcpy_toolbox/` | First version — uses ArcGIS native Thiessen polygon tools; requires Advanced license |
| `pure_centerline/` | Pure Python Voronoi skeleton; correct but slow (unvectorised) |
| `fast_centerline/` | Vectorised Voronoi skeleton; extracts longest path only (no branching) |
| `auto_centerline/` | Automatic parameter tuning wrapper around `fast_centerline` |
| `steiner_centerline/` | Steiner-tree branch preservation (O(T²·V·log V), slower than degree-aware) |
| `gdal_centerline/` | Open-source (no ArcPy) version using shapely / geopandas |

For the current, recommended toolboxes see [`degree_centerline/`](../degree_centerline/)
and [`proportional_buffer/`](../proportional_buffer/).
