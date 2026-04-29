# -*- coding: utf-8 -*-
"""
Split_and_Process.pyt
=====================
ArcGIS Python Toolbox — **Tiled Centerline Extraction**.

Wraps ``split_and_process.tile_and_extract_centerline`` to provide the
three-phase tiled extraction workflow for large or complex polygon datasets
directly within ArcGIS Pro / ArcCatalog.

Three-phase strategy
--------------------
Phase A – Connected-component split
    A MULTIPOLYGON feature is automatically split into independent polygon
    parts.  Each part is processed in isolation so that unconnected water
    bodies cannot interfere with each other's skeleton graph.

Phase B – Adaptive quad-tree subdivision
    Polygon parts that exceed the **Max Boundary Vertices** threshold are
    recursively split into four equal quadrants until every tile is below
    the threshold or the **Max Tiling Depth** limit is reached.

Phase C – Overlap-buffer extraction
    Each tile is expanded outward by ``Buffer Factor × Densification
    Distance`` before the source polygon is clipped to it.  This prevents
    the Voronoi algorithm from treating the tile boundary as a real polygon
    edge (which would produce spurious short branches at every tile seam).
    After extraction the centerline is clipped back to the original tile
    boundary, and all tiles are merged into a single output feature.

When to use this tool
---------------------
Use **Polygon to Centerline (Tiled)** instead of the plain
**Polygon to Centerline (Degree-Aware)** when:

  • The input contains very large or country-scale polygon features.
  • Long, narrow waterways are being lost in the plain tool's output.
  • The plain tool runs out of memory or takes an impractically long time.

Runtime dependencies
--------------------
Same as ``Degree_Centerline.pyt``:
    numpy    – pre-installed in every ArcGIS Pro Python environment.
    scipy    – usually pre-installed in ArcGIS Pro.
    networkx – install from conda-forge (see ``install_dependencies.bat``):
                   conda install -c conda-forge networkx

How to load this toolbox
------------------------
In **ArcGIS Pro** (Catalog pane) or **ArcCatalog**:
  1. Right-click a folder → Add Toolbox → select ``Split_and_Process.pyt``.
  2. Expand the toolbox and run **Polygon to Centerline (Tiled)**.

Both ``centerline_degree.py`` and ``split_and_process.py`` must be in the
**same directory** as this ``.pyt`` file.
"""

import os
import sys
import time

import arcpy

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_SYSTEM_FIELDS = frozenset({
    "SHAPE", "SHAPE_LENGTH", "SHAPE_AREA",
    "SHAPE.STLENGTH()", "SHAPE.STAREA()",
})

_ARCPY_FIELD_TYPE_MAP = {
    "SmallInteger":    "SHORT",
    "Integer":         "LONG",
    "BigInteger":      "BIG_INTEGER",
    "Single":          "FLOAT",
    "Double":          "DOUBLE",
    "String":          "TEXT",
    "Date":            "DATE",
    "DateOnly":        "DATE_ONLY",
    "TimeOnly":        "TIME_ONLY",
    "TimestampOffset": "TIMESTAMP_OFFSET",
    "GUID":            "GUID",
    "GlobalID":        "GUID",
    "Raster":          "TEXT",
    "Blob":            "TEXT",
}

# ---------------------------------------------------------------------------
# Pre-flight dependency check
# ---------------------------------------------------------------------------


def _check_dependencies():
    """Return a list of package names that are NOT importable."""
    _required = [
        ("scipy",    "scipy"),
        ("numpy",    "numpy"),
        ("networkx", "networkx"),
    ]
    missing = []
    for pkg_name, import_name in _required:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg_name)
    return missing


_MISSING_DEPS = _check_dependencies()

_INSTALL_HELP = (
    "\n"
    "REQUIRED PACKAGES ARE NOT INSTALLED\n"
    "====================================\n"
    "Missing: {missing}\n"
    "\n"
    "Quick fix — run 'install_dependencies.bat' found in the same\n"
    "folder as this toolbox (degree_centerline/).  See README.md for\n"
    "full instructions.\n"
    "\n"
    "Manual installation (ArcGIS Pro Python Command Prompt):\n"
    "\n"
    "  Step 1 — Clone the default environment (only once):\n"
    "    conda create --name arcgispro-py3-degree --clone arcgispro-py3\n"
    "\n"
    "  Step 2 — Install networkx into the clone:\n"
    "    activate arcgispro-py3-degree\n"
    "    conda install -c conda-forge -y networkx\n"
    "\n"
    "  Step 3 — Set the clone as the active environment in ArcGIS Pro:\n"
    "    Project > Python > Python Environments > arcgispro-py3-degree\n"
    "    Restart ArcGIS Pro.\n"
    "\n"
    "  Note: numpy and scipy are usually already present in the default\n"
    "  'arcgispro-py3' environment.  matplotlib is also usually present\n"
    "  and further accelerates rasterisation (no extra install needed).\n"
)

# ---------------------------------------------------------------------------
# Toolbox
# ---------------------------------------------------------------------------


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the .pyt file)."""
        self.label = "Split and Process"
        self.alias = "SplitAndProcess"
        self.tools = [PolygonToCenterlineTiled]


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


class PolygonToCenterlineTiled(object):
    """ArcGIS tool: tiled centerline extraction using split_and_process.py."""

    def __init__(self):
        self.label = "Polygon to Centerline (Tiled)"
        self.description = (
            "Converts polygon features to branching centerline polylines using "
            "a three-phase tiled extraction strategy designed for large or "
            "complex datasets (e.g. country-scale waterway networks).\n\n"
            "Phase A  — Connected-component split: each polygon part in a "
            "MULTIPOLYGON is processed independently.\n\n"
            "Phase B  — Adaptive quad-tree subdivision: polygon parts that "
            "exceed the Max Boundary Vertices threshold are recursively split "
            "into four quadrant tiles until all tiles are below the threshold "
            "or the Max Tiling Depth is reached.\n\n"
            "Phase C  — Overlap-buffer extraction: each tile is expanded "
            "outward by (Buffer Factor × Densification Distance) before the "
            "source polygon is clipped to it. This prevents the Voronoi "
            "algorithm from producing spurious short branches at tile seams. "
            "After extraction the centerline is clipped back to the original "
            "tile boundary; all tiles are then merged into one output feature."
        )
        self.canRunInBackground = True

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def getParameterInfo(self):
        """Define the tool parameters."""

        p_in = arcpy.Parameter(
            displayName="Input Polygon Features",
            name="in_features",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input",
        )
        p_in.filter.list = ["Polygon"]
        p_in.description = (
            "The input polygon feature layer or feature class.  Each polygon "
            "is processed independently through the three-phase tiled pipeline."
        )

        p_out = arcpy.Parameter(
            displayName="Output Centerline Features",
            name="out_features",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Output",
        )
        p_out.description = (
            "Path to the output polyline feature class.  Each output feature "
            "is the merged tiled centerline of the corresponding input polygon. "
            "Attribute fields from the input are copied over, plus an ORIG_FID "
            "field."
        )

        p_densify = arcpy.Parameter(
            displayName="Densification Distance (CRS units)",
            name="densify_distance",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input",
        )
        p_densify.value = 1.0
        p_densify.description = (
            "Maximum spacing between boundary vertices before Voronoi "
            "tessellation.  Smaller values produce a more detailed centerline "
            "but increase computation time and memory usage.\n\n"
            "Recommended range: 0.5 – 5.0 CRS units.  Default: 1.0."
        )

        p_prune = arcpy.Parameter(
            displayName="Branch Prune Threshold (CRS units; 0 = no pruning)",
            name="prune_threshold",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input",
        )
        p_prune.value = 0.0
        p_prune.description = (
            "Minimum branch length to keep after initial skeleton construction. "
            "Set to 0 (default) to rely solely on the degree-aware ratio-based "
            "filtering built into the algorithm."
        )

        p_full = arcpy.Parameter(
            displayName="Return Full Raw Skeleton (skip degree-aware filtering)",
            name="full_skeleton",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
        )
        p_full.value = False
        p_full.description = (
            "When checked, returns ALL Voronoi skeleton edges without "
            "degree-aware branch filtering (raw medial axis).\n\n"
            "Default: unchecked (degree-aware filtering applied)."
        )

        p_max_pts = arcpy.Parameter(
            displayName="Max Densify Points (per tile)",
            name="max_densify_points",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
        )
        p_max_pts.value = 10000
        p_max_pts.description = (
            "Per-tile upper limit on densified boundary points.  If a tile's "
            "perimeter would exceed this cap, the densification distance is "
            "automatically increased to stay within the limit.\n\n"
            "Default: 10,000."
        )

        # ── Tiling parameters ────────────────────────────────────────────────

        p_max_v = arcpy.Parameter(
            displayName="Max Boundary Vertices per Tile (Phase B threshold)",
            name="max_vertices",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
        )
        p_max_v.value = 8000
        p_max_v.description = (
            "Phase B — adaptive subdivision threshold.  Any tile whose total "
            "boundary vertex count (exterior ring + all hole rings) exceeds "
            "this value is split into four quadrant sub-tiles recursively.\n\n"
            "Lower values produce more, smaller tiles (slower overall but each "
            "tile is processed faster and with better detail).  Higher values "
            "produce fewer, larger tiles.\n\n"
            "Recommended range: 2,000 – 20,000.  Default: 8,000."
        )

        p_max_area = arcpy.Parameter(
            displayName="Max Tile Bounding-Box Area (CRS area units; 0 = disabled)",
            name="max_bbox_area",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input",
        )
        p_max_area.value = 0.0
        p_max_area.description = (
            "Optional Phase B area trigger.  Set to the maximum acceptable "
            "bounding-box area (in CRS area units, e.g. square metres or "
            "square degrees) for a single tile.  Tiles larger than this value "
            "are subdivided even if their vertex count is below the threshold.\n\n"
            "Set to 0 (default) to disable the area check and rely only on "
            "vertex count."
        )

        p_buf = arcpy.Parameter(
            displayName="Overlap Buffer Factor (Phase C; multiple of Densification Distance)",
            name="buffer_factor",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input",
        )
        p_buf.value = 5.0
        p_buf.description = (
            "Phase C — overlap buffer expressed as a multiple of the "
            "Densification Distance.  Each tile boundary is expanded outward "
            "by this many × Densification Distance units before the source "
            "polygon is clipped to it.  A larger buffer reduces seam artefacts "
            "but slightly increases processing time per tile.\n\n"
            "Recommended range: 3 – 8.  Default: 5."
        )

        p_depth = arcpy.Parameter(
            displayName="Max Tiling Depth (Phase B recursion limit)",
            name="max_depth",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
        )
        p_depth.value = 5
        p_depth.description = (
            "Maximum quad-tree recursion depth for Phase B subdivision.  At "
            "depth d a single polygon can produce at most 4^d tiles "
            "(depth 5 → at most 1,024 tiles per input polygon).\n\n"
            "Increase this value if very complex polygons still exceed the "
            "vertex threshold at the deepest level.  Default: 5."
        )

        return [
            p_in, p_out,
            p_densify, p_prune, p_full, p_max_pts,
            p_max_v, p_max_area, p_buf, p_depth,
        ]

    # ------------------------------------------------------------------
    # Licensing
    # ------------------------------------------------------------------

    def isLicensed(self):
        return True

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        if _MISSING_DEPS:
            parameters[0].setErrorMessage(
                _INSTALL_HELP.format(missing=", ".join(_MISSING_DEPS))
            )
            return

        # densify_distance (index 2)
        if parameters[2].value is not None and float(parameters[2].value) <= 0:
            parameters[2].setErrorMessage("Densification Distance must be > 0.")

        # prune_threshold (index 3)
        if parameters[3].value is not None and float(parameters[3].value) < 0:
            parameters[3].setErrorMessage("Branch Prune Threshold must be >= 0.")

        # max_densify_points (index 5)
        if parameters[5].value is not None and int(parameters[5].value) < 1:
            parameters[5].setErrorMessage("Max Densify Points must be >= 1.")

        # max_vertices (index 6)
        if parameters[6].value is not None and int(parameters[6].value) < 1:
            parameters[6].setErrorMessage("Max Boundary Vertices must be >= 1.")

        # buffer_factor (index 8)
        if parameters[8].value is not None and float(parameters[8].value) < 0:
            parameters[8].setErrorMessage("Overlap Buffer Factor must be >= 0.")

        # max_depth (index 9)
        if parameters[9].value is not None and int(parameters[9].value) < 1:
            parameters[9].setErrorMessage("Max Tiling Depth must be >= 1.")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, parameters, messages):
        """Run the tiled centerline algorithm."""

        tbx_dir = os.path.dirname(os.path.abspath(__file__))
        if tbx_dir not in sys.path:
            sys.path.insert(0, tbx_dir)

        # ── Unpack parameters ────────────────────────────────────────────────
        in_features        = parameters[0].valueAsText
        out_features       = parameters[1].valueAsText
        densify_distance   = float(parameters[2].value) if parameters[2].value is not None else 1.0
        prune_threshold    = float(parameters[3].value) if parameters[3].value is not None else 0.0
        single_line        = not bool(parameters[4].value)
        max_densify_points = int(parameters[5].value)   if parameters[5].value is not None else 10000
        max_vertices       = int(parameters[6].value)   if parameters[6].value is not None else 8000
        _raw_area          = float(parameters[7].value) if parameters[7].value is not None else 0.0
        max_bbox_area      = _raw_area if _raw_area > 0 else None
        buffer_factor      = float(parameters[8].value) if parameters[8].value is not None else 5.0
        max_depth          = int(parameters[9].value)   if parameters[9].value is not None else 5

        method = "voronoi"

        # ── Import tiled extraction module ───────────────────────────────────
        try:
            from split_and_process import tile_and_extract_centerline
        except ImportError:
            messages.addErrorMessage(
                "Could not import 'split_and_process'.\n"
                "Ensure 'split_and_process.py' and 'centerline_degree.py' are "
                "in the same folder as this toolbox:\n  {}".format(tbx_dir)
            )
            raise

        if _MISSING_DEPS:
            messages.addErrorMessage(
                _INSTALL_HELP.format(missing=", ".join(_MISSING_DEPS))
            )
            raise RuntimeError(
                "Missing required packages: {}".format(", ".join(_MISSING_DEPS))
            )

        # ── Read input polygons ──────────────────────────────────────────────
        t0 = time.time()
        messages.addMessage("Step 1/3  Reading input polygon features ...")

        desc        = arcpy.Describe(in_features)
        spatial_ref = desc.spatialReference
        oid_field   = desc.OIDFieldName

        _skip_upper = _SYSTEM_FIELDS | {oid_field.upper()}
        attr_fields = [
            f for f in arcpy.ListFields(in_features)
            if f.type not in ("OID", "Geometry")
            and f.name.upper() not in _skip_upper
        ]
        attr_field_names = [f.name for f in attr_fields]
        cursor_fields    = ["OID@", "SHAPE@WKT"] + attr_field_names

        input_rows = []
        with arcpy.da.SearchCursor(in_features, cursor_fields) as cursor:
            for row in cursor:
                input_rows.append(row)

        messages.addMessage(
            "         {:,} polygon(s) read. [{:.1f}s]".format(
                len(input_rows), time.time() - t0)
        )

        if not input_rows:
            messages.addWarningMessage("No polygon features found.")
            return

        # ── Compute centerlines ──────────────────────────────────────────────
        messages.addMessage(
            "Step 2/3  Computing tiled centerlines "
            "(method={}, densify={}, max_vertices={}, buffer_factor={}) ...".format(
                method, densify_distance, max_vertices, buffer_factor)
        )

        n_total = len(input_rows)
        arcpy.SetProgressor("step", "Computing tiled centerlines ...", 0, 100, 1)
        arcpy.SetProgressorPosition(5)

        def _make_progress_cb(poly_i, poly_fid):
            def _cb(msg, pct=-1):
                elapsed = time.time() - t0
                label = "Polygon {}/{} (FID {}): {}".format(
                    poly_i + 1, n_total, poly_fid, msg)
                messages.addMessage("  [{:.1f}s] {}".format(elapsed, label))
                arcpy.SetProgressorLabel(label)
                if pct >= 0:
                    overall = 5 + int(
                        (poly_i + pct / 100.0) * 85.0 / n_total)
                    arcpy.SetProgressorPosition(min(overall, 90))
            return _cb

        results  = []
        n_skipped = 0

        for i, row in enumerate(input_rows):
            orig_fid = row[0]
            wkt      = row[1]
            attr_dict = {
                name: val for name, val in zip(attr_field_names, row[2:])
            }

            if not wkt:
                n_skipped += 1
                continue

            progress_cb = _make_progress_cb(i, orig_fid)

            try:
                result_wkt = tile_and_extract_centerline(
                    wkt,
                    method=method,
                    densify_distance=densify_distance,
                    prune_threshold=prune_threshold,
                    single_line=single_line,
                    max_vertices=max_vertices,
                    max_bbox_area=max_bbox_area,
                    buffer_factor=buffer_factor,
                    max_depth=max_depth,
                    max_densify_points=max_densify_points,
                    progress_callback=progress_cb,
                )
            except Exception as exc:
                messages.addWarningMessage(
                    "Skipping FID {}: {}".format(orig_fid, exc)
                )
                n_skipped += 1
                result_wkt = None

            if result_wkt:
                results.append((result_wkt, orig_fid, attr_dict))
            else:
                n_skipped += 1

        n_out = len(results)
        messages.addMessage(
            "         {:,} centerline(s) generated{}. [{:.1f}s]".format(
                n_out,
                " ({:,} polygon(s) skipped)".format(n_skipped) if n_skipped else "",
                time.time() - t0,
            )
        )

        if n_out == 0:
            messages.addWarningMessage(
                "No centerlines were generated.  "
                "Try a smaller Densification Distance or Max Boundary Vertices."
            )
            return

        # ── Write output ─────────────────────────────────────────────────────
        arcpy.SetProgressorPosition(90)
        messages.addMessage("Step 3/3  Writing output feature class ...")

        out_dir  = os.path.dirname(out_features)
        out_name = os.path.basename(out_features)
        if not out_dir:
            out_dir = arcpy.env.scratchGDB

        arcpy.management.CreateFeatureclass(
            out_dir, out_name, "POLYLINE",
            spatial_reference=(
                spatial_ref
                if (spatial_ref and spatial_ref.name != "Unknown")
                else None
            ),
        )
        arcpy.env.overwriteOutput = True

        arcpy.management.AddField(
            out_features, "ORIG_FID", "LONG",
            field_alias="Original Feature ID",
        )

        existing_out_field_names = {
            f.name.upper() for f in arcpy.ListFields(out_features)
        }
        fields_to_add = []
        for src_field in attr_fields:
            fname = src_field.name
            if fname.upper() not in existing_out_field_names:
                ftype = _ARCPY_FIELD_TYPE_MAP.get(src_field.type, "TEXT")
                fl    = src_field.length if src_field.type == "String" else None
                try:
                    if fl:
                        arcpy.management.AddField(
                            out_features, fname, ftype, field_length=fl)
                    else:
                        arcpy.management.AddField(out_features, fname, ftype)
                    fields_to_add.append(fname)
                    existing_out_field_names.add(fname.upper())
                except Exception:
                    pass
            else:
                fields_to_add.append(fname)

        _field_type_by_name = {f.name: f.type for f in attr_fields}
        insert_fields = ["SHAPE@WKT", "ORIG_FID"] + fields_to_add

        with arcpy.da.InsertCursor(out_features, insert_fields) as cursor:
            for result_wkt, orig_fid, attr_dict in results:
                attr_vals = [
                    str(attr_dict.get(f))
                    if _field_type_by_name.get(f) == "String"
                    else attr_dict.get(f)
                    for f in fields_to_add
                ]
                cursor.insertRow([result_wkt, orig_fid] + attr_vals)

        messages.addMessage(
            "Done.  Output saved to: {} [{:.1f}s total]".format(
                out_features, time.time() - t0)
        )
        arcpy.SetProgressorPosition(100)
        arcpy.ResetProgressor()

    def postExecute(self, parameters):
        return
