# -*- coding: utf-8 -*-
"""
Degree_Centerline.pyt
======================
Degree-Aware Branching Centerline Toolbox — ArcGIS Python Toolbox that wraps
the degree-aware branching centerline algorithm in ``centerline_degree.py``.

Unlike ``fast_centerline/Fast_Centerline.pyt`` which extracts only the single
longest path, this toolbox preserves ALL meaningful branches by decomposing the
Voronoi skeleton graph into topological segments and filtering noise branches.

The output is a branching ``MULTILINESTRING`` that faithfully represents the
full skeleton topology of the input polygon.

Runtime dependencies
--------------------
Same as ``fast_centerline``:
    numpy    – pre-installed in every ArcGIS Pro Python environment.
    scipy    – usually pre-installed in ArcGIS Pro.
    networkx – install from conda-forge (see install_dependencies.bat):
                   conda install -c conda-forge networkx

Optional (recommended for fastest rasterisation):
    matplotlib – usually pre-installed in ArcGIS Pro.

How to load this toolbox
------------------------
In **ArcGIS Pro** (Catalog pane) or **ArcCatalog**:
  1. Right-click a folder → Add Toolbox → select ``Degree_Centerline.pyt``.
  2. Expand the toolbox and run **Polygon to Centerline (Degree-Aware)**.

``centerline_degree.py`` must be in the **same directory** as this ``.pyt``
file so that it can be imported at run-time.
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
        self.label = "Degree Centerline"
        self.alias = "DegreeCenterline"
        self.tools = [PolygonToCenterlineDegree]


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


class PolygonToCenterlineDegree(object):
    """ArcGIS tool that wraps the degree-aware polygon_to_centerline_wkt()."""

    def __init__(self):
        self.label = "Polygon to Centerline (Degree-Aware)"
        self.description = (
            "Converts polygon features to branching centerline polylines using "
            "degree-aware Voronoi skeleton decomposition that preserves ALL "
            "meaningful branches of the medial-axis graph.\n\n"
            "Unlike a single-path tool which extracts only the longest path "
            "(losing all branches), this tool decomposes the Voronoi skeleton "
            "into topological segments between junctions and leaf nodes, filters "
            "out short noise branches, and returns a MULTILINESTRING that "
            "faithfully represents the branching centerline.\n\n"
            "Faster than the Steiner-tree approach because it uses O(V+E) graph "
            "traversal instead of NP-hard Steiner-tree approximation.\n\n"
            "Acceleration techniques:\n"
            "  1. Vectorised boundary densification.\n"
            "  2. Batch Voronoi ridge filtering (midpoint PIP + crossing test).\n"
            "  3. Degree-aware skeleton decomposition and noise filtering.\n\n"
            "Requires only numpy (pre-installed), scipy (usually pre-installed),\n"
            "and networkx (conda-forge).  No shapely, geopandas, or pandas required."
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
            "The input polygon feature layer or feature class whose shapes "
            "will be converted to centerline polylines.  Each polygon is "
            "processed independently."
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
            "is a MULTILINESTRING representing the branching centerline of "
            "the corresponding input polygon.  Attribute fields from the "
            "input are copied over, plus an ORIG_FID field."
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
            "tessellation.  Smaller values produce a more detailed "
            "centerline but increase computation time and memory usage.\n\n"
            "Recommended range: 0.5 – 5.0 (in CRS map units).\n"
            "  - Use 0.5 – 1.0 for small or detailed polygons.\n"
            "  - Use 2.0 – 5.0 for large or simple polygons.\n\n"
            "Must be > 0.  Default: 1.0."
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
            "Minimum branch length to keep after initial skeleton "
            "construction.  Branches shorter than this value are pruned "
            "as noise.  Set to 0 to skip this pruning step and rely "
            "solely on the degree-aware ratio-based filtering.\n\n"
            "Recommended range: 0 – 10.0 (in CRS map units).\n"
            "  - 0 (default): no explicit pruning; the algorithm's "
            "built-in ratio filter handles noise.\n"
            "  - 1.0 – 5.0: light cleanup for noisy boundaries.\n"
            "  - 5.0 – 10.0: aggressive pruning; keeps only major branches.\n\n"
            "Must be >= 0.  Default: 0.0."
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
            "When checked (True), returns ALL Voronoi skeleton edges "
            "inside the polygon without degree-aware branch filtering.  "
            "This produces the raw medial-axis graph, which may contain "
            "many short noise branches.\n\n"
            "When unchecked (False, default), the degree-aware algorithm "
            "decomposes the skeleton into topological segments and filters "
            "out short terminal branches, producing a clean branching "
            "centerline.\n\n"
            "Default: False (unchecked)."
        )

        p_max_pts = arcpy.Parameter(
            displayName="Max Densify Points",
            name="max_densify_points",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
        )
        p_max_pts.value = 10000
        p_max_pts.description = (
            "Upper limit on the number of densified boundary points.  "
            "If the polygon perimeter divided by Densification Distance "
            "would exceed this cap, the distance is automatically "
            "increased to stay within the limit.\n\n"
            "Prevents excessive memory usage on very large polygons.\n\n"
            "Recommended range: 5,000 – 100,000.\n"
            "  - 10,000 (default): good balance of detail and speed.\n"
            "  - 5,000: faster, less detail.\n"
            "  - 50,000 – 100,000: more detail; requires more RAM.\n\n"
            "Must be >= 1.  Default: 10,000."
        )

        return [p_in, p_out, p_densify, p_prune, p_full, p_max_pts]

    # ------------------------------------------------------------------
    # Licensing
    # ------------------------------------------------------------------

    def isLicensed(self):
        """Works with any ArcGIS licence level — see updateMessages for dep errors."""
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

        # p_densify (index 2)
        densify_param = parameters[2]
        if densify_param.value is not None and float(densify_param.value) <= 0:
            densify_param.setErrorMessage(
                "Densification Distance must be > 0."
            )

        # p_prune (index 3)
        prune_param = parameters[3]
        if prune_param.value is not None and float(prune_param.value) < 0:
            prune_param.setErrorMessage(
                "Branch Prune Threshold must be >= 0."
            )

        # p_max_pts (index 5)
        max_pts_param = parameters[5]
        if max_pts_param.value is not None and int(max_pts_param.value) < 1:
            max_pts_param.setErrorMessage("Max Densify Points must be >= 1.")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, parameters, messages):
        """Run the degree-aware centerline algorithm."""

        tbx_dir = os.path.dirname(os.path.abspath(__file__))
        if tbx_dir not in sys.path:
            sys.path.insert(0, tbx_dir)

        # ---- Unpack parameters -------------------------------------------
        in_features = parameters[0].valueAsText
        out_features = parameters[1].valueAsText
        method = "voronoi"
        densify_distance = (
            float(parameters[2].value) if parameters[2].value is not None else 1.0
        )
        prune_threshold = (
            float(parameters[3].value) if parameters[3].value is not None else 0.0
        )
        single_line = not bool(parameters[4].value)
        max_densify_points = (
            int(parameters[5].value) if parameters[5].value is not None else 10000
        )

        # ---- Import fast algorithm -----------------------------------------
        try:
            from centerline_degree import polygon_to_centerline_wkt
        except ImportError:
            messages.addErrorMessage(
                "Could not import 'centerline_degree' module.\n"
                "Ensure 'centerline_degree.py' is in the same folder as this toolbox:\n"
                "  {}".format(tbx_dir)
            )
            raise

        if _MISSING_DEPS:
            messages.addErrorMessage(
                _INSTALL_HELP.format(missing=", ".join(_MISSING_DEPS))
            )
            raise RuntimeError(
                "Missing required packages: {}".format(", ".join(_MISSING_DEPS))
            )

        # ---- Read input polygons -------------------------------------------
        t0 = time.time()
        messages.addMessage("Step 1/3  Reading input polygon features ...")

        desc = arcpy.Describe(in_features)
        spatial_ref = desc.spatialReference
        oid_field = desc.OIDFieldName

        _skip_upper = _SYSTEM_FIELDS | {oid_field.upper()}
        attr_fields = [
            f for f in arcpy.ListFields(in_features)
            if f.type not in ("OID", "Geometry")
            and f.name.upper() not in _skip_upper
        ]
        attr_field_names = [f.name for f in attr_fields]
        cursor_fields = ["OID@", "SHAPE@WKT"] + attr_field_names

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

        # ---- Compute centerlines ------------------------------------------
        messages.addMessage(
            "Step 2/3  Computing centerlines "
            "(method={}, densify={}) ...".format(method, densify_distance)
        )

        n_total = len(input_rows)
        arcpy.SetProgressor("step", "Computing centerlines ...", 0, 100, 1)
        arcpy.SetProgressorPosition(5)

        # Progress callback factory — captures messages, t0, n_total from
        # the enclosing scope; poly_i and poly_fid are bound per-polygon
        # via factory arguments to avoid closure-in-a-loop pitfalls.
        def _make_progress_cb(poly_i, poly_fid):
            """Create a progress callback for one polygon."""
            def _cb(msg, pct=-1):
                elapsed = time.time() - t0
                label = "Polygon {}/{} (FID {}): {}".format(
                    poly_i + 1, n_total, poly_fid, msg)
                messages.addMessage(
                    "  [{:.1f}s] {}".format(elapsed, label))
                arcpy.SetProgressorLabel(label)
                if pct >= 0:
                    # Map per-polygon pct (0-100) into the overall 5%-90%
                    # range allocated to Step 2.
                    overall = 5 + int(
                        (poly_i + pct / 100.0) * 85.0 / n_total)
                    arcpy.SetProgressorPosition(min(overall, 90))
            return _cb

        results = []
        n_skipped = 0

        for i, row in enumerate(input_rows):
            orig_fid = row[0]
            wkt = row[1]
            attr_dict = {
                name: val for name, val in zip(attr_field_names, row[2:])
            }

            if not wkt:
                n_skipped += 1
                continue

            progress_cb = _make_progress_cb(i, orig_fid)

            try:
                result_wkt = polygon_to_centerline_wkt(
                    wkt,
                    method=method,
                    densify_distance=densify_distance,
                    prune_threshold=prune_threshold,
                    single_line=single_line,
                    progress_callback=progress_cb,
                    max_densify_points=max_densify_points,
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
                " ({:,} polygon(s) skipped)".format(n_skipped)
                if n_skipped > 0 else "",
                time.time() - t0,
            )
        )

        if n_out == 0:
            messages.addWarningMessage(
                "No centerlines were generated.  "
                "Try a smaller Densification Distance."
            )
            return

        # ---- Write output -------------------------------------------------
        arcpy.SetProgressorPosition(90)
        messages.addMessage("Step 3/3  Writing output feature class ...")

        out_dir = os.path.dirname(out_features)
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
                fl = src_field.length if src_field.type == "String" else None
                try:
                    if fl:
                        arcpy.management.AddField(
                            out_features, fname, ftype, field_length=fl
                        )
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
                out_features, time.time() - t0))
        arcpy.SetProgressorPosition(100)
        arcpy.ResetProgressor()

    def postExecute(self, parameters):
        return
