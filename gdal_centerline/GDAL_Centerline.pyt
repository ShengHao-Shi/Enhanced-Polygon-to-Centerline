# -*- coding: utf-8 -*-
"""
GDAL_Centerline.pyt
===================
ArcGIS Python Toolbox that wraps the open-source GDAL/Shapely centerline
algorithm implemented in ``centerline.py``.

Key differences from ``arcpy_toolbox/Polygon_to_Centerline.pyt``
-----------------------------------------------------------------
* Requires **only a Basic (formerly ArcView) ArcGIS licence** — no
  Standard/Advanced licence is needed because all geometry computations are
  performed by open-source Python libraries (shapely, scipy, networkx).
* Exposes both algorithm methods: ``voronoi`` (default, vector-based) and
  ``skeleton`` (raster-based morphological thinning).
* Supports polygon shapes with interior holes (e.g. letter 'O' or 'A') and
  correctly returns a closed loop for ring-shaped polygons.

Runtime dependencies
--------------------
The following packages must be available in the ArcGIS Pro Python environment
(all are available from conda-forge)::

    conda install -c conda-forge numpy scipy shapely geopandas networkx

For ``method=skeleton`` also install::

    conda install -c conda-forge scikit-image

How to load this toolbox
------------------------
In **ArcGIS Pro** (Catalog pane) or **ArcCatalog**:
  1. Right-click a folder → Add Toolbox → select ``GDAL_Centerline.pyt``.
  2. Expand the toolbox and run **Polygon to Centerline (GDAL)**.

``centerline.py`` must be in the **same directory** as this ``.pyt`` file so
that it can be imported at run-time.
"""

import os
import sys

import arcpy


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# ArcGIS system fields that should never be copied to the output feature class.
# These are managed internally by the geodatabase and cannot be inserted into
# by an InsertCursor.
_SYSTEM_FIELDS = frozenset({
    "SHAPE", "SHAPE_LENGTH", "SHAPE_AREA",
    "SHAPE.STLENGTH()", "SHAPE.STAREA()",
})

# Mapping from arcpy field type strings to the type token accepted by
# arcpy.management.AddField().  Types not listed here fall back to "TEXT".
_ARCPY_FIELD_TYPE_MAP = {
    "SmallInteger": "SHORT",
    "Integer":      "LONG",
    "BigInteger":   "BIG_INTEGER",
    "Single":       "FLOAT",
    "Double":       "DOUBLE",
    "String":       "TEXT",
    "Date":         "DATE",
    "DateOnly":     "DATE_ONLY",
    "TimeOnly":     "TIME_ONLY",
    "TimestampOffset": "TIMESTAMP_OFFSET",
    "GUID":         "GUID",
    "GlobalID":     "GUID",  # treat as GUID
    "Raster":       "TEXT",  # unsupported; fall back to TEXT
    "Blob":         "TEXT",  # unsupported; fall back to TEXT
}

# ---------------------------------------------------------------------------
# Pre-flight dependency check (runs once when the toolbox is loaded)
# ---------------------------------------------------------------------------
# Probe for required packages at import time so that isLicensed() can surface
# a friendly error in the tool dialog before the user clicks "Run".

def _check_dependencies():
    """
    Return a list of (package_name, import_name) tuples for packages that are
    NOT currently importable.  An empty list means all dependencies are present.
    """
    _required = [
        ("shapely",    "shapely"),
        ("geopandas",  "geopandas"),
        ("scipy",      "scipy"),
        ("numpy",      "numpy"),
        ("networkx",   "networkx"),
        ("pandas",     "pandas"),
    ]
    missing = []
    for pkg_name, import_name in _required:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg_name)
    return missing


_MISSING_DEPS = _check_dependencies()

# Human-readable install instructions shown in both isLicensed() and execute().
_INSTALL_HELP = (
    "\n"
    "REQUIRED PACKAGES ARE NOT INSTALLED\n"
    "====================================\n"
    "Missing: {missing}\n"
    "\n"
    "Quick fix — run 'install_dependencies.bat' found in the same\n"
    "folder as this toolbox (gdal_centerline/).  See README.md for\n"
    "full instructions.\n"
    "\n"
    "Manual installation (ArcGIS Pro Python Command Prompt):\n"
    "\n"
    "  Step 1 — Clone the default environment (only once):\n"
    "    conda create --name arcgispro-py3-gdal --clone arcgispro-py3\n"
    "\n"
    "  Step 2 — Install the packages into the clone:\n"
    "    activate arcgispro-py3-gdal\n"
    "    conda install -c conda-forge -y shapely geopandas scipy networkx\n"
    "\n"
    "  Step 3 — Set the clone as the active environment in ArcGIS Pro:\n"
    "    Project > Python > Python Environments > arcgispro-py3-gdal\n"
    "    Restart ArcGIS Pro.\n"
    "\n"
    "  Note: The default 'arcgispro-py3' environment is read-only when\n"
    "  ArcGIS Pro is installed in 'C:\\Program Files\\'.  Cloning it\n"
    "  creates a writable copy that you can freely extend.\n"
)

# ---------------------------------------------------------------------------
# Toolbox
# ---------------------------------------------------------------------------


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the .pyt file)."""
        self.label = "GDAL Centerline"
        self.alias = "GDALCenterline"
        self.tools = [PolygonToCenterlineGDAL]


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


class PolygonToCenterlineGDAL(object):
    """ArcGIS tool that wraps the open-source polygon_to_centerline() function."""

    def __init__(self):
        self.label = "Polygon to Centerline (GDAL)"
        self.description = (
            "Converts polygon features to centerline polylines using the "
            "open-source GDAL / Shapely implementation.\n\n"
            "Unlike the ArcPy-based tool, this tool works with any ArcGIS "
            "licence level (Basic / Standard / Advanced) because it uses only "
            "open-source Python libraries for computation.\n\n"
            "Two algorithms are available:\n"
            "  voronoi  — Vector-based Voronoi / Thiessen skeleton (default).\n"
            "             Fast, precise, works well for elongated polygons.\n"
            "  skeleton — Raster-based morphological skeletonisation.\n"
            "             Better for very jagged or organic shapes.\n\n"
            "Both methods correctly handle polygons with interior holes (e.g. "
            "ring-shaped or letter 'O' / 'A' shapes) and never produce a "
            "centerline that crosses through a hollow region."
        )
        self.canRunInBackground = True

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def getParameterInfo(self):
        """Define the tool parameters shown in the ArcGIS dialog."""

        # 0 — Input polygon features
        p_in = arcpy.Parameter(
            displayName="Input Polygon Features",
            name="in_features",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input",
        )
        p_in.filter.list = ["Polygon"]

        # 1 — Output centerline feature class
        p_out = arcpy.Parameter(
            displayName="Output Centerline Features",
            name="out_features",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Output",
        )

        # 2 — Algorithm method
        p_method = arcpy.Parameter(
            displayName="Method",
            name="method",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )
        p_method.filter.type = "ValueList"
        p_method.filter.list = ["voronoi", "skeleton"]
        p_method.value = "voronoi"

        # 3 — Densification distance (both methods)
        p_densify = arcpy.Parameter(
            displayName="Densification Distance (CRS units)",
            name="densify_distance",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input",
        )
        p_densify.value = 1.0

        # 4 — Prune threshold (Voronoi only)
        p_prune = arcpy.Parameter(
            displayName="Branch Prune Threshold (CRS units; 0 = no pruning)",
            name="prune_threshold",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input",
        )
        p_prune.value = 0.0
        p_prune.category = "Voronoi Options"

        # 5 — Gaussian smooth sigma (Skeleton only)
        p_smooth = arcpy.Parameter(
            displayName="Gaussian Smooth Sigma (CRS units; 0 = no smoothing)",
            name="smooth_sigma",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input",
        )
        p_smooth.value = 0.0
        p_smooth.category = "Skeleton Options"

        # 6 — Raster resolution (Skeleton only)
        p_res = arcpy.Parameter(
            displayName="Raster Resolution Override (CRS units; blank = auto)",
            name="raster_resolution",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input",
        )
        p_res.category = "Skeleton Options"

        # 7 — Return full skeleton instead of single trunk line
        p_full = arcpy.Parameter(
            displayName="Return Full Skeleton (may contain branches / forks)",
            name="full_skeleton",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
        )
        p_full.value = False

        return [p_in, p_out, p_method, p_densify, p_prune, p_smooth, p_res, p_full]

    # ------------------------------------------------------------------
    # Licensing
    # ------------------------------------------------------------------

    def isLicensed(self):
        """
        The tool works with any ArcGIS licence level (Basic / Standard / Advanced).
        All geometry operations are performed by open-source libraries; no
        Standard/Advanced geoprocessing tools are called.

        NOTE: Missing Python packages are reported via updateMessages() (a
        parameter error on the input layer), not here.  Returning False from
        isLicensed() would cause ArcGIS Pro to display the misleading
        "Tool not licensed" message and lock the dialog, hiding the actual
        install instructions from the user.
        """
        return True

    # ------------------------------------------------------------------
    # Validation hooks
    # ------------------------------------------------------------------

    def updateParameters(self, parameters):
        """
        Enable / disable method-specific parameters when the Method drop-down
        changes so that the dialog guides the user appropriately.
        """
        method = parameters[2].valueAsText or "voronoi"
        # Voronoi-only parameter
        parameters[4].enabled = method == "voronoi"
        # Skeleton-only parameters
        parameters[5].enabled = method == "skeleton"
        parameters[6].enabled = method == "skeleton"

    def updateMessages(self, parameters):
        """Validate parameter values and surface any warnings."""

        # If required packages are missing, set a visible error on the input
        # feature parameter.  This surfaces the install instructions in the
        # tool dialog and prevents the user from clicking Run, without
        # triggering the misleading "Tool not licensed" lock that isLicensed()
        # returning False would cause.
        if _MISSING_DEPS:
            parameters[0].setErrorMessage(
                _INSTALL_HELP.format(missing=", ".join(_MISSING_DEPS))
            )
            return  # skip remaining validation while deps are absent

        for idx in (3, 4, 5):
            param = parameters[idx]
            if param.value is not None and float(param.value) < 0:
                param.setErrorMessage("Value must be ≥ 0.")

        res_param = parameters[6]
        if res_param.value is not None and float(res_param.value) <= 0:
            res_param.setErrorMessage("Raster Resolution must be > 0.")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, parameters, messages):
        """Run the GDAL centerline algorithm."""

        # Add the toolbox directory to sys.path so we can import centerline.py
        tbx_dir = os.path.dirname(os.path.abspath(__file__))
        if tbx_dir not in sys.path:
            sys.path.insert(0, tbx_dir)

        # ---- Unpack parameters -------------------------------------------
        in_features = parameters[0].valueAsText
        out_features = parameters[1].valueAsText
        method = parameters[2].valueAsText or "voronoi"
        densify_distance = (
            float(parameters[3].value) if parameters[3].value is not None else 1.0
        )
        prune_threshold = (
            float(parameters[4].value) if parameters[4].value is not None else 0.0
        )
        smooth_sigma = (
            float(parameters[5].value) if parameters[5].value is not None else 0.0
        )
        raster_resolution = (
            float(parameters[6].value) if parameters[6].value is not None else None
        )
        single_line = not bool(parameters[7].value)

        # ---- Import open-source dependencies --------------------------------
        try:
            import geopandas as gpd
            import pandas as pd
            from shapely.wkt import loads as wkt_loads
        except ImportError as exc:
            messages.addErrorMessage(
                _INSTALL_HELP.format(missing=str(exc).replace("No module named ", ""))
            )
            raise

        try:
            from centerline import polygon_to_centerline
        except ImportError:
            messages.addErrorMessage(
                "Could not import 'centerline' module.\n"
                "Ensure 'centerline.py' is in the same folder as this toolbox:\n"
                "  {}".format(tbx_dir)
            )
            raise

        # ---- Read input polygons via arcpy cursor ---------------------------
        messages.addMessage("Step 1/3  Reading input polygon features …")

        desc = arcpy.Describe(in_features)
        spatial_ref = desc.spatialReference
        oid_field = desc.OIDFieldName

        # Collect attribute fields — skip system/geometry fields using the
        # module-level _SYSTEM_FIELDS constant to avoid code duplication.
        _skip_upper = _SYSTEM_FIELDS | {oid_field.upper()}
        attr_fields = [
            f for f in arcpy.ListFields(in_features)
            if f.type not in ("OID", "Geometry") and f.name.upper() not in _skip_upper
        ]
        attr_field_names = [f.name for f in attr_fields]

        cursor_fields = ["OID@", "SHAPE@WKT"] + attr_field_names
        geometries = []
        orig_fids = []
        attr_rows = []

        with arcpy.da.SearchCursor(in_features, cursor_fields) as cursor:
            for row in cursor:
                orig_fid = row[0]
                wkt = row[1]
                geom = wkt_loads(wkt) if wkt else None
                geometries.append(geom)
                orig_fids.append(orig_fid)
                attr_rows.append(row[2:])

        messages.addMessage("         {:,} polygon(s) read.".format(len(geometries)))

        if not geometries:
            messages.addWarningMessage("No polygon features found in the input layer.")
            return

        # Build GeoDataFrame
        attr_df = pd.DataFrame(attr_rows, columns=attr_field_names)
        attr_df.insert(0, "ORIG_FID", orig_fids)
        gdf = gpd.GeoDataFrame(attr_df, geometry=geometries)

        # Assign CRS from the arcpy spatial reference when possible
        epsg = spatial_ref.factoryCode if spatial_ref else 0
        if epsg > 0:
            try:
                gdf = gdf.set_crs(epsg=epsg)
            except Exception:
                pass  # unrecognised EPSG — leave CRS unset

        # ---- Compute centerlines --------------------------------------------
        messages.addMessage(
            "Step 2/3  Computing centerlines "
            "(method={}, densify={}) …".format(method, densify_distance)
        )
        try:
            result_gdf = polygon_to_centerline(
                gdf,
                method=method,
                densify_distance=densify_distance,
                prune_threshold=prune_threshold,
                smooth_sigma=smooth_sigma,
                raster_resolution=raster_resolution,
                single_line=single_line,
            )
        except Exception as exc:
            messages.addErrorMessage(
                "Centerline computation failed: {}".format(exc)
            )
            raise

        n_out = len(result_gdf)
        n_skipped = len(gdf) - n_out
        messages.addMessage(
            "         {:,} centerline(s) generated"
            "{}.".format(
                n_out,
                " ({:,} polygon(s) skipped — too small or degenerate)".format(n_skipped)
                if n_skipped > 0 else "",
            )
        )

        if n_out == 0:
            messages.addWarningMessage(
                "No centerlines were generated.  "
                "Try a smaller Densification Distance."
            )
            return

        # ---- Write output feature class -------------------------------------
        messages.addMessage("Step 3/3  Writing output feature class …")

        # Determine workspace and feature-class name from the full output path
        out_dir = os.path.dirname(out_features)
        out_name = os.path.basename(out_features)
        if not out_dir:
            out_dir = arcpy.env.scratchGDB

        arcpy.management.CreateFeatureclass(
            out_dir,
            out_name,
            "POLYLINE",
            spatial_reference=spatial_ref if (spatial_ref and spatial_ref.name != "Unknown") else None,
        )
        arcpy.env.overwriteOutput = True

        # Add ORIG_FID field to link back to input polygons
        arcpy.management.AddField(out_features, "ORIG_FID", "LONG",
                                  field_alias="Original Feature ID")

        # Add the same attribute fields that existed in the input, preserving
        # their original field types using the module-level _ARCPY_FIELD_TYPE_MAP.
        existing_out_field_names = {
            f.name.upper() for f in arcpy.ListFields(out_features)
        }
        fields_to_add = []
        for src_field in attr_fields:
            fname = src_field.name
            if fname.upper() not in existing_out_field_names:
                ftype = _ARCPY_FIELD_TYPE_MAP.get(src_field.type, "TEXT")
                field_length = src_field.length if src_field.type == "String" else None
                try:
                    if field_length:
                        arcpy.management.AddField(out_features, fname, ftype,
                                                 field_length=field_length)
                    else:
                        arcpy.management.AddField(out_features, fname, ftype)
                    fields_to_add.append(fname)
                    existing_out_field_names.add(fname.upper())
                except Exception:
                    pass  # skip fields that cannot be added (e.g. reserved names)
            else:
                fields_to_add.append(fname)

        # Build a name→type lookup so we can cast values appropriately
        # when inserting into the cursor.
        _field_type_by_name = {f.name: f.type for f in attr_fields}

        # Insert centerline features
        insert_fields = ["SHAPE@WKT", "ORIG_FID"] + fields_to_add
        with arcpy.da.InsertCursor(out_features, insert_fields) as cursor:
            for _, row in result_gdf.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                orig_fid = row.get("ORIG_FID", None)
                attr_vals = []
                for f in fields_to_add:
                    val = row[f] if f in row.index else None
                    if val is None:
                        attr_vals.append(None)
                    elif _field_type_by_name.get(f) == "String":
                        # Only coerce to str for text fields; numeric/date
                        # values are passed as-is so the cursor receives the
                        # correct Python type (int, float, datetime, …).
                        attr_vals.append(str(val))
                    else:
                        attr_vals.append(val)
                cursor.insertRow([geom.wkt, orig_fid] + attr_vals)

        messages.addMessage(
            "Done.  Output saved to: {}".format(out_features)
        )

    def postExecute(self, parameters):
        return


# ---------------------------------------------------------------------------
# Stand-alone helper — also callable from a plain Python script
# ---------------------------------------------------------------------------


def run_centerline(
    in_features,
    out_features,
    method="voronoi",
    densify_distance=1.0,
    prune_threshold=0.0,
    smooth_sigma=0.0,
    raster_resolution=None,
    single_line=True,
):
    """
    Run the GDAL centerline algorithm from a plain Python script (no ArcGIS
    dialog required).

    Parameters
    ----------
    in_features : str
        Path to an ArcGIS feature class or any vector file readable by
        geopandas (GeoJSON, Shapefile, GPKG, …).
    out_features : str
        Path for the output polyline feature class (ArcGIS geodatabase or
        vector file).
    method : {"voronoi", "skeleton"}
        Algorithm.  "voronoi" (default) is vector-based; "skeleton" is
        raster-based.
    densify_distance : float
        Maximum vertex spacing for boundary densification (CRS units).
    prune_threshold : float
        Remove skeleton branches shorter than this value.  0 = no pruning.
        Voronoi method only.
    smooth_sigma : float
        Gaussian blur sigma applied before skeletonisation.  0 = no blur.
        Skeleton method only.
    raster_resolution : float or None
        Raster cell size override for the skeleton method.  None = auto.
    single_line : bool
        When True (default) return a single non-branching centerline.
        When False return the full skeleton (may contain forks).
    """
    tbx_dir = os.path.dirname(os.path.abspath(__file__))
    if tbx_dir not in sys.path:
        sys.path.insert(0, tbx_dir)

    import geopandas as gpd
    import pandas as pd
    from shapely.wkt import loads as wkt_loads
    from centerline import polygon_to_centerline

    # Read input
    desc = arcpy.Describe(in_features)
    spatial_ref = desc.spatialReference
    oid_field = desc.OIDFieldName

    _skip_upper = _SYSTEM_FIELDS | {oid_field.upper()}
    attr_field_names = [
        f.name
        for f in arcpy.ListFields(in_features)
        if f.type not in ("OID", "Geometry") and f.name.upper() not in _skip_upper
    ]

    geometries, orig_fids, attr_rows = [], [], []
    with arcpy.da.SearchCursor(in_features, ["OID@", "SHAPE@WKT"] + attr_field_names) as cursor:
        for row in cursor:
            geometries.append(wkt_loads(row[1]) if row[1] else None)
            orig_fids.append(row[0])
            attr_rows.append(row[2:])

    attr_df = pd.DataFrame(attr_rows, columns=attr_field_names)
    attr_df.insert(0, "ORIG_FID", orig_fids)
    gdf = gpd.GeoDataFrame(attr_df, geometry=geometries)
    epsg = spatial_ref.factoryCode if spatial_ref else 0
    if epsg > 0:
        try:
            gdf = gdf.set_crs(epsg=epsg)
        except Exception:
            pass

    # Compute
    result_gdf = polygon_to_centerline(
        gdf,
        method=method,
        densify_distance=densify_distance,
        prune_threshold=prune_threshold,
        smooth_sigma=smooth_sigma,
        raster_resolution=raster_resolution,
        single_line=single_line,
    )

    # Write output
    out_dir = os.path.dirname(out_features) or arcpy.env.scratchGDB
    out_name = os.path.basename(out_features)
    arcpy.env.overwriteOutput = True
    arcpy.management.CreateFeatureclass(
        out_dir, out_name, "POLYLINE",
        spatial_reference=spatial_ref if (spatial_ref and spatial_ref.name != "Unknown") else None,
    )
    arcpy.management.AddField(out_features, "ORIG_FID", "LONG")

    with arcpy.da.InsertCursor(out_features, ["SHAPE@WKT", "ORIG_FID"]) as cursor:
        for _, row in result_gdf.iterrows():
            geom = row.geometry
            if geom is not None and not geom.is_empty:
                cursor.insertRow([geom.wkt, row.get("ORIG_FID")])

    return result_gdf
