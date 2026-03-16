# -*- coding: utf-8 -*-
"""
Pure_Centerline.pyt
===================
ArcGIS Python Toolbox that wraps the shapely-free centerline algorithm
implemented in ``centerline_pure.py``.

Why use this toolbox instead of ``gdal_centerline/GDAL_Centerline.pyt``?
------------------------------------------------------------------------
This toolbox requires **only two third-party packages** (scipy and networkx)
in addition to numpy, which ships with every ArcGIS Pro installation.
Notably, it does **not** require:

    * shapely
    * geopandas
    * pandas

Those packages sometimes cause licence-conflict warnings in ArcGIS Pro that
appear as "Tool not licensed", even when the user holds valid ArcGIS licences.
This toolbox avoids that issue entirely by replacing every shapely call with
pure-numpy geometry arithmetic (ray-casting, segment-intersection tests).

Runtime dependencies
--------------------
The following packages must be available in the ArcGIS Pro Python environment:

    numpy    – pre-installed in every ArcGIS Pro Python environment.
    scipy    – typically pre-installed in ArcGIS Pro (used for Voronoi).
    networkx – install from conda-forge (see install_dependencies.bat):
                   conda install -c conda-forge networkx

For ``method=skeleton`` also install scikit-image:
                   conda install -c conda-forge scikit-image

How to load this toolbox
------------------------
In **ArcGIS Pro** (Catalog pane) or **ArcCatalog**:
  1. Right-click a folder → Add Toolbox → select ``Pure_Centerline.pyt``.
  2. Expand the toolbox and run **Polygon to Centerline (Pure)**.

``centerline_pure.py`` must be in the **same directory** as this ``.pyt``
file so that it can be imported at run-time.
"""

import os
import sys

import arcpy

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# ArcGIS system fields that must never be copied to the output feature class.
_SYSTEM_FIELDS = frozenset({
    "SHAPE", "SHAPE_LENGTH", "SHAPE_AREA",
    "SHAPE.STLENGTH()", "SHAPE.STAREA()",
})

# arcpy field-type string → AddField() type token.
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
    "Raster":          "TEXT",  # unsupported — fall back to TEXT
    "Blob":            "TEXT",  # unsupported — fall back to TEXT
}

# ---------------------------------------------------------------------------
# Pre-flight dependency check (runs once when the toolbox is loaded)
# ---------------------------------------------------------------------------


def _check_dependencies():
    """
    Return a list of package names that are NOT currently importable.
    An empty list means all required packages are present.
    """
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
    "folder as this toolbox (pure_centerline/).  See README.md for\n"
    "full instructions.\n"
    "\n"
    "Manual installation (ArcGIS Pro Python Command Prompt):\n"
    "\n"
    "  Step 1 — Clone the default environment (only once):\n"
    "    conda create --name arcgispro-py3-pure --clone arcgispro-py3\n"
    "\n"
    "  Step 2 — Install the packages into the clone:\n"
    "    activate arcgispro-py3-pure\n"
    "    conda install -c conda-forge -y networkx\n"
    "\n"
    "  Step 3 — Set the clone as the active environment in ArcGIS Pro:\n"
    "    Project > Python > Python Environments > arcgispro-py3-pure\n"
    "    Restart ArcGIS Pro.\n"
    "\n"
    "  Note: numpy and scipy are usually already present in the default\n"
    "  'arcgispro-py3' environment shipped with ArcGIS Pro.\n"
    "  Only networkx typically needs to be installed from conda-forge.\n"
)

# ---------------------------------------------------------------------------
# Toolbox
# ---------------------------------------------------------------------------


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the .pyt file)."""
        self.label = "Pure Centerline"
        self.alias = "PureCenterline"
        self.tools = [PolygonToCenterlinePure]


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


class PolygonToCenterlinePure(object):
    """ArcGIS tool that wraps the shapely-free polygon_to_centerline_wkt() function."""

    def __init__(self):
        self.label = "Polygon to Centerline (Pure)"
        self.description = (
            "Converts polygon features to centerline polylines using a "
            "shapely-free open-source implementation.\n\n"
            "Unlike the GDAL Centerline toolbox, this tool requires only "
            "numpy (pre-installed), scipy (usually pre-installed), and "
            "networkx (conda-forge).  No shapely, geopandas, or pandas "
            "are needed.\n\n"
            "Two algorithms are available:\n"
            "  voronoi  — Vector-based Voronoi / Thiessen skeleton (default).\n"
            "             Fast, precise, works well for elongated polygons.\n"
            "  skeleton — Raster-based morphological skeletonisation.\n"
            "             Requires scikit-image; better for jagged shapes.\n\n"
            "Both methods work with any ArcGIS licence level "
            "(Basic / Standard / Advanced) and correctly handle polygons "
            "with interior holes (e.g. ring-shaped or letter 'O' / 'A' shapes)."
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
        All geometry operations are performed by open-source Python libraries;
        no Standard/Advanced geoprocessing tools are called.

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
        parameters[4].enabled = method == "voronoi"   # prune threshold
        parameters[5].enabled = method == "skeleton"  # smooth sigma
        parameters[6].enabled = method == "skeleton"  # raster resolution

    def updateMessages(self, parameters):
        """Validate parameter values and surface dependency errors."""

        # If required packages are missing, set a visible error on the input
        # feature parameter so the user sees the install instructions in the
        # tool dialog before clicking Run — without triggering the misleading
        # "Tool not licensed" lock that isLicensed() returning False would cause.
        if _MISSING_DEPS:
            parameters[0].setErrorMessage(
                _INSTALL_HELP.format(missing=", ".join(_MISSING_DEPS))
            )
            return  # skip remaining validation while deps are absent

        for idx in (3, 4, 5):
            param = parameters[idx]
            if param.value is not None and float(param.value) < 0:
                param.setErrorMessage("Value must be >= 0.")

        res_param = parameters[6]
        if res_param.value is not None and float(res_param.value) <= 0:
            res_param.setErrorMessage("Raster Resolution must be > 0.")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, parameters, messages):
        """Run the shapely-free centerline algorithm."""

        # Add the toolbox directory to sys.path so we can import centerline_pure.py
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

        # ---- Import the core algorithm module ------------------------------
        try:
            from centerline_pure import polygon_to_centerline_wkt
        except ImportError:
            messages.addErrorMessage(
                "Could not import 'centerline_pure' module.\n"
                "Ensure 'centerline_pure.py' is in the same folder as this toolbox:\n"
                "  {}".format(tbx_dir)
            )
            raise

        # ---- Check for missing packages ------------------------------------
        if _MISSING_DEPS:
            messages.addErrorMessage(
                _INSTALL_HELP.format(missing=", ".join(_MISSING_DEPS))
            )
            raise RuntimeError("Missing required packages: {}".format(
                ", ".join(_MISSING_DEPS)
            ))

        # ---- Read input polygons via arcpy cursor --------------------------
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
            "         {:,} polygon(s) read.".format(len(input_rows))
        )

        if not input_rows:
            messages.addWarningMessage("No polygon features found in the input layer.")
            return

        # ---- Compute centerlines -------------------------------------------
        messages.addMessage(
            "Step 2/3  Computing centerlines "
            "(method={}, densify={}) ...".format(method, densify_distance)
        )

        results = []   # list of (result_wkt, orig_fid, attr_dict)
        n_skipped = 0

        for row in input_rows:
            orig_fid = row[0]
            wkt = row[1]
            attr_dict = {
                name: val
                for name, val in zip(attr_field_names, row[2:])
            }

            if not wkt:
                n_skipped += 1
                continue

            try:
                result_wkt = polygon_to_centerline_wkt(
                    wkt,
                    method=method,
                    densify_distance=densify_distance,
                    prune_threshold=prune_threshold,
                    smooth_sigma=smooth_sigma,
                    raster_resolution=raster_resolution,
                    single_line=single_line,
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
            "         {:,} centerline(s) generated{}.".format(
                n_out,
                " ({:,} polygon(s) skipped — too small or degenerate)".format(
                    n_skipped
                ) if n_skipped > 0 else "",
            )
        )

        if n_out == 0:
            messages.addWarningMessage(
                "No centerlines were generated.  "
                "Try a smaller Densification Distance."
            )
            return

        # ---- Write output feature class ------------------------------------
        messages.addMessage("Step 3/3  Writing output feature class ...")

        out_dir = os.path.dirname(out_features)
        out_name = os.path.basename(out_features)
        if not out_dir:
            out_dir = arcpy.env.scratchGDB

        arcpy.management.CreateFeatureclass(
            out_dir,
            out_name,
            "POLYLINE",
            spatial_reference=(
                spatial_ref
                if (spatial_ref and spatial_ref.name != "Unknown")
                else None
            ),
        )
        arcpy.env.overwriteOutput = True

        # Add ORIG_FID field to link back to input polygons
        arcpy.management.AddField(
            out_features, "ORIG_FID", "LONG", field_alias="Original Feature ID"
        )

        # Add the same attribute fields that existed in the input, preserving
        # their original field types via _ARCPY_FIELD_TYPE_MAP.
        existing_out_field_names = {
            f.name.upper() for f in arcpy.ListFields(out_features)
        }
        fields_to_add = []
        for src_field in attr_fields:
            fname = src_field.name
            if fname.upper() not in existing_out_field_names:
                ftype = _ARCPY_FIELD_TYPE_MAP.get(src_field.type, "TEXT")
                field_length = (
                    src_field.length if src_field.type == "String" else None
                )
                try:
                    if field_length:
                        arcpy.management.AddField(
                            out_features, fname, ftype,
                            field_length=field_length,
                        )
                    else:
                        arcpy.management.AddField(out_features, fname, ftype)
                    fields_to_add.append(fname)
                    existing_out_field_names.add(fname.upper())
                except Exception:
                    pass  # skip fields that cannot be added (e.g. reserved names)
            else:
                fields_to_add.append(fname)

        _field_type_by_name = {f.name: f.type for f in attr_fields}

        # Insert centerline features
        insert_fields = ["SHAPE@WKT", "ORIG_FID"] + fields_to_add
        with arcpy.da.InsertCursor(out_features, insert_fields) as cursor:
            for result_wkt, orig_fid, attr_dict in results:
                attr_vals = []
                for fname in fields_to_add:
                    val = attr_dict.get(fname)
                    if val is None:
                        attr_vals.append(None)
                    elif _field_type_by_name.get(fname) == "String":
                        attr_vals.append(str(val))
                    else:
                        attr_vals.append(val)
                cursor.insertRow([result_wkt, orig_fid] + attr_vals)

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
    Run the shapely-free centerline algorithm from a plain Python script.

    Parameters
    ----------
    in_features : str
        Path to an ArcGIS feature class or any vector file.
    out_features : str
        Path for the output polyline feature class.
    method : {"voronoi", "skeleton"}
    densify_distance : float
    prune_threshold : float
    smooth_sigma : float
    raster_resolution : float or None
    single_line : bool

    Returns
    -------
    str
        Path to the output feature class.
    """
    tbx_dir = os.path.dirname(os.path.abspath(__file__))
    if tbx_dir not in sys.path:
        sys.path.insert(0, tbx_dir)

    from centerline_pure import polygon_to_centerline_wkt

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

    results = []
    with arcpy.da.SearchCursor(in_features, cursor_fields) as cursor:
        for row in cursor:
            orig_fid, wkt = row[0], row[1]
            attr_dict = {
                name: val for name, val in zip(attr_field_names, row[2:])
            }
            if not wkt:
                continue
            result_wkt = polygon_to_centerline_wkt(
                wkt,
                method=method,
                densify_distance=densify_distance,
                prune_threshold=prune_threshold,
                smooth_sigma=smooth_sigma,
                raster_resolution=raster_resolution,
                single_line=single_line,
            )
            if result_wkt:
                results.append((result_wkt, orig_fid, attr_dict))

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
        out_features, "ORIG_FID", "LONG", field_alias="Original Feature ID"
    )

    existing = {f.name.upper() for f in arcpy.ListFields(out_features)}
    fields_to_add = []
    for src_field in attr_fields:
        fname = src_field.name
        if fname.upper() not in existing:
            ftype = _ARCPY_FIELD_TYPE_MAP.get(src_field.type, "TEXT")
            fl = src_field.length if src_field.type == "String" else None
            try:
                if fl:
                    arcpy.management.AddField(out_features, fname, ftype,
                                              field_length=fl)
                else:
                    arcpy.management.AddField(out_features, fname, ftype)
                fields_to_add.append(fname)
                existing.add(fname.upper())
            except Exception:
                pass
        else:
            fields_to_add.append(fname)

    _field_type_by_name = {f.name: f.type for f in attr_fields}
    insert_fields = ["SHAPE@WKT", "ORIG_FID"] + fields_to_add

    with arcpy.da.InsertCursor(out_features, insert_fields) as cursor:
        for result_wkt, orig_fid, attr_dict in results:
            attr_vals = [
                str(attr_dict.get(f)) if _field_type_by_name.get(f) == "String"
                else attr_dict.get(f)
                for f in fields_to_add
            ]
            cursor.insertRow([result_wkt, orig_fid] + attr_vals)

    return out_features
