# -*- coding: utf-8 -*-
"""
Proportional_Buffer.pyt
=======================
ArcGIS Python Toolbox – Proportional Buffer Along Centerlines.

This toolbox is **completely standalone**: it does **not** call any
centerline-extraction code and does **not** require shapely or geopandas.

Inputs
------
1. Channel / waterway polygon feature class.
2. Pre-computed centerline polyline feature class (produced by any tool).

Output
------
Buffer polygons whose cross-sectional width at every point equals
``buffer_ratio × 2 × local_half_width``, where the local half-width is
estimated as the distance from the centerline sample point to the nearest
densified polygon-boundary vertex (KDTree query, O(N log M)).

Algorithm
---------
All geometry operations are implemented in pure NumPy / SciPy so that the
toolbox works with any ArcGIS licence level (Basic / Standard / Advanced)
and does **not** require shapely or geopandas.

1. For each polygon, find its matching centerlines via a spatial-join
   (``arcpy.SpatialJoin_analysis``).
2. For each polygon / centerline pair:
   a. Extract boundary coordinates → densify → build ``scipy.spatial.cKDTree``.
   b. Resample the centerline at ``sample_distance`` intervals.
   c. Query the KDTree for the local half-width at each sample point.
   d. Compute left / right offset points using normal vectors.
   e. Assemble a closed polygon ring; optionally add round end-caps.
   f. Optionally apply Chaikin smoothing and clip to the source polygon.
3. Write each buffer feature with width-statistic attributes.

Runtime dependencies
--------------------
    numpy    – pre-installed in every ArcGIS Pro Python environment.
    scipy    – typically pre-installed in ArcGIS Pro (1.x+).

No additional packages are required.

How to load this toolbox
------------------------
In **ArcGIS Pro** (Catalog pane):
  1. Right-click a folder → Add Toolbox.
  2. Browse to and select ``Proportional_Buffer.pyt``.
  3. Expand the toolbox and run **Proportional Buffer Along Centerlines**.
"""

import math
import os
import sys

import arcpy

# ---------------------------------------------------------------------------
# Pre-flight dependency check
# ---------------------------------------------------------------------------


def _check_dependencies():
    missing = []
    for pkg in ("numpy", "scipy"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


_MISSING_DEPS = _check_dependencies()

_INSTALL_HELP = (
    "\n"
    "REQUIRED PACKAGES ARE NOT INSTALLED\n"
    "=====================================\n"
    "Missing: {missing}\n"
    "\n"
    "numpy and scipy are normally pre-installed in ArcGIS Pro.\n"
    "If they are missing, open the ArcGIS Pro Python Command Prompt and run:\n"
    "\n"
    "    conda install -c conda-forge -y numpy scipy\n"
    "\n"
    "Then restart ArcGIS Pro.\n"
)


# ---------------------------------------------------------------------------
# Pure NumPy / SciPy geometry helpers
# (no shapely, no geopandas)
# ---------------------------------------------------------------------------


def _densify_coords(coords, max_spacing):
    """
    Resample a coordinate sequence so consecutive points are at most
    *max_spacing* apart.

    Parameters
    ----------
    coords : numpy.ndarray, shape (N, 2)
    max_spacing : float

    Returns
    -------
    numpy.ndarray, shape (M, 2)
    """
    import numpy as np

    coords = np.asarray(coords, dtype=float)
    if len(coords) < 2 or max_spacing <= 0:
        return coords

    diffs = np.diff(coords, axis=0)
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum_len[-1]
    if total <= 0.0:
        return coords

    n = max(2, int(math.ceil(total / max_spacing)) + 1)
    t = np.linspace(0.0, total, n)
    x = np.interp(t, cum_len, coords[:, 0])
    y = np.interp(t, cum_len, coords[:, 1])
    return np.column_stack([x, y])


def _compute_normals(sample_pts):
    """
    Compute left-side unit normal vectors at each sample point.

    Parameters
    ----------
    sample_pts : numpy.ndarray, shape (N, 2)

    Returns
    -------
    numpy.ndarray, shape (N, 2)
    """
    import numpy as np

    n = len(sample_pts)
    tangents = np.zeros((n, 2), dtype=float)
    tangents[1:-1] = sample_pts[2:] - sample_pts[:-2]
    if n >= 2:
        tangents[0] = sample_pts[1] - sample_pts[0]
        tangents[-1] = sample_pts[-1] - sample_pts[-2]

    lengths = np.hypot(tangents[:, 0], tangents[:, 1])
    lengths = np.where(lengths < 1e-12, 1.0, lengths)
    tangents /= lengths[:, np.newaxis]

    # Rotate 90° CCW → left-hand normal
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    return normals


def _arc_points(center, radius, angle_start, angle_end, n_pts):
    """Generate arc coordinate array between two angles (radians)."""
    import numpy as np

    angles = np.linspace(angle_start, angle_end, n_pts)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack([x, y])


def _chaikin_smooth(coords, iterations):
    """Apply Chaikin corner-cutting smoothing to a closed ring."""
    import numpy as np

    pts = np.asarray(coords, dtype=float)
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    for _ in range(max(0, iterations)):
        n = len(pts) - 1
        new_pts = np.empty((2 * n, 2), dtype=float)
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            new_pts[2 * i] = 0.75 * p0 + 0.25 * p1
            new_pts[2 * i + 1] = 0.25 * p0 + 0.75 * p1
        pts = np.vstack([new_pts, new_pts[0]])

    return pts


def _arcpy_polygon_rings(shape):
    """
    Extract coordinate arrays for all rings of an arcpy Polygon geometry.

    Returns a list of numpy arrays ``[(N_i, 2), …]``.
    The first element is the exterior ring; subsequent elements are holes.
    """
    import numpy as np

    rings = []
    for part_idx in range(shape.partCount):
        part = shape.getPart(part_idx)
        current = []
        for pt in part:
            if pt is None:
                # None separates rings within a multi-ring part
                if len(current) >= 2:
                    rings.append(np.array(current, dtype=float))
                current = []
            else:
                current.append([pt.X, pt.Y])
        if len(current) >= 2:
            rings.append(np.array(current, dtype=float))
    return rings


def _arcpy_polyline_parts(shape):
    """
    Extract coordinate arrays for all parts of an arcpy Polyline geometry.

    Returns a list of numpy arrays ``[(N_i, 2), …]``.
    """
    import numpy as np

    parts = []
    for part_idx in range(shape.partCount):
        part = shape.getPart(part_idx)
        coords = []
        for pt in part:
            if pt is not None:
                coords.append([pt.X, pt.Y])
        if len(coords) >= 2:
            parts.append(np.array(coords, dtype=float))
    return parts


def _build_buffer_ring(
    sample_pts,
    half_widths,
    normals,
    buffer_ratio,
    width_min,
    width_max,
    end_cap,
    n_cap_pts,
):
    """
    Build the closed coordinate ring for one variable-width buffer segment.

    Parameters
    ----------
    sample_pts : numpy.ndarray, shape (N, 2)
    half_widths : numpy.ndarray, shape (N,)
    normals : numpy.ndarray, shape (N, 2)
    buffer_ratio : float
    width_min : float
    width_max : float
    end_cap : str – 'round' or 'flat'
    n_cap_pts : int

    Returns
    -------
    numpy.ndarray, shape (M, 2) – closed ring
    """
    import numpy as np

    radii = np.clip(
        half_widths * buffer_ratio,
        width_min / 2.0,
        width_max / 2.0,
    )

    left_pts = sample_pts + radii[:, np.newaxis] * normals
    right_pts = sample_pts - radii[:, np.newaxis] * normals

    ring = []
    ring.extend(left_pts.tolist())

    if end_cap == "round":
        nrm = normals[-1]
        angle_L = math.atan2(nrm[1], nrm[0])
        arc = _arc_points(sample_pts[-1], radii[-1], angle_L, angle_L - math.pi, n_cap_pts)
        ring.extend(arc.tolist())
    else:
        ring.append(right_pts[-1].tolist())

    ring.extend(right_pts[::-1].tolist())

    if end_cap == "round":
        nrm = normals[0]
        angle_R = math.atan2(-nrm[1], -nrm[0])
        arc = _arc_points(sample_pts[0], radii[0], angle_R, angle_R - math.pi, n_cap_pts)
        ring.extend(arc.tolist())
    else:
        ring.append(left_pts[0].tolist())

    ring.append(ring[0])
    return np.array(ring, dtype=float)


def _ring_to_arcpy_polygon(ring_coords):
    """Convert a (N, 2) numpy ring array to an arcpy Polygon geometry."""
    arr = arcpy.Array([arcpy.Point(x, y) for x, y in ring_coords])
    return arcpy.Polygon(arr)


def _compute_buffer_for_pair(
    poly_shape,
    cl_shape,
    buffer_ratio,
    sample_distance,
    end_cap,
    width_min,
    width_max,
    n_cap_pts,
    smooth_iterations,
    clip_to_polygon,
):
    """
    Compute the proportional buffer for one polygon / centerline pair.

    Parameters
    ----------
    poly_shape : arcpy Polygon geometry
    cl_shape   : arcpy Polyline geometry
    … (see tool parameters)

    Returns
    -------
    arcpy.Polygon or None
        The buffer geometry, or *None* if it could not be computed.
    tuple (w_min, w_max, w_mean) : float
        Width statistics (2 × half-width, CRS units).
    """
    import numpy as np
    from scipy.spatial import cKDTree

    # Extract boundary points from polygon and build KDTree
    rings = _arcpy_polygon_rings(poly_shape)
    if not rings:
        return None, (0.0, 0.0, 0.0)

    dense_rings = [_densify_coords(r, sample_distance / 2.0) for r in rings]
    boundary_pts = np.vstack(dense_rings)
    tree = cKDTree(boundary_pts)

    # Extract and sample each centerline part
    cl_parts = _arcpy_polyline_parts(cl_shape)
    if not cl_parts:
        return None, (0.0, 0.0, 0.0)

    all_rings = []
    all_hw = []

    for part_coords in cl_parts:
        seg_pts = _densify_coords(part_coords, sample_distance)
        if len(seg_pts) < 2:
            continue

        dists, _ = tree.query(seg_pts)
        all_hw.extend(dists.tolist())

        normals = _compute_normals(seg_pts)
        ring = _build_buffer_ring(
            seg_pts,
            dists,
            normals,
            buffer_ratio,
            width_min,
            width_max,
            end_cap,
            n_cap_pts,
        )

        if smooth_iterations > 0:
            ring = _chaikin_smooth(ring, smooth_iterations)

        all_rings.append(ring)

    if not all_rings:
        return None, (0.0, 0.0, 0.0)

    # Build arcpy geometry for each ring and union them
    buf_geoms = [_ring_to_arcpy_polygon(r) for r in all_rings]
    result = buf_geoms[0]
    for g in buf_geoms[1:]:
        result = result.union(g)

    if clip_to_polygon:
        result = result.intersect(poly_shape, 4)  # 4 = polygon output

    # Width statistics
    if all_hw:
        hw_arr = np.array(all_hw)
        stats = (
            float(2.0 * np.min(hw_arr)),
            float(2.0 * np.max(hw_arr)),
            float(2.0 * np.mean(hw_arr)),
        )
    else:
        stats = (0.0, 0.0, 0.0)

    return result, stats


def _auto_sample_distance_arcpy(poly_shape):
    """
    Estimate a sensible sample distance from an arcpy Polygon shape.
    Uses hydraulic-radius approximation: min_hw ≈ area / perimeter.
    """
    try:
        area = poly_shape.area
        length = poly_shape.length  # perimeter
        if length <= 0:
            return 1.0
        min_hw = area / length
        return max(min_hw / 10.0, 1e-6)
    except Exception:
        return 1.0


def _parse_linear_unit_value(value_str):
    """Return the numeric part of an ArcGIS linear unit string, e.g. '5 Meters' → 5.0."""
    try:
        return float(str(value_str).split()[0])
    except (ValueError, IndexError, AttributeError):
        return 0.0


# ---------------------------------------------------------------------------
# Toolbox
# ---------------------------------------------------------------------------


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the .pyt file)."""
        self.label = "Proportional Buffer"
        self.alias = "ProportionalBuffer"
        self.tools = [ProportionalBufferTool]


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


class ProportionalBufferTool(object):
    """ArcGIS tool: Proportional Buffer Along Centerlines."""

    def __init__(self):
        self.label = "Proportional Buffer Along Centerlines"
        self.description = (
            "Generates variable-width buffer polygons along pre-existing "
            "centerlines, where the buffer width at every cross-section is "
            "proportional to the local channel width.\n\n"
            "Inputs\n"
            "------\n"
            "1. Channel / waterway polygon feature class.\n"
            "2. Pre-computed centerline polyline feature class (produced by\n"
            "   any tool, or supplied manually).\n\n"
            "Algorithm\n"
            "---------\n"
            "For each polygon the tool:\n"
            "  1. Samples the matched centerline(s) at regular intervals.\n"
            "  2. Estimates the local half-width at each sample point as the\n"
            "     distance to the nearest polygon boundary vertex "
            "(via KDTree).\n"
            "  3. Offsets left and right by buffer_ratio × half-width.\n"
            "  4. Assembles a closed buffer polygon (with optional round\n"
            "     end-caps and Chaikin smoothing).\n\n"
            "This tool requires only numpy and scipy (both pre-installed in\n"
            "ArcGIS Pro).  No shapely, geopandas, or ArcGIS Advanced licence\n"
            "features are needed."
        )
        self.canRunInBackground = True

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def getParameterInfo(self):
        """Define the tool parameters shown in the ArcGIS dialog."""

        # 0 – Input polygon features
        p_poly = arcpy.Parameter(
            displayName="Input Polygon Features",
            name="in_polygon",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input",
        )
        p_poly.filter.list = ["Polygon"]

        # 1 – Input centerline features
        p_cl = arcpy.Parameter(
            displayName="Input Centerline Features",
            name="in_centerline",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input",
        )
        p_cl.filter.list = ["Polyline"]

        # 2 – Output feature class
        p_out = arcpy.Parameter(
            displayName="Output Buffer Feature Class",
            name="out_buffer",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Output",
        )

        # 3 – Buffer ratio
        p_ratio = arcpy.Parameter(
            displayName="Buffer Ratio (0 < ratio ≤ 1)",
            name="buffer_ratio",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input",
        )
        p_ratio.value = 0.5
        p_ratio.filter.type = "Range"
        p_ratio.filter.list = [0.0001, 1.0]

        # 4 – Sample distance
        p_sample = arcpy.Parameter(
            displayName="Sample Distance (blank = automatic)",
            name="sample_distance",
            datatype="GPLinearUnit",
            parameterType="Optional",
            direction="Input",
        )
        p_sample.value = ""

        # 5 – End-cap style
        p_cap = arcpy.Parameter(
            displayName="End-Cap Style",
            name="end_cap",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )
        p_cap.filter.type = "ValueList"
        p_cap.filter.list = ["ROUND", "FLAT"]
        p_cap.value = "ROUND"

        # 6 – Clip to polygon
        p_clip = arcpy.Parameter(
            displayName="Clip Buffer to Polygon Boundary",
            name="clip_to_polygon",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
        )
        p_clip.value = True

        # 7 – Smooth iterations
        p_smooth = arcpy.Parameter(
            displayName="Chaikin Smoothing Iterations (0 = none)",
            name="smooth_iterations",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
        )
        p_smooth.value = 0
        p_smooth.filter.type = "Range"
        p_smooth.filter.list = [0, 10]

        # 8 – Minimum buffer width
        p_wmin = arcpy.Parameter(
            displayName="Minimum Buffer Width (0 = no minimum)",
            name="width_min",
            datatype="GPLinearUnit",
            parameterType="Optional",
            direction="Input",
        )
        p_wmin.value = "0 Meters"

        # 9 – Maximum buffer width
        p_wmax = arcpy.Parameter(
            displayName="Maximum Buffer Width (blank = no maximum)",
            name="width_max",
            datatype="GPLinearUnit",
            parameterType="Optional",
            direction="Input",
        )
        p_wmax.value = ""

        return [
            p_poly, p_cl, p_out,
            p_ratio, p_sample, p_cap,
            p_clip, p_smooth,
            p_wmin, p_wmax,
        ]

    # ------------------------------------------------------------------
    # Licensing
    # ------------------------------------------------------------------

    def isLicensed(self):
        """Allow the tool to execute regardless of licence level."""
        if _MISSING_DEPS:
            return False
        return True

    # ------------------------------------------------------------------
    # Validation hooks
    # ------------------------------------------------------------------

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        if _MISSING_DEPS:
            msg = _INSTALL_HELP.format(missing=", ".join(_MISSING_DEPS))
            parameters[0].setErrorMessage(msg)
        return

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, parameters, messages):
        """Run the tool."""
        in_polygon = parameters[0].valueAsText
        in_centerline = parameters[1].valueAsText
        out_buffer = parameters[2].valueAsText
        buffer_ratio = float(parameters[3].value) if parameters[3].value else 0.5
        sample_distance_str = parameters[4].valueAsText
        end_cap = (parameters[5].valueAsText or "ROUND").upper()
        clip_to_polygon = bool(parameters[6].value) if parameters[6].value is not None else True
        smooth_iterations = int(parameters[7].value) if parameters[7].value else 0
        width_min_str = parameters[8].valueAsText
        width_max_str = parameters[9].valueAsText

        width_min = _parse_linear_unit_value(width_min_str) if width_min_str else 0.0
        width_max_val = _parse_linear_unit_value(width_max_str) if width_max_str else None
        width_max = width_max_val if width_max_val and width_max_val > 0 else float("inf")

        end_cap_lower = "round" if end_cap == "ROUND" else "flat"

        arcpy.env.overwriteOutput = True
        workspace = arcpy.env.scratchGDB

        def log(msg):
            messages.addMessage(msg)

        try:
            _run_tool(
                in_polygon=in_polygon,
                in_centerline=in_centerline,
                out_buffer=out_buffer,
                buffer_ratio=buffer_ratio,
                sample_distance_str=sample_distance_str,
                end_cap=end_cap_lower,
                clip_to_polygon=clip_to_polygon,
                smooth_iterations=smooth_iterations,
                width_min=width_min,
                width_max=width_max,
                workspace=workspace,
                log=log,
            )
        except arcpy.ExecuteError:
            messages.addErrorMessage(arcpy.GetMessages(2))
            raise
        except Exception as exc:
            messages.addErrorMessage(str(exc))
            raise

    def postExecute(self, parameters):
        return


# ---------------------------------------------------------------------------
# Core execution function (also callable from external scripts)
# ---------------------------------------------------------------------------


def _run_tool(
    in_polygon,
    in_centerline,
    out_buffer,
    buffer_ratio=0.5,
    sample_distance_str=None,
    end_cap="round",
    clip_to_polygon=True,
    smooth_iterations=0,
    width_min=0.0,
    width_max=float("inf"),
    workspace=None,
    log=None,
    n_cap_pts=8,
):
    """
    Core execution logic, callable independently of the ArcGIS GUI.

    Parameters
    ----------
    in_polygon : str
        Path to the input polygon feature class.
    in_centerline : str
        Path to the input centerline feature class.
    out_buffer : str
        Path for the output buffer feature class.
    buffer_ratio : float
    sample_distance_str : str or None
        ArcGIS linear unit string, e.g. ``'10 Meters'``.
        ``None`` or empty → automatic selection per polygon.
    end_cap : str
        ``'round'`` or ``'flat'``.
    clip_to_polygon : bool
    smooth_iterations : int
    width_min : float
    width_max : float
    workspace : str or None
        Scratch GDB path.
    log : callable or None
        Message function (e.g. ``messages.addMessage``).
    n_cap_pts : int
        Arc interpolation points for round end-caps.
    """
    if workspace is None:
        workspace = arcpy.env.scratchGDB
    if log is None:
        log = arcpy.AddMessage

    arcpy.env.overwriteOutput = True

    def tmp(name):
        return os.path.join(workspace, "pb_" + name)

    # ── Parse sample distance ────────────────────────────────────────────────
    fixed_sample_distance = None
    if sample_distance_str:
        v = _parse_linear_unit_value(sample_distance_str)
        if v > 0:
            fixed_sample_distance = v

    # ── Step 1: Spatial join centerlines → polygons ──────────────────────────
    log("Step 1/4  Matching centerlines to polygons (spatial join) …")
    joined_cl = tmp("joined_cl")
    arcpy.SpatialJoin_analysis(
        target_features=in_centerline,
        join_features=in_polygon,
        out_feature_class=joined_cl,
        join_operation="JOIN_ONE_TO_MANY",
        join_type="KEEP_COMMON",
        match_option="INTERSECT",
    )

    n_joined = int(arcpy.GetCount_management(joined_cl).getOutput(0))
    log(f"         {n_joined} centerline–polygon match(es) found.")
    if n_joined == 0:
        arcpy.Delete_management(joined_cl)
        raise RuntimeError(
            "No centerlines intersect any polygon.  "
            "Check that the two layers share the same coordinate system "
            "and that centerlines actually overlap their polygons."
        )

    # ── Step 2: Create output feature class ──────────────────────────────────
    log("Step 2/4  Creating output feature class …")
    out_dir = os.path.dirname(out_buffer)
    out_name = os.path.basename(out_buffer)
    sr = arcpy.Describe(in_polygon).spatialReference
    arcpy.CreateFeatureclass_management(out_dir, out_name, "POLYGON", spatial_reference=sr)

    # Add attribute fields
    for fname, ftype, flength in [
        ("src_poly_oid", "LONG", None),
        ("buffer_ratio", "DOUBLE", None),
        ("width_min_m", "DOUBLE", None),
        ("width_max_m", "DOUBLE", None),
        ("width_mean_m", "DOUBLE", None),
    ]:
        kwargs = {"field_length": flength} if flength else {}
        arcpy.AddField_management(out_buffer, fname, ftype, **kwargs)

    # ── Step 3: Process each polygon ─────────────────────────────────────────
    log("Step 3/4  Computing buffers …")

    # Build a dict: polygon_OID → polygon_shape
    poly_shapes = {}
    with arcpy.da.SearchCursor(in_polygon, ["OID@", "SHAPE@"]) as cur:
        for oid, shape in cur:
            poly_shapes[oid] = shape

    # Build a dict: polygon_OID → list of centerline shapes
    # The SpatialJoin output has a "JOIN_FID" field pointing to the source polygon OID.
    # Field name for the joined polygon OID in the joined feature class:
    joined_fields = [f.name for f in arcpy.ListFields(joined_cl)]
    # ArcGIS names the join OID field "JOIN_FID" for SpatialJoin by default,
    # or it may be stored as "FID_<layer_name>".  Find it robustly.
    poly_oid_field = None
    for candidate in ("JOIN_FID", "FID_" + os.path.basename(in_polygon)):
        if candidate in joined_fields:
            poly_oid_field = candidate
            break
    if poly_oid_field is None:
        # Fall back: find the first LONG/INTEGER field that is not OID@
        for f in arcpy.ListFields(joined_cl):
            if f.type in ("Integer", "SmallInteger", "Long") and f.name not in (
                "OBJECTID", "FID"
            ):
                poly_oid_field = f.name
                break

    if poly_oid_field is None:
        raise RuntimeError(
            "Could not identify the polygon OID field in the spatial-join "
            "output.  Please report this issue."
        )

    cl_map = {}  # poly_oid → [shape, shape, …]
    with arcpy.da.SearchCursor(joined_cl, [poly_oid_field, "SHAPE@"]) as cur:
        for poly_oid, cl_shape in cur:
            cl_map.setdefault(poly_oid, []).append(cl_shape)

    out_fields = ["SHAPE@", "src_poly_oid", "buffer_ratio", "width_min_m", "width_max_m", "width_mean_m"]
    n_success = 0
    n_failed = 0

    with arcpy.da.InsertCursor(out_buffer, out_fields) as insert_cur:
        for poly_oid, poly_shape in poly_shapes.items():
            cl_shapes = cl_map.get(poly_oid, [])
            if not cl_shapes:
                continue

            # Merge all centerline parts for this polygon into one combined shape
            if len(cl_shapes) == 1:
                combined_cl = cl_shapes[0]
            else:
                arr = arcpy.Array()
                for cs in cl_shapes:
                    for pi in range(cs.partCount):
                        arr.add(cs.getPart(pi))
                combined_cl = arcpy.Polyline(arr)

            # Determine sample distance for this polygon
            sd = fixed_sample_distance
            if sd is None:
                sd = _auto_sample_distance_arcpy(poly_shape)

            try:
                buf_shape, (w_min, w_max, w_mean) = _compute_buffer_for_pair(
                    poly_shape=poly_shape,
                    cl_shape=combined_cl,
                    buffer_ratio=buffer_ratio,
                    sample_distance=sd,
                    end_cap=end_cap,
                    width_min=width_min,
                    width_max=width_max,
                    n_cap_pts=n_cap_pts,
                    smooth_iterations=smooth_iterations,
                    clip_to_polygon=clip_to_polygon,
                )
                if buf_shape is None or buf_shape.area == 0:
                    n_failed += 1
                    arcpy.AddWarning(
                        f"Polygon OID {poly_oid}: buffer has zero area – skipped."
                    )
                    continue
                insert_cur.insertRow([buf_shape, poly_oid, buffer_ratio, w_min, w_max, w_mean])
                n_success += 1
            except Exception as exc:
                n_failed += 1
                arcpy.AddWarning(f"Polygon OID {poly_oid}: {exc}")

    # ── Step 4: Clean up temporary datasets ──────────────────────────────────
    log("Step 4/4  Cleaning up …")
    try:
        arcpy.Delete_management(joined_cl)
    except Exception:
        pass

    log(
        f"Done.  {n_success} buffer feature(s) written to: {out_buffer}"
        + (f"  ({n_failed} failed – see warnings)" if n_failed else "")
    )
