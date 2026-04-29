# -*- coding: utf-8 -*-
"""
proportional_buffer.py
======================
Variable-width proportional buffer along pre-existing centerlines.

This module is **completely independent** of any centerline-extraction code.

Inputs
------
* polygon layer  – waterway / channel boundary polygons.
* centerline layer – pre-computed centerlines (produced by any tool, or
                     supplied manually).

Output
------
* Buffer polygons whose cross-sectional width at every point equals
  ``buffer_ratio × 2 × local_half_width``, where ``local_half_width``
  is the distance from the centerline sample point to the nearest polygon
  boundary vertex (estimated via a KDTree on the densified boundary).

Algorithm overview
------------------
1. Validate inputs; optionally re-project the centerline to match the
   polygon CRS; warn if the CRS is geographic.
2. For each polygon / centerline pair:

   a. *Sample* the centerline uniformly at ``sample_distance`` intervals.
   b. *Build a KDTree* from densified polygon-boundary points; query to
      obtain the local half-width ``hw_i`` at every sample point ``P_i``.
   c. *Compute normal vectors* at each sample point (central differences
      for interior points; one-sided differences at the endpoints).
   d. *Offset* left ``L_i = P_i + r_i * n_i`` and right
      ``R_i = P_i – r_i * n_i``, where ``r_i = buffer_ratio * hw_i``
      (clamped by ``width_min`` / ``width_max``).
   e. *Assemble* a closed polygon ring: forward left side → front end-cap
      → reverse right side → back end-cap.
   f. Optionally *clip* to the original polygon and *smooth* with the
      Chaikin corner-cutting algorithm.

3. Write results with width-statistic attributes to a GeoDataFrame.

Dependencies
------------
    numpy    >= 1.24
    scipy    >= 1.10     (cKDTree for fast boundary-distance queries)
    shapely  >= 2.0
    geopandas >= 0.13    (only for process_geodataframes())

Usage (standalone)
------------------
::

    from proportional_buffer import compute_proportional_buffer
    from shapely.geometry import Polygon, LineString

    poly = Polygon([(-50,-5),(50,-5),(50,5),(-50,5)])
    cl   = LineString([(-50, 0), (50, 0)])
    buf  = compute_proportional_buffer(poly, cl, buffer_ratio=0.5)

Usage (GeoDataFrames)
---------------------
::

    import geopandas as gpd
    from proportional_buffer import process_geodataframes

    poly_gdf = gpd.read_file("waterways.gpkg", layer="polygons")
    cl_gdf   = gpd.read_file("waterways.gpkg", layer="centerlines")
    out_gdf  = process_geodataframes(poly_gdf, cl_gdf, buffer_ratio=0.5)
    out_gdf.to_file("buffers.gpkg", layer="buffers")
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree

try:
    from shapely.geometry import (
        LineString,
        MultiLineString,
        MultiPolygon,
        Point,
        Polygon,
    )
    from shapely.ops import unary_union

    _HAS_SHAPELY = True
except ImportError:  # pragma: no cover
    _HAS_SHAPELY = False

# ---------------------------------------------------------------------------
# Internal type aliases
# ---------------------------------------------------------------------------

_Coords2D = np.ndarray  # shape (N, 2), float64


# ---------------------------------------------------------------------------
# Guard helper
# ---------------------------------------------------------------------------


def _require_shapely() -> None:
    if not _HAS_SHAPELY:
        raise ImportError(
            "shapely >= 2.0 is required.  Install it with:\n"
            "    pip install shapely\n"
            "or (conda):\n"
            "    conda install -c conda-forge shapely"
        )


# ---------------------------------------------------------------------------
# Low-level geometry helpers (pure numpy, no shapely)
# ---------------------------------------------------------------------------


def _densify_coords(coords: _Coords2D, max_spacing: float) -> _Coords2D:
    """
    Resample a coordinate sequence so that consecutive points are at most
    *max_spacing* apart.

    Parameters
    ----------
    coords :
        ``(N, 2)`` array of (x, y) coordinates.
    max_spacing :
        Maximum allowed distance between consecutive output points.

    Returns
    -------
    numpy.ndarray
        ``(M, 2)`` resampled coordinate array (``M >= N``).
    """
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


def _compute_normals(sample_pts: _Coords2D) -> _Coords2D:
    """
    Compute unit left-side normal vectors at each sample point.

    Uses central differences for interior points and one-sided differences
    at the two endpoints.

    Parameters
    ----------
    sample_pts :
        ``(N, 2)`` array of ordered sample points along the centerline.

    Returns
    -------
    numpy.ndarray
        ``(N, 2)`` unit normal vectors pointing to the *left* of the
        travel direction.
    """
    n = len(sample_pts)
    tangents = np.zeros((n, 2), dtype=float)

    # Central differences for interior points
    tangents[1:-1] = sample_pts[2:] - sample_pts[:-2]
    # One-sided differences at the endpoints
    if n >= 2:
        tangents[0] = sample_pts[1] - sample_pts[0]
        tangents[-1] = sample_pts[-1] - sample_pts[-2]

    # Normalise (guard against zero-length segments)
    lengths = np.hypot(tangents[:, 0], tangents[:, 1])
    lengths = np.where(lengths < 1e-12, 1.0, lengths)
    tangents /= lengths[:, np.newaxis]

    # Rotate 90° CCW: (dx, dy) → (–dy, dx)  [left-hand normal]
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    return normals


def _arc_points(
    center: Tuple[float, float],
    radius: float,
    angle_start: float,
    angle_end: float,
    n_pts: int,
) -> _Coords2D:
    """
    Generate *n_pts* points along a circular arc.

    Parameters
    ----------
    center :
        ``(cx, cy)`` centre of the arc.
    radius :
        Arc radius.
    angle_start, angle_end :
        Start and end angles in **radians**.  The arc sweeps linearly
        from *angle_start* to *angle_end* (CW when end < start).
    n_pts :
        Number of output points (including endpoints).

    Returns
    -------
    numpy.ndarray
        ``(n_pts, 2)`` array of arc coordinates.
    """
    angles = np.linspace(angle_start, angle_end, n_pts)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack([x, y])


def chaikin_smooth(
    coords: Union[List[Tuple[float, float]], _Coords2D],
    iterations: int = 2,
) -> _Coords2D:
    """
    Smooth a closed polygon ring using Chaikin's corner-cutting algorithm.

    Each iteration replaces every edge with two new points at the ¼ and ¾
    positions, effectively cutting corners and producing a smoother curve.

    Parameters
    ----------
    coords :
        Sequence of ``(x, y)`` coordinates.  The ring is treated as closed
        (first point equals last point is handled automatically).
    iterations :
        Number of smoothing passes.  More passes → smoother but shorter ring.

    Returns
    -------
    numpy.ndarray
        ``(M, 2)`` smoothed ring coordinates (closed: first == last).
    """
    pts = np.asarray(coords, dtype=float)

    # Ensure ring is closed for processing
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    for _ in range(max(0, iterations)):
        n = len(pts) - 1  # exclude the duplicated closing point
        new_pts = np.empty((2 * n, 2), dtype=float)
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            new_pts[2 * i] = 0.75 * p0 + 0.25 * p1
            new_pts[2 * i + 1] = 0.25 * p0 + 0.75 * p1
        pts = np.vstack([new_pts, new_pts[0]])  # re-close

    return pts


# ---------------------------------------------------------------------------
# Shapely-based helpers (require shapely)
# ---------------------------------------------------------------------------


def _ring_to_arrays(polygon: "Polygon") -> List[_Coords2D]:
    """
    Return the exterior ring and all interior rings of *polygon* as a list
    of ``(N_i, 2)`` numpy arrays.
    """
    rings = [np.asarray(polygon.exterior.coords, dtype=float)]
    for interior in polygon.interiors:
        rings.append(np.asarray(interior.coords, dtype=float))
    return rings


def _build_boundary_tree(polygon: "Polygon", densify_spacing: float) -> cKDTree:
    """
    Build a ``scipy.spatial.cKDTree`` from densified polygon boundary points.

    Densifying the boundary before building the tree ensures that the
    nearest-vertex distance is a close approximation of the true
    nearest-point-on-edge distance.

    Parameters
    ----------
    polygon :
        Input shapely Polygon.
    densify_spacing :
        Maximum spacing between consecutive boundary sample points.
        Should be ≤ ``sample_distance / 2`` for accurate results.

    Returns
    -------
    scipy.spatial.cKDTree
    """
    ring_arrays = _ring_to_arrays(polygon)
    all_pts: List[_Coords2D] = []
    for ring in ring_arrays:
        all_pts.append(_densify_coords(ring, densify_spacing))
    boundary_pts = np.vstack(all_pts)
    return cKDTree(boundary_pts)


def _auto_sample_distance(polygon: "Polygon") -> float:
    """
    Estimate a sensible centerline sample distance from the polygon geometry.

    Uses the hydraulic-radius approximation  r ≈ area / perimeter  as a
    proxy for the minimum half-width, then divides by 10 to get a resolution
    that produces ~20 samples across the narrowest cross-section.

    Returns
    -------
    float
        Recommended sample distance (same units as the polygon CRS).
    """
    area = polygon.area
    perimeter = polygon.exterior.length
    if perimeter <= 0:
        return 1.0
    min_hw = area / perimeter  # approximate minimum half-width
    return max(min_hw / 10.0, 1e-6)


def _sample_centerline(
    centerline: Union["LineString", "MultiLineString"],
    sample_distance: float,
) -> List[_Coords2D]:
    """
    Uniformly resample a centerline at *sample_distance* intervals.

    Parameters
    ----------
    centerline :
        A shapely LineString or MultiLineString.
    sample_distance :
        Sampling interval (same units as the geometry CRS).

    Returns
    -------
    list of numpy.ndarray
        One ``(N_i, 2)`` array per LineString component.
    """
    _require_shapely()
    geoms = list(centerline.geoms) if isinstance(centerline, MultiLineString) else [centerline]

    result = []
    for g in geoms:
        coords = np.asarray(g.coords, dtype=float)[:, :2]  # drop Z if present
        sampled = _densify_coords(coords, sample_distance)
        if len(sampled) >= 2:
            result.append(sampled)
    return result


def _compute_half_widths(
    sample_pts: _Coords2D,
    boundary_tree: cKDTree,
) -> np.ndarray:
    """
    Compute the local half-width at each sample point.

    The half-width is approximated as the distance from the sample point
    to the nearest point in the densified boundary KDTree.

    Parameters
    ----------
    sample_pts :
        ``(N, 2)`` array of sample points.
    boundary_tree :
        Pre-built cKDTree (see :func:`_build_boundary_tree`).

    Returns
    -------
    numpy.ndarray
        ``(N,)`` array of half-width values.
    """
    dists, _ = boundary_tree.query(sample_pts)
    return dists


def _build_buffer_ring(
    sample_pts: _Coords2D,
    half_widths: np.ndarray,
    normals: _Coords2D,
    buffer_ratio: float,
    width_min: float,
    width_max: float,
    end_cap: str,
    n_cap_pts: int,
) -> _Coords2D:
    """
    Construct the closed coordinate ring of the variable-width buffer polygon.

    The ring is assembled as:
    forward left side  → front end-cap (L_n → R_n)
    → backward right side  → back end-cap (R_0 → L_0)  → close.

    Parameters
    ----------
    sample_pts :
        ``(N, 2)`` ordered sample points along the centerline.
    half_widths :
        ``(N,)`` local half-width at each sample point.
    normals :
        ``(N, 2)`` left-side unit normal vectors.
    buffer_ratio :
        Fraction of the half-width to use as buffer radius.
    width_min :
        Minimum total buffer width (half applied to each side).
    width_max :
        Maximum total buffer width (half applied to each side).
    end_cap :
        ``'round'`` or ``'flat'``.
    n_cap_pts :
        Number of arc interpolation points for round end caps.

    Returns
    -------
    numpy.ndarray
        ``(M, 2)`` closed ring coordinates (first == last).
    """
    # Per-point radii (half the total buffer width), clamped
    radii = np.clip(
        half_widths * buffer_ratio,
        width_min / 2.0,
        width_max / 2.0,
    )

    left_pts = sample_pts + radii[:, np.newaxis] * normals
    right_pts = sample_pts - radii[:, np.newaxis] * normals

    ring: List[List[float]] = []

    # ── Forward pass: left side ─────────────────────────────────────────────
    ring.extend(left_pts.tolist())

    # ── Front end-cap at P_n (L_n → R_n sweeping through forward direction) ─
    if end_cap == "round":
        nrm = normals[-1]
        angle_L = math.atan2(nrm[1], nrm[0])
        # Sweep CW (angle_L → angle_L − π) through the forward direction
        arc = _arc_points(sample_pts[-1], radii[-1], angle_L, angle_L - math.pi, n_cap_pts)
        ring.extend(arc.tolist())
    else:
        ring.append(right_pts[-1].tolist())

    # ── Backward pass: right side ────────────────────────────────────────────
    ring.extend(right_pts[::-1].tolist())

    # ── Back end-cap at P_0 (R_0 → L_0 sweeping through backward direction) ─
    if end_cap == "round":
        nrm = normals[0]
        angle_R = math.atan2(-nrm[1], -nrm[0])
        # Sweep CW (angle_R → angle_R − π) through the backward direction
        arc = _arc_points(sample_pts[0], radii[0], angle_R, angle_R - math.pi, n_cap_pts)
        ring.extend(arc.tolist())
    else:
        ring.append(left_pts[0].tolist())

    # Close the ring
    ring.append(ring[0])

    return np.array(ring, dtype=float)


# ---------------------------------------------------------------------------
# Public API – single feature
# ---------------------------------------------------------------------------


def compute_proportional_buffer(
    polygon: "Polygon",
    centerline: Union["LineString", "MultiLineString"],
    buffer_ratio: float = 0.5,
    sample_distance: Optional[float] = None,
    end_cap: str = "round",
    clip_to_polygon: bool = True,
    smooth_tolerance: float = 0.0,
    width_min: float = 0.0,
    width_max: float = float("inf"),
    n_cap_pts: int = 8,
) -> "Union[Polygon, MultiPolygon]":
    """
    Compute a variable-width proportional buffer along a centerline.

    The buffer width at each cross-section equals
    ``buffer_ratio × 2 × local_half_width``, where the local half-width is
    the distance from the centerline sample point to the nearest polygon
    boundary point.

    Parameters
    ----------
    polygon :
        The waterway / channel boundary polygon.  If a MultiPolygon is
        supplied, the largest component is used.
    centerline :
        Pre-computed centerline of the polygon.  A MultiLineString is
        processed component-by-component and the results are unioned.
    buffer_ratio :
        Fraction of the local half-width used as the buffer radius on each
        side (0 < ratio ≤ 1).  Default 0.5 → buffer width = local channel
        width × 0.5.
    sample_distance :
        Spacing between centerline sample points (in the CRS units).
        ``None`` → automatic (≈ min half-width / 10).
    end_cap :
        Style of the end-caps at the centerline termini.
        ``'round'`` (default) adds a semicircular cap; ``'flat'`` closes
        the buffer with a straight edge.
    clip_to_polygon :
        When ``True`` (default) the output buffer is clipped to the
        original polygon boundary so it cannot extend outside the channel.
    smooth_tolerance :
        Number of Chaikin smoothing iterations to apply to the output ring
        (0 = no smoothing).  Fractional values are rounded.
    width_min :
        Minimum total buffer width (metres or CRS units).  Prevents
        degenerate slivers near the polygon tips.
    width_max :
        Maximum total buffer width.  Caps the buffer at very wide sections.
    n_cap_pts :
        Number of interpolation points in each round end-cap arc.

    Returns
    -------
    shapely.geometry.Polygon or MultiPolygon
        The variable-width buffer geometry.

    Raises
    ------
    ImportError
        If shapely is not installed.
    ValueError
        If the input polygon or centerline is empty, or if no valid buffer
        segments can be constructed.
    """
    _require_shapely()

    if polygon is None or polygon.is_empty:
        raise ValueError("Input polygon is empty.")
    if centerline is None or centerline.is_empty:
        raise ValueError("Input centerline is empty.")

    # Normalise MultiPolygon → largest component
    if isinstance(polygon, MultiPolygon):
        polygon = max(polygon.geoms, key=lambda g: g.area)

    # Determine sample distance
    if sample_distance is None or sample_distance <= 0:
        sample_distance = _auto_sample_distance(polygon)

    # Build boundary KDTree (densified at half the sample distance for accuracy)
    boundary_tree = _build_boundary_tree(polygon, sample_distance / 2.0)

    # Sample all centerline components
    segments = _sample_centerline(centerline, sample_distance)
    if not segments:
        raise ValueError(
            "Centerline is too short to sample at the given distance "
            f"({sample_distance}).  Use a smaller sample_distance."
        )

    smooth_iters = max(0, int(round(smooth_tolerance)))

    buffer_parts: List["Union[Polygon, MultiPolygon]"] = []
    for seg_pts in segments:
        if len(seg_pts) < 2:
            continue

        half_widths = _compute_half_widths(seg_pts, boundary_tree)
        normals = _compute_normals(seg_pts)

        ring = _build_buffer_ring(
            seg_pts,
            half_widths,
            normals,
            buffer_ratio,
            width_min,
            width_max,
            end_cap,
            n_cap_pts,
        )

        if smooth_iters > 0:
            ring = chaikin_smooth(ring, iterations=smooth_iters)

        try:
            buf_poly = Polygon(ring).buffer(0)  # repair self-intersections
        except Exception:
            try:
                buf_poly = Polygon(ring).convex_hull  # last-resort fallback
            except Exception:
                continue

        if not buf_poly.is_empty:
            buffer_parts.append(buf_poly)

    if not buffer_parts:
        raise ValueError(
            "No valid buffer segments could be generated.  Check that the "
            "centerline lies inside the polygon and that sample_distance is "
            "smaller than the polygon's narrowest section."
        )

    result = unary_union(buffer_parts)

    if clip_to_polygon:
        result = result.intersection(polygon)

    return result


# ---------------------------------------------------------------------------
# Public API – multiple features (GeoDataFrame interface)
# ---------------------------------------------------------------------------


def process_geodataframes(
    poly_gdf: "gpd.GeoDataFrame",
    cl_gdf: "gpd.GeoDataFrame",
    poly_id_field: Optional[str] = None,
    cl_id_field: Optional[str] = None,
    buffer_ratio: float = 0.5,
    sample_distance: Optional[float] = None,
    end_cap: str = "round",
    clip_to_polygon: bool = True,
    smooth_tolerance: float = 0.0,
    width_min: float = 0.0,
    width_max: float = float("inf"),
    n_cap_pts: int = 8,
) -> "gpd.GeoDataFrame":
    """
    Process multiple polygon / centerline feature pairs and return a
    GeoDataFrame of variable-width buffer polygons.

    Centerlines are spatially matched to polygons with an *intersects*
    predicate.  Multiple centerlines that intersect the same polygon are
    unioned into a single MultiLineString before processing, so a branching
    waterway is handled correctly.

    Parameters
    ----------
    poly_gdf :
        Polygon layer (channel / waterway boundaries).
    cl_gdf :
        Centerline layer.  Must be in the same CRS as *poly_gdf* or
        compatible (automatic reprojection is attempted).
    poly_id_field :
        Attribute field in *poly_gdf* to use as the ``src_id`` column in
        the output.  When ``None`` the GeoDataFrame index is used.
    cl_id_field :
        Attribute field in *cl_gdf* used only for diagnostic messages.
    buffer_ratio :
        See :func:`compute_proportional_buffer`.
    sample_distance :
        See :func:`compute_proportional_buffer`.
    end_cap :
        See :func:`compute_proportional_buffer`.
    clip_to_polygon :
        See :func:`compute_proportional_buffer`.
    smooth_tolerance :
        See :func:`compute_proportional_buffer`.
    width_min :
        See :func:`compute_proportional_buffer`.
    width_max :
        See :func:`compute_proportional_buffer`.
    n_cap_pts :
        See :func:`compute_proportional_buffer`.

    Returns
    -------
    geopandas.GeoDataFrame
        Output buffer polygons with columns:

        ``src_id``
            Identifier from *poly_id_field* (or the polygon index).
        ``buffer_ratio``
            The ratio used for this feature.
        ``width_min_m``
            Minimum channel width sampled along the centerline (CRS units).
        ``width_max_m``
            Maximum channel width.
        ``width_mean_m``
            Mean channel width.
        ``geometry``
            Buffer polygon geometry.
    """
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError(
            "geopandas is required.  Install it with:\n"
            "    pip install geopandas"
        )

    _require_shapely()

    _EMPTY_COLS = [
        "src_id",
        "buffer_ratio",
        "width_min_m",
        "width_max_m",
        "width_mean_m",
        "geometry",
    ]

    # ── CRS harmonisation ───────────────────────────────────────────────────
    if poly_gdf.crs is not None and cl_gdf.crs is not None:
        if poly_gdf.crs != cl_gdf.crs:
            cl_gdf = cl_gdf.to_crs(poly_gdf.crs)

    if poly_gdf.crs is not None and poly_gdf.crs.is_geographic:
        warnings.warn(
            "Input CRS appears to be geographic (degrees).  "
            "Buffer distances will be in degrees, not metres.  "
            "Consider reprojecting to a projected CRS before calling "
            "this function.",
            UserWarning,
            stacklevel=2,
        )

    # ── Spatial join: match each centerline to its polygon ──────────────────
    joined = gpd.sjoin(cl_gdf, poly_gdf, how="inner", predicate="intersects")

    if len(joined) == 0:
        warnings.warn(
            "No centerlines intersect any polygon.  "
            "Returning an empty GeoDataFrame.",
            UserWarning,
            stacklevel=2,
        )
        return gpd.GeoDataFrame(columns=_EMPTY_COLS, crs=poly_gdf.crs)

    poly_idx_col = "index_right"
    results = []

    for poly_idx, group in joined.groupby(poly_idx_col):
        poly_row = poly_gdf.loc[poly_idx]
        polygon = poly_row.geometry

        src_id = (
            poly_row[poly_id_field]
            if poly_id_field and poly_id_field in poly_gdf.columns
            else poly_idx
        )

        # Collect all centerlines matched to this polygon
        cl_geoms = [g for g in group.geometry.tolist() if g is not None and not g.is_empty]
        if not cl_geoms:
            continue

        # Merge into a single geometry for uniform processing
        if len(cl_geoms) == 1:
            merged_cl = cl_geoms[0]
        else:
            parts: List["LineString"] = []
            for g in cl_geoms:
                if isinstance(g, MultiLineString):
                    parts.extend(g.geoms)
                else:
                    parts.append(g)
            merged_cl = MultiLineString(parts)

        try:
            buf = compute_proportional_buffer(
                polygon=polygon,
                centerline=merged_cl,
                buffer_ratio=buffer_ratio,
                sample_distance=sample_distance,
                end_cap=end_cap,
                clip_to_polygon=clip_to_polygon,
                smooth_tolerance=smooth_tolerance,
                width_min=width_min,
                width_max=width_max,
                n_cap_pts=n_cap_pts,
            )
        except Exception as exc:
            warnings.warn(
                f"Failed to compute buffer for polygon index {poly_idx}: {exc}",
                UserWarning,
                stacklevel=2,
            )
            continue

        # ── Width statistics ──────────────────────────────────────────────
        sd = sample_distance or _auto_sample_distance(polygon)
        bt = _build_boundary_tree(polygon, sd / 2.0)
        all_hw: List[float] = []
        for seg in _sample_centerline(merged_cl, sd):
            all_hw.extend(_compute_half_widths(seg, bt).tolist())

        if all_hw:
            hw_arr = np.array(all_hw)
            w_min = float(2.0 * np.min(hw_arr))
            w_max = float(2.0 * np.max(hw_arr))
            w_mean = float(2.0 * np.mean(hw_arr))
        else:
            w_min = w_max = w_mean = 0.0

        results.append(
            {
                "src_id": src_id,
                "buffer_ratio": buffer_ratio,
                "width_min_m": w_min,
                "width_max_m": w_max,
                "width_mean_m": w_mean,
                "geometry": buf,
            }
        )

    if not results:
        return gpd.GeoDataFrame(columns=_EMPTY_COLS, crs=poly_gdf.crs)

    return gpd.GeoDataFrame(results, crs=poly_gdf.crs)
