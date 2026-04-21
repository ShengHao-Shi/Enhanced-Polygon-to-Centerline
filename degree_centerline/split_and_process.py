# -*- coding: utf-8 -*-
"""
split_and_process.py
====================
Three-phase tiled centerline extraction for large or complex polygon datasets.

Motivation
----------
Running ``degree_centerline`` on a country-scale polygon dataset in one pass
leads to two failure modes:

  1. Long, narrow waterways are dropped by the pruning filter because the global
     reference length is dominated by a few very long main-channel segments.
  2. In topographically complex regions the densified edge-point count becomes
     insufficient for a reliable Voronoi skeleton.

Strategy (three phases)
-----------------------
Phase A – Connected-component split
    A MULTIPOLYGON is already decomposed into independent polygon parts by the
    existing ``_parse_wkt_polygon`` parser in ``centerline_degree.py``.  Each
    part is processed separately so that unconnected water bodies cannot
    interfere with each other's skeleton graph or pruning reference lengths.

Phase B – Adaptive quad-tree subdivision
    Each polygon part is tested against two complexity thresholds:
      • ``max_vertices`` — total boundary vertex count (exterior + all holes).
      • ``max_bbox_area`` — bounding-box area (optional; useful when CRS units
        are known, e.g. square metres).
    Polygons that exceed any active threshold are split into four equal
    quadrants via Sutherland-Hodgman polygon clipping.  Each quadrant is then
    tested recursively until all tiles are below the thresholds or the maximum
    recursion depth is reached.

Phase C – Overlap-buffer extraction
    Before clipping the source polygon to a grid cell the cell bounding box is
    expanded outward by ``buffer`` units (``buffer = buffer_factor ×
    densify_distance``).  The extra ring of source geometry prevents the Voronoi
    diagram from treating the cut boundary as a real polygon edge, which would
    otherwise produce spurious short branches at every tile seam.  After the
    centerline is extracted from the buffered tile the result is clipped back
    to the original (non-buffered) grid cell so that no duplicate segments
    appear in the merged output.

Public API
----------
::

    from split_and_process import tile_and_extract_centerline

    result_wkt = tile_and_extract_centerline(
        wkt,                        # WKT POLYGON / MULTIPOLYGON string
        method="voronoi",           # passed through to polygon_to_centerline_wkt
        densify_distance=1.0,
        prune_threshold=0.0,
        smooth_sigma=0.0,
        raster_resolution=None,
        single_line=True,
        # --- Tiling parameters ---
        max_vertices=8000,          # Phase B vertex threshold
        max_bbox_area=None,         # Phase B area threshold (None = disabled)
        buffer_factor=5.0,          # Phase C buffer = factor × densify_distance
        max_depth=5,                # maximum quad-tree recursion depth
        progress_callback=None,     # optional callable(message, percentage)
    )
    # Returns WKT MULTILINESTRING, or None if nothing was extracted.

No dependencies beyond those already required by ``centerline_degree.py``
(numpy, scipy, networkx).  No shapely, geopandas, or pandas required.
"""

from __future__ import annotations

import math
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Module-level constants / defaults
# ---------------------------------------------------------------------------

#: Default maximum number of boundary vertices (exterior + holes) per tile.
#: Tiles exceeding this value are subdivided further (Phase B).
_DEFAULT_MAX_VERTICES: int = 8_000

#: Default buffer multiplier.  The overlap buffer added around each tile is
#: ``buffer_factor × densify_distance`` CRS units (Phase C).
_DEFAULT_BUFFER_FACTOR: float = 5.0

#: Default maximum quad-tree recursion depth.
#: At depth *d* a polygon can be split into at most 4^d tiles.
_DEFAULT_MAX_DEPTH: int = 5

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

_Pt = Tuple[float, float]               # (x, y) point
_Ring = List[_Pt]                       # open ring (last ≠ first)
_BBox = Tuple[float, float, float, float]  # (minx, miny, maxx, maxy)


# ---------------------------------------------------------------------------
# Sutherland-Hodgman polygon clipping to an axis-aligned rectangle
# ---------------------------------------------------------------------------


def _sh_clip(ring_open: _Ring, bbox: _BBox) -> _Ring:
    """Clip an open polygon ring to an axis-aligned bounding box.

    Uses the Sutherland-Hodgman algorithm with four clip planes
    (LEFT x ≥ minx, RIGHT x ≤ maxx, BOTTOM y ≥ miny, TOP y ≤ maxy).

    The algorithm preserves concave polygons but, for multiply-concave shapes
    that straddle a clip edge, may produce a single merged ring instead of two
    separate components.  For the purpose of complexity-based tiling and
    centerline extraction this behaviour is acceptable.

    Parameters
    ----------
    ring_open:
        Open polygon ring — last vertex must NOT be a repeat of the first.
    bbox:
        ``(minx, miny, maxx, maxy)`` clip rectangle.

    Returns
    -------
    Open clipped ring, or ``[]`` if the ring is fully outside *bbox*.
    """
    if not ring_open:
        return []

    minx, miny, maxx, maxy = bbox

    # ── intersection helpers ────────────────────────────────────────────────
    def _intersect_left(p1: _Pt, p2: _Pt) -> _Pt:
        dx = p2[0] - p1[0]
        t = (minx - p1[0]) / dx if dx else 0.0
        return (minx, p1[1] + t * (p2[1] - p1[1]))

    def _intersect_right(p1: _Pt, p2: _Pt) -> _Pt:
        dx = p2[0] - p1[0]
        t = (maxx - p1[0]) / dx if dx else 0.0
        return (maxx, p1[1] + t * (p2[1] - p1[1]))

    def _intersect_bottom(p1: _Pt, p2: _Pt) -> _Pt:
        dy = p2[1] - p1[1]
        t = (miny - p1[1]) / dy if dy else 0.0
        return (p1[0] + t * (p2[0] - p1[0]), miny)

    def _intersect_top(p1: _Pt, p2: _Pt) -> _Pt:
        dy = p2[1] - p1[1]
        t = (maxy - p1[1]) / dy if dy else 0.0
        return (p1[0] + t * (p2[0] - p1[0]), maxy)

    # Each entry: (inside_test, intersect_fn)
    planes = [
        (lambda p: p[0] >= minx, _intersect_left),
        (lambda p: p[0] <= maxx, _intersect_right),
        (lambda p: p[1] >= miny, _intersect_bottom),
        (lambda p: p[1] <= maxy, _intersect_top),
    ]

    output: _Ring = list(ring_open)
    for inside_fn, intersect_fn in planes:
        if not output:
            return []
        inp = output
        output = []
        n = len(inp)
        for i in range(n):
            curr = inp[i]
            prev = inp[i - 1]   # wraps at i == 0
            c_in = inside_fn(curr)
            p_in = inside_fn(prev)
            if c_in:
                if not p_in:
                    output.append(intersect_fn(prev, curr))
                output.append(curr)
            elif p_in:
                output.append(intersect_fn(prev, curr))

    return output


# ---------------------------------------------------------------------------
# Cohen-Sutherland line segment clipping to an axis-aligned rectangle
# ---------------------------------------------------------------------------

_CS_INSIDE = 0
_CS_LEFT   = 1
_CS_RIGHT  = 2
_CS_BOTTOM = 4
_CS_TOP    = 8


def _outcode(x: float, y: float, minx: float, miny: float,
             maxx: float, maxy: float) -> int:
    code = _CS_INSIDE
    if x < minx:
        code |= _CS_LEFT
    elif x > maxx:
        code |= _CS_RIGHT
    if y < miny:
        code |= _CS_BOTTOM
    elif y > maxy:
        code |= _CS_TOP
    return code


def _clip_segment(
    x0: float, y0: float, x1: float, y1: float, bbox: _BBox
) -> Optional[Tuple[_Pt, _Pt]]:
    """Cohen-Sutherland line-segment clipping to an axis-aligned rectangle.

    Parameters
    ----------
    x0, y0, x1, y1:
        Segment endpoints.
    bbox:
        ``(minx, miny, maxx, maxy)`` clip rectangle.

    Returns
    -------
    ``((x0', y0'), (x1', y1'))`` — the clipped segment, or ``None`` if the
    segment lies entirely outside *bbox*.
    """
    minx, miny, maxx, maxy = bbox
    code0 = _outcode(x0, y0, minx, miny, maxx, maxy)
    code1 = _outcode(x1, y1, minx, miny, maxx, maxy)

    while True:
        if not (code0 | code1):          # trivially inside
            return (x0, y0), (x1, y1)
        if code0 & code1:               # trivially outside (same region)
            return None

        # Pick an outside endpoint to clip
        code_out = code1 if code0 == _CS_INSIDE else code0
        dx = x1 - x0
        dy = y1 - y0

        if code_out & _CS_TOP:
            xi = x0 + dx * (maxy - y0) / dy if dy else x0
            yi = maxy
        elif code_out & _CS_BOTTOM:
            xi = x0 + dx * (miny - y0) / dy if dy else x0
            yi = miny
        elif code_out & _CS_RIGHT:
            yi = y0 + dy * (maxx - x0) / dx if dx else y0
            xi = maxx
        else:  # _CS_LEFT
            yi = y0 + dy * (minx - x0) / dx if dx else y0
            xi = minx

        if code_out == code0:
            x0, y0 = xi, yi
            code0 = _outcode(x0, y0, minx, miny, maxx, maxy)
        else:
            x1, y1 = xi, yi
            code1 = _outcode(x1, y1, minx, miny, maxx, maxy)


# ---------------------------------------------------------------------------
# Polygon complexity metrics
# ---------------------------------------------------------------------------


def _ring_area(ring_open: _Ring) -> float:
    """Absolute area of an open polygon ring (shoelace formula)."""
    n = len(ring_open)
    if n < 3:
        return 0.0
    acc = 0.0
    for i in range(n):
        x1, y1 = ring_open[i]
        x2, y2 = ring_open[(i + 1) % n]
        acc += x1 * y2 - x2 * y1
    return abs(acc) * 0.5


def _ring_perimeter(ring_open: _Ring) -> float:
    """Perimeter of an open polygon ring."""
    n = len(ring_open)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        dx = ring_open[(i + 1) % n][0] - ring_open[i][0]
        dy = ring_open[(i + 1) % n][1] - ring_open[i][1]
        total += math.hypot(dx, dy)
    return total


def _complexity_metrics(
    ext_open: _Ring, holes_open: List[_Ring]
) -> dict:
    """Compute basic complexity metrics for one polygon tile.

    Returns a dict with keys:

    * ``n_vertices``  — total boundary vertex count (exterior + all holes).
    * ``bbox_area``   — area of the axis-aligned bounding box.
    * ``poly_area``   — approximate polygon area (exterior minus holes).
    """
    n_v = len(ext_open) + sum(len(h) for h in holes_open)
    if not ext_open:
        return {"n_vertices": 0, "bbox_area": 0.0, "poly_area": 0.0}

    xs = [p[0] for p in ext_open]
    ys = [p[1] for p in ext_open]
    bbox_area = (max(xs) - min(xs)) * (max(ys) - min(ys))

    poly_area = _ring_area(ext_open) - sum(_ring_area(h) for h in holes_open)
    poly_area = max(poly_area, 0.0)

    return {
        "n_vertices": n_v,
        "bbox_area": bbox_area,
        "poly_area": poly_area,
    }


# ---------------------------------------------------------------------------
# Adaptive quad-tree tiling  (Phase B + Phase C geometry)
# ---------------------------------------------------------------------------


def _adaptive_tiles(
    ext_open: _Ring,
    holes_open: List[_Ring],
    clip_bbox: _BBox,
    max_vertices: int,
    max_bbox_area: Optional[float],
    buffer: float,
    max_depth: int,
    depth: int = 0,
) -> List[Tuple[_Ring, List[_Ring], _BBox]]:
    """Recursively split a polygon tile into sub-tiles.

    Parameters
    ----------
    ext_open:
        Open exterior ring of the current tile polygon (already clipped to a
        buffered quadrant at the previous level).
    holes_open:
        Open hole rings of the current tile polygon.
    clip_bbox:
        The **original (non-buffered)** bounding box of this tile.  Used to
        clip the extracted centerline back after processing (Phase C).
    max_vertices:
        Phase B trigger: vertex count threshold.
    max_bbox_area:
        Phase B trigger: bounding-box area threshold (``None`` = disabled).
    buffer:
        Phase C overlap buffer to add around each quadrant when clipping the
        source polygon.
    max_depth:
        Maximum recursion depth.
    depth:
        Current recursion depth (internal use).

    Returns
    -------
    List of ``(ext_open, holes_open, clip_bbox)`` tuples — one per leaf tile.
    Each tile polygon has been clipped to the **buffered** quadrant boundary
    and its ``clip_bbox`` is the **non-buffered** quadrant boundary.
    """
    if not ext_open or len(ext_open) < 3:
        return []

    metrics = _complexity_metrics(ext_open, holes_open)

    # Check whether this tile is simple enough to process directly
    too_complex = metrics["n_vertices"] > max_vertices
    if (not too_complex and max_bbox_area is not None
            and metrics["bbox_area"] > max_bbox_area):
        too_complex = True

    if not too_complex or depth >= max_depth:
        return [(ext_open, holes_open, clip_bbox)]

    # ── Subdivide into four quadrants ───────────────────────────────────────
    minx, miny, maxx, maxy = clip_bbox
    cx = (minx + maxx) * 0.5
    cy = (miny + maxy) * 0.5

    quadrants: List[_BBox] = [
        (minx, miny, cx,   cy),    # SW
        (cx,   miny, maxx, cy),    # SE
        (minx, cy,   cx,   maxy),  # NW
        (cx,   cy,   maxx, maxy),  # NE
    ]

    result: List[Tuple[_Ring, List[_Ring], _BBox]] = []

    for q_bbox in quadrants:
        qminx, qminy, qmaxx, qmaxy = q_bbox

        # Expand by buffer for Phase C overlap
        buf_bbox: _BBox = (
            qminx - buffer, qminy - buffer,
            qmaxx + buffer, qmaxy + buffer,
        )

        # Clip polygon to buffered quadrant
        q_ext = _sh_clip(ext_open, buf_bbox)
        if len(q_ext) < 3:
            continue

        q_holes = [_sh_clip(h, buf_bbox) for h in holes_open]
        q_holes = [h for h in q_holes if len(h) >= 3]

        # Recurse into the quadrant
        sub = _adaptive_tiles(
            q_ext, q_holes, q_bbox,
            max_vertices, max_bbox_area, buffer, max_depth, depth + 1,
        )
        result.extend(sub)

    return result


# ---------------------------------------------------------------------------
# WKT helpers
# ---------------------------------------------------------------------------


def _rings_to_polygon_wkt(ext_open: _Ring, holes_open: List[_Ring]) -> str:
    """Serialise open rings to a WKT ``POLYGON`` string."""

    def _ring_str(ring: _Ring) -> str:
        # Close the ring: repeat the first vertex at the end
        pts = ring + [ring[0]]
        return "({})".format(
            ", ".join("{} {}".format(x, y) for x, y in pts)
        )

    parts = [_ring_str(ext_open)] + [_ring_str(h) for h in holes_open]
    return "POLYGON ({})".format(", ".join(parts))


def _parse_multilinestring_to_segments(
    wkt: str,
) -> List[Tuple[_Pt, _Pt]]:
    """Parse a WKT ``MULTILINESTRING`` or ``LINESTRING`` into a list of
    ``((x0, y0), (x1, y1))`` segment tuples.

    Each consecutive pair of vertices in a linestring part is returned as one
    segment, so a part with *N* vertices yields *N − 1* segments.
    """
    wkt = wkt.strip()
    upper = wkt.upper()
    segments: List[Tuple[_Pt, _Pt]] = []

    def _parse_coords(coord_str: str) -> List[_Pt]:
        pts: List[_Pt] = []
        for pair in coord_str.strip().split(","):
            parts = pair.strip().split()
            if len(parts) >= 2:
                try:
                    pts.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
        return pts

    def _add_segments(linestring_body: str) -> None:
        pts = _parse_coords(linestring_body.strip().strip("()"))
        for i in range(len(pts) - 1):
            segments.append((pts[i], pts[i + 1]))

    if upper.startswith("MULTILINESTRING"):
        # Strip "MULTILINESTRING (" prefix and final ")"
        inner_start = wkt.index("(") + 1
        inner_end = wkt.rindex(")")
        inner = wkt[inner_start:inner_end]

        # Walk character by character to split on top-level "," separators
        depth = 0
        current: List[str] = []
        for ch in inner:
            if ch == "(":
                depth += 1
                current.append(ch)
            elif ch == ")":
                depth -= 1
                current.append(ch)
                if depth == 0:
                    _add_segments("".join(current))
                    current = []
            elif ch == "," and depth == 0:
                pass  # separator between linestring parts
            else:
                current.append(ch)
        if current:
            _add_segments("".join(current))

    elif upper.startswith("LINESTRING"):
        inner = wkt[wkt.index("(") + 1: wkt.rindex(")")]
        _add_segments(inner)

    return segments


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def tile_and_extract_centerline(
    wkt: str,
    method: str = "voronoi",
    densify_distance: float = 1.0,
    prune_threshold: float = 0.0,
    smooth_sigma: float = 0.0,
    raster_resolution: Optional[float] = None,
    single_line: bool = True,
    max_vertices: int = _DEFAULT_MAX_VERTICES,
    max_bbox_area: Optional[float] = None,
    buffer_factor: float = _DEFAULT_BUFFER_FACTOR,
    max_depth: int = _DEFAULT_MAX_DEPTH,
    progress_callback=None,
) -> Optional[str]:
    """Three-phase tiled centerline extraction for large or complex polygons.

    Applies the following pipeline to a WKT polygon (or multipolygon):

    **Phase A** — Parse a MULTIPOLYGON into independent polygon parts so that
    disconnected water bodies cannot interfere with each other's skeleton graph
    or pruning decisions.

    **Phase B** — For each polygon part that exceeds *max_vertices* (or
    *max_bbox_area* when specified), recursively split it into four equal
    quadrants using Sutherland-Hodgman polygon clipping.  The recursion
    continues until every tile is below the thresholds or *max_depth* is
    reached.

    **Phase C** — Before clipping the source polygon to a quadrant, expand the
    quadrant boundary outward by ``buffer_factor × densify_distance`` units
    (the *overlap buffer*).  The centerline is extracted from this larger
    buffered tile and then clipped back to the original quadrant boundary so
    that adjacent tiles produce seamless results when merged.

    Parameters
    ----------
    wkt:
        WKT ``POLYGON`` or ``MULTIPOLYGON`` string.
    method:
        Centerline algorithm — ``"voronoi"`` (default) or ``"skeleton"``.
    densify_distance:
        Maximum spacing between boundary sample points for the Voronoi
        tessellation, in CRS units.
    prune_threshold:
        Minimum branch length to retain; 0 disables explicit pruning.
    smooth_sigma:
        Gaussian smoothing radius for the raster skeleton method.
    raster_resolution:
        Pixel size for the skeleton method (defaults to *densify_distance*).
    single_line:
        ``True`` — degree-aware branching skeleton (default).
        ``False`` — raw Voronoi skeleton.
    max_vertices:
        Phase B threshold: tiles with more than this many boundary vertices
        are subdivided.  Lower values create more, smaller tiles.
        Default 8,000.
    max_bbox_area:
        Phase B threshold: tiles whose bounding-box area exceeds this value
        are subdivided.  Set in CRS area units (e.g. square metres or square
        degrees).  ``None`` (default) disables the area check.
    buffer_factor:
        Phase C overlap buffer expressed as a multiple of *densify_distance*.
        Default 5.0 (recommended range: 3–8).
    max_depth:
        Maximum quad-tree recursion depth.  At depth *d* a single polygon can
        produce at most 4 ** *d* tiles.  Default 5 (≤ 1 024 tiles).
    progress_callback:
        Optional ``callable(message: str, percentage: int)`` for progress
        reporting.  *percentage* is 0–100 or -1 if indeterminate.

    Returns
    -------
    WKT ``MULTILINESTRING`` with the merged centerline, or ``None`` if no
    centerline could be extracted from any tile.
    """
    # ── Lazy import of core algorithm ───────────────────────────────────────
    _tbx_dir = os.path.dirname(os.path.abspath(__file__))
    if _tbx_dir not in sys.path:
        sys.path.insert(0, _tbx_dir)

    try:
        from centerline_degree import (
            polygon_to_centerline_wkt as _extract,
            _parse_wkt_polygon,
        )
    except ImportError as exc:
        raise ImportError(
            "Cannot import 'centerline_degree'.  "
            "Ensure 'centerline_degree.py' is in the same directory as "
            "'split_and_process.py': {}".format(_tbx_dir)
        ) from exc

    def _report(msg: str, pct: int = -1) -> None:
        if progress_callback is not None:
            progress_callback(msg, pct)

    buffer = buffer_factor * densify_distance

    # ── Phase A: parse MULTIPOLYGON → list of (exterior_np, holes_np) ───────
    polygons = _parse_wkt_polygon(wkt)
    if not polygons:
        return None

    n_polygons = len(polygons)
    all_segments: List[Tuple[_Pt, _Pt]] = []

    for poly_idx, (exterior_np, holes_np) in enumerate(polygons):
        # exterior_np shape: (N, 2), closed ring (first == last)
        if len(exterior_np) < 4:
            continue

        # Convert to open rings (list of tuples) for SH clipping
        ext_open: _Ring = [
            (float(exterior_np[i, 0]), float(exterior_np[i, 1]))
            for i in range(len(exterior_np) - 1)    # drop repeated last vertex
        ]
        holes_open: List[_Ring] = []
        for hole_np in holes_np:
            if len(hole_np) >= 4:
                holes_open.append([
                    (float(hole_np[i, 0]), float(hole_np[i, 1]))
                    for i in range(len(hole_np) - 1)
                ])

        # Polygon bounding box
        xs = [p[0] for p in ext_open]
        ys = [p[1] for p in ext_open]
        poly_bbox: _BBox = (min(xs), min(ys), max(xs), max(ys))

        # ── Phase B: get tiles via adaptive quad-tree ────────────────────────
        _report(
            "Polygon {}/{}: computing tiles ...".format(poly_idx + 1, n_polygons),
            int(100 * poly_idx / n_polygons),
        )
        tiles = _adaptive_tiles(
            ext_open, holes_open, poly_bbox,
            max_vertices, max_bbox_area, buffer, max_depth,
        )

        n_tiles = len(tiles)
        _report(
            "Polygon {}/{}: {:,} tile(s).".format(poly_idx + 1, n_polygons, n_tiles),
            int(100 * poly_idx / n_polygons),
        )

        for tile_idx, (tile_ext, tile_holes, tile_bbox) in enumerate(tiles):
            if len(tile_ext) < 3:
                continue

            _report(
                "Polygon {}/{}, tile {}/{}: extracting centerline ...".format(
                    poly_idx + 1, n_polygons, tile_idx + 1, n_tiles,
                ),
                int(100 * (poly_idx + tile_idx / max(n_tiles, 1)) / n_polygons),
            )

            # Build WKT for this buffered tile polygon
            tile_wkt = _rings_to_polygon_wkt(tile_ext, tile_holes)

            # ── Phase C: extract centerline from buffered tile ───────────────
            try:
                result_wkt = _extract(
                    tile_wkt,
                    method=method,
                    densify_distance=densify_distance,
                    prune_threshold=prune_threshold,
                    smooth_sigma=smooth_sigma,
                    raster_resolution=raster_resolution,
                    single_line=single_line,
                )
            except Exception:
                result_wkt = None

            if result_wkt is None:
                continue

            # ── Phase C: clip centerline back to original tile bbox ──────────
            segs = _parse_multilinestring_to_segments(result_wkt)
            for (x0, y0), (x1, y1) in segs:
                clipped = _clip_segment(x0, y0, x1, y1, tile_bbox)
                if clipped is not None:
                    all_segments.append(clipped)

    if not all_segments:
        return None

    _report("Merging {:,} centerline segment(s) ...".format(len(all_segments)), 99)

    # Build output MULTILINESTRING WKT
    parts = [
        "({} {}, {} {})".format(p0[0], p0[1], p1[0], p1[1])
        for p0, p1 in all_segments
    ]
    return "MULTILINESTRING ({})".format(", ".join(parts))
