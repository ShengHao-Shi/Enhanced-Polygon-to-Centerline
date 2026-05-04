# -*- coding: utf-8 -*-
"""
centerline_pure.py
==================
Shapely-free polygon-to-centerline algorithm for ArcGIS Pro.

No shapely, geopandas, or pandas required.

Dependencies
------------
The following packages must be available in the Python environment:

    numpy    – almost always pre-installed in ArcGIS Pro's default environment.
    scipy    – usually pre-installed in ArcGIS Pro (Voronoi tessellation).
    networkx – install from conda-forge:
                   conda install -c conda-forge networkx

For ``method="skeleton"`` also install scikit-image:
                   conda install -c conda-forge scikit-image

Why no shapely?
---------------
The original ``centerline.py`` in the ``gdal_centerline/`` folder depends on
shapely (≥2.0) for polygon containment tests and geometry construction.
On some ArcGIS Pro installations the conda-forge ``shapely`` package triggers
licence-related conflicts that cause the toolbox to be marked "not licensed",
even when all required ArcGIS licences are present.

This module re-implements every shapely call using only numpy arithmetic:

* ``polygon.exterior / .interiors``
      → ring coordinate arrays (numpy ndarray).
* ``polygon.contains(LineString([v0,v1]))``
      → midpoint ray-casting PIP test + no-proper-crossing check.
* ``polygon.contains(Point(x,y))``
      → vectorised ray-casting PIP test (one full raster row at a time).
* ``polygon.bounds``
      → ``np.min / np.max`` on coordinate arrays.
* ``polygon.buffer(0)``
      → degenerate polygons silently skipped.
* ``linemerge / unary_union``
      → WKT string construction (LINESTRING / MULTILINESTRING).

Algorithm overview
------------------
Two methods (identical to ``centerline.py``):

Method A – ``"voronoi"`` (default)
    1. Densify polygon boundary points.
    2. Compute Voronoi tessellation (``scipy.spatial.Voronoi``).
    3. Filter ridges that lie fully inside the polygon (ray-casting +
       segment-intersection test — no shapely).
    4. Build NetworkX graph; iteratively prune dead-end branches.
    5. Extract the single longest trunk path (two-pass Dijkstra diameter).

Method B – ``"skeleton"``
    1. Rasterise polygon to a binary image using vectorised ray-casting.
    2. Optional Gaussian blur (``scipy.ndimage.gaussian_filter``).
    3. Zhang-Suen morphological thinning (``skimage.morphology.skeletonize``).
    4. Build pixel graph (NetworkX); extract longest trunk path.

Public API
----------
    result_wkt = polygon_to_centerline_wkt(
        wkt,                        # WKT string of the input polygon
        method="voronoi",           # "voronoi" or "skeleton"
        densify_distance=1.0,       # vertex spacing for boundary densification
        prune_threshold=0.0,        # branch-pruning length (0 = no pruning)
        smooth_sigma=0.0,           # Gaussian sigma for skeleton method
        raster_resolution=None,     # raster cell size for skeleton method
        single_line=True,           # True = single trunk; False = full skeleton
    )
    # Returns a WKT LINESTRING / MULTILINESTRING string, or None.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import networkx as nx
import numpy as np
from scipy.spatial import Voronoi

try:
    from skimage.morphology import skeletonize as _skeletonize

    _SKIMAGE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKIMAGE_AVAILABLE = False


# ---------------------------------------------------------------------------
# WKT parsing helpers
# ---------------------------------------------------------------------------


def _split_at_depth(text: str, depth: int) -> list:
    """
    Split *text* at commas that are at exactly *depth* levels of nested
    parentheses (depth 0 = outside all parentheses).

    Returns a list of stripped, non-empty sub-strings.
    """
    result = []
    current: list = []
    d = 0
    for ch in text:
        if ch == "(":
            d += 1
            current.append(ch)
        elif ch == ")":
            d -= 1
            current.append(ch)
        elif ch == "," and d == depth:
            result.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        result.append("".join(current).strip())
    return [s for s in result if s]


def _parse_ring(ring_str: str) -> np.ndarray:
    """
    Parse a coordinate string ``'x1 y1, x2 y2, ...'`` (with optional
    surrounding parentheses) into an ``(N, 2)`` float64 ndarray.

    Z-values (``x y z``) are accepted but silently discarded.
    """
    ring_str = ring_str.strip().strip("()")
    coords = []
    for pair in ring_str.split(","):
        parts = pair.strip().split()
        if len(parts) >= 2:
            try:
                coords.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if coords:
        return np.array(coords, dtype=np.float64)
    return np.empty((0, 2), dtype=np.float64)


def _parse_polygon_body(body: str):
    """
    Parse the body of a ``POLYGON (...)`` — the content inside the outermost
    parentheses — and return ``(exterior, holes)``.

    Parameters
    ----------
    body : str
        Content between the outermost ``POLYGON`` parentheses, e.g.
        ``'(x y, ...), (hx hy, ...)'``.

    Returns
    -------
    exterior : np.ndarray, shape (N, 2)
    holes    : list of np.ndarray
    """
    ring_strs = _split_at_depth(body, 0)
    rings = [_parse_ring(r) for r in ring_strs if r.strip()]
    rings = [r for r in rings if len(r) >= 3]
    if not rings:
        return np.empty((0, 2), dtype=np.float64), []
    return rings[0], rings[1:]


def _parse_wkt_polygon(wkt: str) -> list:
    """
    Parse a WKT ``POLYGON`` or ``MULTIPOLYGON`` string.

    Returns
    -------
    list of (exterior, holes) tuples
        *exterior* and each element of *holes* is an ``(N, 2)`` float64
        ndarray of ring coordinates.  Returns an empty list for unrecognised
        or degenerate geometry.
    """
    wkt = wkt.strip()
    upper = wkt.upper()

    if upper.startswith("MULTIPOLYGON"):
        body = wkt[wkt.index("(") + 1: wkt.rindex(")")]
        poly_strs = _split_at_depth(body, 0)
        result = []
        for ps in poly_strs:
            ps = ps.strip()
            # Each element is '((ring), (hole), ...)'
            if ps.startswith("(") and ps.endswith(")"):
                ps = ps[1:-1]  # strip one layer of parens
            ext, holes = _parse_polygon_body(ps)
            if len(ext) >= 4:
                result.append((ext, holes))
        return result

    if upper.startswith("POLYGON"):
        body = wkt[wkt.index("(") + 1: wkt.rindex(")")]
        ext, holes = _parse_polygon_body(body)
        if len(ext) >= 4:
            return [(ext, holes)]
        return []

    return []


# ---------------------------------------------------------------------------
# WKT output helpers
# ---------------------------------------------------------------------------


def _path_to_linestring_wkt(coords) -> str:
    """Convert an ordered sequence of (x, y) coords to a WKT LINESTRING."""
    pts = ", ".join("{} {}".format(x, y) for x, y in coords)
    return "LINESTRING ({})".format(pts)


def _edges_to_multilinestring_wkt(edge_pairs) -> Optional[str]:
    """
    Convert a list of ``((x1, y1), (x2, y2))`` edge pairs to a WKT
    MULTILINESTRING.  Returns None for an empty list.
    """
    if not edge_pairs:
        return None
    parts = [
        "({} {}, {} {})".format(s[0], s[1], e[0], e[1]) for s, e in edge_pairs
    ]
    return "MULTILINESTRING ({})".format(", ".join(parts))


def _paths_to_wkt(paths: list) -> Optional[str]:
    """
    Convert a list of ordered coordinate lists to a WKT string.
    One path → ``LINESTRING``; multiple paths → ``MULTILINESTRING``.
    """
    if not paths:
        return None
    if len(paths) == 1:
        return _path_to_linestring_wkt(paths[0])
    ring_strs = []
    for path in paths:
        pts = ", ".join("{} {}".format(x, y) for x, y in path)
        ring_strs.append("({})".format(pts))
    return "MULTILINESTRING ({})".format(", ".join(ring_strs))


# ---------------------------------------------------------------------------
# Point-in-polygon — ray-casting (replaces shapely.contains)
# ---------------------------------------------------------------------------


def _pip_ring(x: float, y: float, ring: np.ndarray) -> bool:
    """
    Ray-casting point-in-polygon test for a single ring.

    Works for both CCW (exterior) and CW (hole) winding orders because the
    crossing-number method is winding-order independent.
    """
    n = len(ring)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i, 0], ring[i, 1]
        xj, yj = ring[j, 0], ring[j, 1]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def _point_in_polygon(
    x: float, y: float, exterior: np.ndarray, holes: list
) -> bool:
    """Return True when (x, y) is inside the polygon (exterior minus holes)."""
    if not _pip_ring(x, y, exterior):
        return False
    for hole in holes:
        if _pip_ring(x, y, hole):
            return False
    return True


def _pip_ring_vectorized(xs: np.ndarray, y: float, ring: np.ndarray) -> np.ndarray:
    """
    Vectorised ray-casting for a row of x-values at constant y.

    Returns a boolean array of shape ``(len(xs),)`` where True means the
    point ``(xs[i], y)`` is inside *ring*.
    """
    n = len(ring)
    inside = np.zeros(len(xs), dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i, 0], ring[i, 1]
        xj, yj = ring[j, 0], ring[j, 1]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            inside ^= xs < x_cross
        j = i
    return inside


# ---------------------------------------------------------------------------
# Segment-in-polygon test (replaces shapely.polygon.contains(LineString))
# ---------------------------------------------------------------------------


def _cross2d(
    ox: float, oy: float, ax: float, ay: float, bx: float, by: float
) -> float:
    """2D cross product of vectors ``(A - O)`` × ``(B - O)``."""
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


def _segments_properly_intersect(
    p1: tuple, p2: tuple, p3: tuple, p4: tuple
) -> bool:
    """
    Return True if segments ``[p1, p2]`` and ``[p3, p4]`` *properly* cross
    each other (intersection strictly in the interior of both segments;
    endpoint-only touches are not counted as proper crossings).
    """
    d1 = _cross2d(p3[0], p3[1], p4[0], p4[1], p1[0], p1[1])
    d2 = _cross2d(p3[0], p3[1], p4[0], p4[1], p2[0], p2[1])
    d3 = _cross2d(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])
    d4 = _cross2d(p1[0], p1[1], p2[0], p2[1], p4[0], p4[1])
    return (
        ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0))
        and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))
    )


def _segment_crosses_ring(v0: tuple, v1: tuple, ring: np.ndarray) -> bool:
    """
    Return True if segment ``[v0, v1]`` properly crosses any edge of *ring*.

    A bounding-box pre-filter skips ring edges whose axis-aligned extent does
    not overlap the segment's extent, reducing the number of intersection
    tests significantly for large rings.
    """
    seg_minx = min(v0[0], v1[0])
    seg_maxx = max(v0[0], v1[0])
    seg_miny = min(v0[1], v1[1])
    seg_maxy = max(v0[1], v1[1])

    n = len(ring)
    for i in range(n):
        j = (i + 1) % n
        # Bounding-box pre-filter
        emx = max(ring[i, 0], ring[j, 0])
        emnx = min(ring[i, 0], ring[j, 0])
        emy = max(ring[i, 1], ring[j, 1])
        emny = min(ring[i, 1], ring[j, 1])
        if seg_maxx < emnx or seg_minx > emx:
            continue
        if seg_maxy < emny or seg_miny > emy:
            continue
        if _segments_properly_intersect(
            v0, v1, (ring[i, 0], ring[i, 1]), (ring[j, 0], ring[j, 1])
        ):
            return True
    return False


def _segment_in_polygon(
    v0: tuple, v1: tuple, exterior: np.ndarray, holes: list
) -> bool:
    """
    Return True if segment ``[v0, v1]`` lies fully inside the polygon.

    Strategy
    --------
    1. The midpoint must pass the PIP test (inside exterior, outside all holes).
    2. The segment must not properly cross any polygon ring boundary.

    This is behaviourally equivalent to shapely's
    ``polygon.contains(LineString([v0, v1]))`` for the practical purposes of
    the Voronoi skeleton algorithm.
    """
    mx = (v0[0] + v1[0]) * 0.5
    my = (v0[1] + v1[1]) * 0.5
    if not _point_in_polygon(mx, my, exterior, holes):
        return False
    for ring in [exterior] + holes:
        if _segment_crosses_ring(v0, v1, ring):
            return False
    return True


# ---------------------------------------------------------------------------
# Polygon boundary densification (replaces shapely ring.coords)
# ---------------------------------------------------------------------------


def _densify(
    exterior: np.ndarray, holes: list, max_distance: float
):
    """
    Densify polygon ring coordinates so that no two consecutive vertices are
    more than *max_distance* apart.

    Returns
    -------
    pts      : np.ndarray, shape (N, 2)
    ring_ids : np.ndarray, shape (N,), dtype int
        0 = exterior ring; ≥1 = hole index (1-based).
    """
    rings = [exterior] + holes
    points: list = []
    ring_ids: list = []
    for ring_idx, ring in enumerate(rings):
        coords = ring
        for i in range(len(coords) - 1):
            p0, p1 = coords[i], coords[i + 1]
            seg_len = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            n_insert = max(1, math.ceil(seg_len / max_distance))
            for k in range(n_insert):
                t = k / n_insert
                points.append(p0 + t * (p1 - p0))
                ring_ids.append(ring_idx)
    return np.array(points), np.array(ring_ids, dtype=int)


# ---------------------------------------------------------------------------
# Graph utilities (identical logic to centerline.py — no shapely involved)
# ---------------------------------------------------------------------------


def _prune_branches(G: nx.Graph, threshold: float) -> nx.Graph:
    """
    Iteratively remove dead-end skeleton branches whose total path length
    is shorter than *threshold*.

    A branch starts at a leaf (degree-1 node) and runs to the first junction
    (degree ≥ 3) or to another leaf.  The junction node is kept to preserve
    the connectivity of the remaining skeleton.
    """
    changed = True
    while changed:
        changed = False
        leaves = [n for n in list(G.nodes()) if G.degree(n) == 1]
        for leaf in leaves:
            if leaf not in G or G.degree(leaf) != 1:
                continue
            path = [leaf]
            total_len = 0.0
            current = leaf
            while True:
                neighbors = list(G.neighbors(current))
                nxt = None
                for nb in neighbors:
                    if nb not in path:
                        nxt = nb
                        break
                if nxt is None:
                    break
                edge_len = G[current][nxt].get("weight", 0.0)
                total_len += edge_len
                path.append(nxt)
                current = nxt
                if G.degree(current) != 2:
                    break
                if total_len >= threshold:
                    break
            if total_len < threshold:
                nodes_to_remove = path[:-1]
                if len(path) >= 2 and G.degree(path[-1]) == 1:
                    nodes_to_remove = path
                for node in nodes_to_remove:
                    if node in G:
                        G.remove_node(node)
                changed = True
    return G


def _traverse_cycle(G: nx.Graph) -> list:
    """
    Walk all nodes of a pure-cycle graph (every node degree 2) and return a
    closed node list (first node == last node).

    Used to return the full closed loop for ring-shaped polygons (e.g. 'O').
    """
    if G.number_of_nodes() == 0:
        return []
    start = next(iter(G.nodes()))
    path = [start]
    prev = None
    current = start
    while True:
        nxt = None
        for nb in G.neighbors(current):
            if nb != prev:
                nxt = nb
                break
        if nxt is None or nxt == start:
            path.append(start)
            break
        prev = current
        current = nxt
        if current == start:
            path.append(start)
            break
        path.append(current)
    return path


def _extract_longest_path(G: nx.Graph) -> list:
    """
    Find the single longest (by sum of edge weights) path between two leaf
    nodes using the two-pass Dijkstra diameter algorithm.

    Special cases
    -------------
    * Pure-cycle graph (all nodes degree 2): calls ``_traverse_cycle`` to
      return the full closed loop (e.g. 'O' ring polygon).
    * Disconnected graph: works on the largest connected component.
    """
    if G.number_of_nodes() < 2:
        return list(G.nodes())

    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        if G.number_of_nodes() < 2:
            return list(G.nodes())

    leaves = [n for n in G.nodes() if G.degree(n) == 1]

    if not leaves:
        if all(G.degree(n) == 2 for n in G.nodes()):
            return _traverse_cycle(G)
        leaves = list(G.nodes())

    # Pass 1: from an arbitrary leaf to find the farthest leaf u
    start = leaves[0]
    dist1, _ = nx.single_source_dijkstra(G, start, weight="weight")
    u = max(leaves, key=lambda n: dist1.get(n, 0.0))

    # Pass 2: from u to find the true diameter endpoint v
    dist2, path2 = nx.single_source_dijkstra(G, u, weight="weight")
    v = max(leaves, key=lambda n: dist2.get(n, 0.0))

    return path2.get(v, [u, v])


# ---------------------------------------------------------------------------
# Method A – Voronoi skeleton (shapely-free)
# ---------------------------------------------------------------------------


def _centerline_voronoi(
    exterior: np.ndarray,
    holes: list,
    densify_distance: float,
    prune_threshold: float,
    single_line: bool = True,
):
    """
    Compute the centerline of a polygon using a Voronoi skeleton.

    Identical algorithm to ``centerline.py:_centerline_voronoi`` but without
    any shapely calls.  Containment tests are performed by ``_segment_in_polygon``
    (midpoint PIP + no-crossing check).

    Returns
    -------
    list of (x, y) tuples  when ``single_line=True``  (the ordered trunk path)
    list of ((x1,y1),(x2,y2)) pairs  when ``single_line=False``  (all edges)
    None  if no valid centerline could be computed
    """
    pts, ring_ids = _densify(exterior, holes, densify_distance)
    if len(pts) < 4:
        return None

    vor = Voronoi(pts)
    min_gen_dist = 3.0 * densify_distance

    segments = []
    for ridge_idx, ridge in enumerate(vor.ridge_vertices):
        i, j = ridge
        if i < 0 or j < 0:
            continue  # infinite ridge

        p_idx, q_idx = vor.ridge_points[ridge_idx]

        # Generator-distance filter: suppress same-ring artefact ridges
        # (parallel to boundary).  Cross-ring ridges are NOT filtered here
        # so that narrow ring-shaped polygons (e.g. letter 'O') keep their
        # centerline.
        if ring_ids[p_idx] == ring_ids[q_idx]:
            gp0 = pts[p_idx]
            gp1 = pts[q_idx]
            if math.hypot(gp1[0] - gp0[0], gp1[1] - gp0[1]) < min_gen_dist:
                continue

        v0 = (vor.vertices[i, 0], vor.vertices[i, 1])
        v1 = (vor.vertices[j, 0], vor.vertices[j, 1])

        if _segment_in_polygon(v0, v1, exterior, holes):
            segments.append((v0, v1))

    if not segments:
        return None

    G = nx.Graph()
    for p, q in segments:
        length = math.hypot(q[0] - p[0], q[1] - p[1])
        G.add_edge(p, q, weight=length)

    if prune_threshold > 0:
        G = _prune_branches(G, prune_threshold)

    if G.number_of_edges() == 0:
        return None

    if single_line:
        path_nodes = _extract_longest_path(G)
        if len(path_nodes) < 2:
            return None
        return list(path_nodes)  # list of (x, y) tuples

    # single_line=False: return all edges
    return [(u, v) for u, v in G.edges()]


# ---------------------------------------------------------------------------
# Method B – Raster skeleton (shapely-free)
# ---------------------------------------------------------------------------


def _rasterize_polygon(
    exterior: np.ndarray,
    holes: list,
    minx: float,
    miny: float,
    resolution: float,
    rows: int,
    cols: int,
) -> np.ndarray:
    """
    Rasterise the polygon into a boolean ``(rows × cols)`` array using
    vectorised ray-casting.  No shapely required.

    Pixels whose *centre* falls inside the polygon (exterior minus holes)
    are set to True.
    """
    xs = minx + (np.arange(cols) + 0.5) * resolution
    ys = miny + (np.arange(rows) + 0.5) * resolution

    poly_miny = exterior[:, 1].min()
    poly_maxy = exterior[:, 1].max()

    binary = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        y = ys[r]
        # Bounding-box row skip (significant speedup for tall bounding boxes)
        if y < poly_miny or y > poly_maxy:
            continue
        inside = _pip_ring_vectorized(xs, y, exterior)
        for hole in holes:
            inside &= ~_pip_ring_vectorized(xs, y, hole)
        binary[r] = inside
    return binary


def _build_skeleton_graph(
    skeleton: np.ndarray,
    minx: float,
    miny: float,
    resolution: float,
) -> nx.Graph:
    """
    Build a weighted NetworkX graph from a binary skeleton image.

    Nodes are ``(row, col)`` pixel indices with a ``coord`` attribute giving
    the world-coordinate centre of each pixel.  Edge weights are Euclidean
    pixel distances (1.0 × resolution for cardinal, √2 × resolution for
    diagonal neighbours).
    """

    def px_to_coord(r: int, c: int):
        return (minx + (c + 0.5) * resolution, miny + (r + 0.5) * resolution)

    yx = np.argwhere(skeleton)
    if len(yx) == 0:
        return nx.Graph()

    pixel_set = set(map(tuple, yx))
    G = nx.Graph()
    for r, c in yx:
        G.add_node((r, c), coord=px_to_coord(r, c))
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in pixel_set:
                    G.add_edge(
                        (r, c), nb, weight=math.hypot(dr, dc) * resolution
                    )
    return G


def _centerline_skeleton(
    exterior: np.ndarray,
    holes: list,
    resolution: float,
    smooth_sigma: float = 0.0,
    single_line: bool = True,
):
    """
    Compute the centerline of a polygon using raster skeletonisation.

    Identical algorithm to ``centerline.py:_centerline_skeleton`` but without
    shapely.  Rasterisation uses vectorised ray-casting (``_rasterize_polygon``).

    Returns
    -------
    list of (x, y) tuples  when ``single_line=True``
    list of ((x1,y1),(x2,y2)) pairs  when ``single_line=False``
    None  if no valid centerline could be computed
    """
    if not _SKIMAGE_AVAILABLE:
        raise ImportError(
            "scikit-image is required for method='skeleton'. "
            "Install it with:  conda install -c conda-forge scikit-image"
        )

    # Compute bounding box from coordinate arrays (no shapely .bounds)
    minx = float(exterior[:, 0].min())
    miny = float(exterior[:, 1].min())
    maxx = float(exterior[:, 0].max())
    maxy = float(exterior[:, 1].max())

    # Add a 1-cell border so the skeleton never touches the image edge
    minx -= resolution
    miny -= resolution
    maxx += resolution
    maxy += resolution

    cols = max(2, math.ceil((maxx - minx) / resolution))
    rows = max(2, math.ceil((maxy - miny) / resolution))

    binary = _rasterize_polygon(exterior, holes, minx, miny, resolution, rows, cols)

    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter

        sigma_px = smooth_sigma / resolution
        blurred = gaussian_filter(binary.astype(float), sigma=sigma_px)
        binary = blurred > 0.5

    skeleton = _skeletonize(binary)

    G = _build_skeleton_graph(skeleton, minx, miny, resolution)

    if G.number_of_edges() == 0:
        return None

    if single_line:
        path_pixels = _extract_longest_path(G)
        if len(path_pixels) < 2:
            return None
        return [G.nodes[p]["coord"] for p in path_pixels]

    # single_line=False: return all skeleton edges
    return [(G.nodes[u]["coord"], G.nodes[v]["coord"]) for u, v in G.edges()]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def polygon_to_centerline_wkt(
    wkt: str,
    method: str = "voronoi",
    densify_distance: float = 1.0,
    prune_threshold: float = 0.0,
    smooth_sigma: float = 0.0,
    raster_resolution: Optional[float] = None,
    single_line: bool = True,
) -> Optional[str]:
    """
    Convert a polygon WKT string to a centerline WKT string.

    Parameters
    ----------
    wkt : str
        WKT string of the input polygon (``POLYGON`` or ``MULTIPOLYGON``).
    method : {"voronoi", "skeleton"}
        Algorithm to use.  ``"voronoi"`` (default) is vector-based;
        ``"skeleton"`` is raster-based and requires scikit-image.
    densify_distance : float
        Maximum vertex spacing (CRS units) for boundary densification.
        Smaller values produce a finer skeleton (voronoi method only).
    prune_threshold : float
        Remove dead-end branches shorter than this value (CRS units).
        0 = no pruning (default).  Voronoi method only.
    smooth_sigma : float
        Gaussian smooth sigma (CRS units) applied before skeletonisation.
        0 = no blur (default).  Skeleton method only.
    raster_resolution : float or None
        Raster cell size (CRS units) for the skeleton method.
        None = use *densify_distance* as the cell size.
    single_line : bool
        When True (default) the output is a single non-branching trunk
        ``LINESTRING``.  When False the full skeleton (may contain forks)
        is returned as a ``MULTILINESTRING``.

    Returns
    -------
    str or None
        WKT ``LINESTRING`` or ``MULTILINESTRING``, or None for degenerate
        or empty input.
    """
    polygons = _parse_wkt_polygon(wkt)
    if not polygons:
        return None

    all_results = []
    for exterior, holes in polygons:
        if len(exterior) < 4:
            continue
        try:
            if method == "voronoi":
                result = _centerline_voronoi(
                    exterior, holes, densify_distance, prune_threshold, single_line
                )
            elif method == "skeleton":
                res = raster_resolution if raster_resolution else densify_distance
                result = _centerline_skeleton(
                    exterior, holes, res, smooth_sigma, single_line
                )
            else:
                raise ValueError("Unknown method: {!r}".format(method))
        except (RuntimeError, ValueError) as exc:
            warnings.warn(
                "Skipping polygon part: {}".format(exc), stacklevel=2
            )
            result = None

        if result is not None and len(result) >= 2:
            all_results.append(result)

    if not all_results:
        return None

    if single_line:
        # Each result is a list of (x, y) coordinate tuples — the trunk path
        return _paths_to_wkt(all_results)

    # single_line=False: each result is a list of ((x1,y1),(x2,y2)) edge pairs
    all_edges = [edge for result in all_results for edge in result]
    return _edges_to_multilinestring_wkt(all_edges)
