# -*- coding: utf-8 -*-
"""
tests/test_proportional_buffer.py
==================================
Unit and integration tests for the proportional_buffer module.

Run with pytest from the repository root::

    pytest proportional_buffer/tests/

Dependencies for testing
------------------------
    pytest  >= 7.0
    numpy   >= 1.24
    scipy   >= 1.10
    shapely >= 2.0
    geopandas >= 0.13  (optional – tests that need it are auto-skipped)
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Make sure the parent directory is on sys.path so we can import the module
# directly whether pytest is run from the repo root or from within the
# proportional_buffer/ directory.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from proportional_buffer import (
    _densify_coords,
    _compute_normals,
    _arc_points,
    _build_buffer_ring,
    _auto_sample_distance,
    _build_boundary_tree,
    _sample_centerline,
    _compute_half_widths,
    chaikin_smooth,
    compute_proportional_buffer,
)

try:
    from shapely.geometry import LineString, MultiLineString, Point, Polygon
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

try:
    import geopandas as gpd
    from proportional_buffer import process_geodataframes
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

requires_shapely = pytest.mark.skipif(not HAS_SHAPELY, reason="shapely not installed")
requires_geopandas = pytest.mark.skipif(not HAS_GEOPANDAS, reason="geopandas not installed")


# ===========================================================================
# Pure-numpy helpers (no shapely required)
# ===========================================================================


class TestDensifyCoords:
    def test_already_dense_unchanged(self):
        """If points are closer than max_spacing, no new points are added."""
        coords = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
        out = _densify_coords(coords, max_spacing=2.0)
        # Should still start at 0 and end at 1
        assert out[0, 0] == pytest.approx(0.0)
        assert out[-1, 0] == pytest.approx(1.0)

    def test_single_long_segment_gets_subdivided(self):
        coords = np.array([[0.0, 0.0], [100.0, 0.0]])
        out = _densify_coords(coords, max_spacing=10.0)
        assert len(out) >= 11  # at least 11 points for a 100-unit segment
        diffs = np.diff(out[:, 0])
        assert np.all(diffs <= 10.0 + 1e-9)

    def test_uniform_spacing(self):
        """All consecutive points in the output should be equidistant."""
        coords = np.array([[0.0, 0.0], [60.0, 0.0]])
        out = _densify_coords(coords, max_spacing=10.0)
        dists = np.hypot(np.diff(out[:, 0]), np.diff(out[:, 1]))
        assert np.allclose(dists, dists[0], atol=1e-9)

    def test_degenerate_single_point_unchanged(self):
        coords = np.array([[5.0, 5.0]])
        out = _densify_coords(coords, max_spacing=1.0)
        assert len(out) == 1

    def test_zero_spacing_returns_input(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        out = _densify_coords(coords, max_spacing=0.0)
        np.testing.assert_array_equal(out, coords)


class TestComputeNormals:
    def test_straight_horizontal_centerline(self):
        """Normals to a horizontal line should all be (0, 1)."""
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        normals = _compute_normals(pts)
        assert normals.shape == (4, 2)
        np.testing.assert_allclose(normals[:, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(normals[:, 1], 1.0, atol=1e-10)

    def test_normals_are_unit_vectors(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, -0.3], [3.0, 0.0]])
        normals = _compute_normals(pts)
        lengths = np.hypot(normals[:, 0], normals[:, 1])
        np.testing.assert_allclose(lengths, 1.0, atol=1e-10)

    def test_normals_perpendicular_to_tangents(self):
        """Dot product of tangent and normal should be zero."""
        pts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        normals = _compute_normals(pts)
        tangents = np.zeros_like(pts)
        tangents[0] = pts[1] - pts[0]
        tangents[1] = pts[2] - pts[0]
        tangents[2] = pts[2] - pts[1]
        for i in range(len(pts)):
            t_len = math.hypot(tangents[i, 0], tangents[i, 1])
            if t_len > 1e-12:
                t = tangents[i] / t_len
                dot = abs(np.dot(t, normals[i]))
                assert dot < 1e-10, f"Normal at point {i} is not perpendicular to tangent"

    def test_single_segment_endpoints_consistent(self):
        """For a two-point line, both normals should agree."""
        pts = np.array([[0.0, 0.0], [4.0, 3.0]])
        normals = _compute_normals(pts)
        np.testing.assert_allclose(normals[0], normals[1], atol=1e-10)


class TestArcPoints:
    def test_start_end_angles(self):
        """First and last arc points should match the specified angles."""
        center = (0.0, 0.0)
        radius = 5.0
        arc = _arc_points(center, radius, 0.0, math.pi, 9)
        # At angle 0: (5, 0); at angle π: (–5, 0)
        np.testing.assert_allclose(arc[0], [5.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(arc[-1], [-5.0, 0.0], atol=1e-8)

    def test_all_points_on_circle(self):
        center = (3.0, -2.0)
        radius = 7.0
        arc = _arc_points(center, radius, 0.0, 2 * math.pi, 20)
        dists = np.hypot(arc[:, 0] - center[0], arc[:, 1] - center[1])
        np.testing.assert_allclose(dists, radius, atol=1e-9)


class TestChaikinSmooth:
    def test_output_has_more_points(self):
        square = [(0, 0), (1, 0), (1, 1), (0, 1)]
        smoothed = chaikin_smooth(square, iterations=2)
        assert len(smoothed) > len(square)

    def test_output_is_closed(self):
        square = [(0, 0), (1, 0), (1, 1), (0, 1)]
        smoothed = chaikin_smooth(square, iterations=1)
        np.testing.assert_allclose(smoothed[0], smoothed[-1], atol=1e-10)

    def test_zero_iterations_returns_closed_ring(self):
        square = [(0, 0), (1, 0), (1, 1), (0, 1)]
        out = chaikin_smooth(square, iterations=0)
        assert len(out) >= len(square)
        np.testing.assert_allclose(out[0], out[-1], atol=1e-10)

    def test_already_closed_input_unchanged_topology(self):
        coords = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
        smoothed = chaikin_smooth(coords, iterations=2)
        np.testing.assert_allclose(smoothed[0], smoothed[-1], atol=1e-10)


class TestBuildBufferRing:
    def _rect_centerline(self):
        """10-sample horizontal centerline (y=0) for a 100×10 rectangle."""
        xs = np.linspace(-50.0, 50.0, 10)
        return np.column_stack([xs, np.zeros(10)])

    def test_ring_is_closed(self):
        pts = self._rect_centerline()
        hw = np.full(len(pts), 5.0)
        nrm = _compute_normals(pts)
        ring = _build_buffer_ring(pts, hw, nrm, 1.0, 0, float("inf"), "flat", 8)
        np.testing.assert_allclose(ring[0], ring[-1], atol=1e-10)

    def test_ring_width_at_ratio_1(self):
        """At ratio=1, left/right offset points should be ±hw from centreline."""
        pts = np.array([[0.0, 0.0], [10.0, 0.0]])
        hw = np.array([4.0, 4.0])
        nrm = _compute_normals(pts)
        ring = _build_buffer_ring(pts, hw, nrm, 1.0, 0, float("inf"), "flat", 4)
        # First two points are left-side (y = +4); next after flat cap are right (y = –4)
        y_vals = ring[:, 1]
        assert y_vals.max() == pytest.approx(4.0, abs=1e-9)
        assert y_vals.min() == pytest.approx(-4.0, abs=1e-9)

    def test_width_min_clamp(self):
        """width_min should prevent radii from going below width_min/2."""
        pts = np.array([[0.0, 0.0], [10.0, 0.0]])
        hw = np.array([0.0, 0.0])  # zero half-width
        nrm = _compute_normals(pts)
        ring = _build_buffer_ring(pts, hw, nrm, 1.0, 6.0, float("inf"), "flat", 4)
        y_vals = ring[:, 1]
        assert y_vals.max() == pytest.approx(3.0, abs=1e-9)  # width_min/2 = 3
        assert y_vals.min() == pytest.approx(-3.0, abs=1e-9)

    def test_width_max_clamp(self):
        """width_max should cap radii at width_max/2."""
        pts = np.array([[0.0, 0.0], [10.0, 0.0]])
        hw = np.array([100.0, 100.0])  # very large half-width
        nrm = _compute_normals(pts)
        ring = _build_buffer_ring(pts, hw, nrm, 1.0, 0.0, 4.0, "flat", 4)
        y_vals = ring[:, 1]
        assert y_vals.max() == pytest.approx(2.0, abs=1e-9)  # width_max/2 = 2


# ===========================================================================
# Shapely-dependent tests
# ===========================================================================


@requires_shapely
class TestAutoSampleDistance:
    def test_rectangle_returns_positive(self):
        poly = Polygon([(-50, -5), (50, -5), (50, 5), (-50, 5)])
        sd = _auto_sample_distance(poly)
        assert sd > 0.0

    def test_wider_polygon_larger_sample_distance(self):
        narrow = Polygon([(-50, -1), (50, -1), (50, 1), (-50, 1)])
        wide = Polygon([(-50, -10), (50, -10), (50, 10), (-50, 10)])
        assert _auto_sample_distance(wide) >= _auto_sample_distance(narrow)


@requires_shapely
class TestSampleCenterline:
    def test_linestring_sampled_uniformly(self):
        cl = LineString([(-50.0, 0.0), (50.0, 0.0)])
        segs = _sample_centerline(cl, sample_distance=5.0)
        assert len(segs) == 1
        pts = segs[0]
        assert pts.shape[1] == 2
        dists = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
        assert np.all(dists <= 5.0 + 1e-9)

    def test_multilinestring_returns_multiple_segments(self):
        cl = MultiLineString([
            [(-50, 0), (0, 0)],
            [(0, 0), (50, 0)],
        ])
        segs = _sample_centerline(cl, sample_distance=5.0)
        assert len(segs) == 2

    def test_very_short_line_returns_empty(self):
        """A line shorter than sample_distance should still return 2 points."""
        cl = LineString([(0, 0), (0.001, 0)])
        segs = _sample_centerline(cl, sample_distance=5.0)
        # The densify function returns at least 2 points even if total < spacing
        assert len(segs) == 1
        assert len(segs[0]) >= 2


@requires_shapely
class TestBuildBoundaryTree:
    def test_tree_queries_return_correct_distances(self):
        """For a rectangle, the boundary distance from the centre should be
        the half-height (5 m for a 100×10 m polygon)."""
        poly = Polygon([(-50, -5), (50, -5), (50, 5), (-50, 5)])
        tree = _build_boundary_tree(poly, densify_spacing=0.5)
        centre = np.array([[0.0, 0.0]])
        dists, _ = tree.query(centre)
        assert dists[0] == pytest.approx(5.0, abs=0.1)

    def test_polygon_with_hole(self):
        """Boundary tree for a polygon with a hole should include the interior ring."""
        outer = [(0, 0), (100, 0), (100, 100), (0, 100)]
        hole = [(10, 10), (20, 10), (20, 20), (10, 20)]
        poly = Polygon(outer, [hole])
        tree = _build_boundary_tree(poly, densify_spacing=1.0)
        # Point inside the hole boundary should return a small distance
        pt = np.array([[15.0, 15.0]])
        dists, _ = tree.query(pt)
        assert dists[0] < 10.0


@requires_shapely
class TestComputeHalfWidths:
    def test_rectangle_centerline_half_width(self):
        """For a 100×10 rectangle, centred sample points should have hw ≈ 5 m."""
        poly = Polygon([(-50, -5), (50, -5), (50, 5), (-50, 5)])
        tree = _build_boundary_tree(poly, densify_spacing=0.1)
        sample_pts = np.array([[x, 0.0] for x in np.linspace(-45, 45, 20)])
        hw = _compute_half_widths(sample_pts, tree)
        np.testing.assert_allclose(hw, 5.0, atol=0.2)


# ===========================================================================
# End-to-end tests: compute_proportional_buffer
# ===========================================================================


@requires_shapely
class TestComputeProportionalBuffer:
    """End-to-end tests for the main API function."""

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _rect_poly_and_cl():
        """100 m × 10 m rectangle and its horizontal centreline."""
        poly = Polygon([(-50, -5), (50, -5), (50, 5), (-50, 5)])
        cl = LineString([(-50, 0), (50, 0)])
        return poly, cl

    # ── Basic correctness ────────────────────────────────────────────────────

    def test_returns_valid_polygon(self):
        poly, cl = self._rect_poly_and_cl()
        buf = compute_proportional_buffer(poly, cl, buffer_ratio=0.5, sample_distance=1.0)
        assert buf is not None
        assert not buf.is_empty
        assert buf.is_valid or buf.buffer(0).is_valid

    def test_buffer_is_inside_polygon_when_clipped(self):
        poly, cl = self._rect_poly_and_cl()
        buf = compute_proportional_buffer(
            poly, cl,
            buffer_ratio=0.5,
            sample_distance=1.0,
            clip_to_polygon=True,
        )
        assert poly.contains(buf) or poly.covers(buf)

    def test_buffer_ratio_1_approximately_fills_polygon(self):
        """At ratio=1.0 the buffer should cover most of the polygon."""
        poly, cl = self._rect_poly_and_cl()
        buf = compute_proportional_buffer(
            poly, cl,
            buffer_ratio=1.0,
            sample_distance=0.5,
            clip_to_polygon=True,
            end_cap="flat",
        )
        overlap_ratio = buf.area / poly.area
        assert overlap_ratio > 0.8, f"Expected > 80% coverage, got {overlap_ratio:.2%}"

    def test_buffer_ratio_half_is_narrower_than_polygon(self):
        poly, cl = self._rect_poly_and_cl()
        buf_full = compute_proportional_buffer(poly, cl, buffer_ratio=1.0, sample_distance=1.0, clip_to_polygon=True)
        buf_half = compute_proportional_buffer(poly, cl, buffer_ratio=0.5, sample_distance=1.0, clip_to_polygon=True)
        assert buf_half.area < buf_full.area

    def test_no_clip_can_extend_outside_polygon(self):
        """Without clipping, the round end caps may extend outside the polygon."""
        poly, cl = self._rect_poly_and_cl()
        buf = compute_proportional_buffer(
            poly, cl,
            buffer_ratio=0.8,
            sample_distance=1.0,
            clip_to_polygon=False,
            end_cap="round",
        )
        # With round caps, the buffer may extend slightly beyond the polygon
        # It should at least cover the clipped version
        buf_clipped = compute_proportional_buffer(
            poly, cl,
            buffer_ratio=0.8,
            sample_distance=1.0,
            clip_to_polygon=True,
            end_cap="round",
        )
        assert buf.area >= buf_clipped.area - 1e-6

    # ── End-cap styles ───────────────────────────────────────────────────────

    def test_round_cap_larger_than_flat_cap(self):
        """Round end-caps produce a larger area than flat end-caps."""
        poly, cl = self._rect_poly_and_cl()
        buf_round = compute_proportional_buffer(
            poly, cl, buffer_ratio=0.8, sample_distance=1.0, end_cap="round", clip_to_polygon=False
        )
        buf_flat = compute_proportional_buffer(
            poly, cl, buffer_ratio=0.8, sample_distance=1.0, end_cap="flat", clip_to_polygon=False
        )
        assert buf_round.area >= buf_flat.area - 1e-6

    # ── Width clamping ───────────────────────────────────────────────────────

    def test_width_min_prevents_zero_area(self):
        """A centerline near the polygon tip has near-zero half-width;
        width_min should prevent a degenerate output."""
        poly = Polygon([(0, 0), (100, 1), (100, -1)])  # very pointy triangle
        cl = LineString([(1, 0), (90, 0)])
        buf = compute_proportional_buffer(
            poly, cl,
            buffer_ratio=0.5,
            sample_distance=1.0,
            width_min=0.5,
            clip_to_polygon=True,
        )
        assert buf.area > 0

    def test_width_max_limits_buffer_area(self):
        """Applying width_max should produce a smaller buffer than without it."""
        poly, cl = self._rect_poly_and_cl()
        buf_unlimited = compute_proportional_buffer(
            poly, cl, buffer_ratio=1.0, sample_distance=1.0, clip_to_polygon=True
        )
        buf_limited = compute_proportional_buffer(
            poly, cl, buffer_ratio=1.0, sample_distance=1.0,
            width_max=4.0, clip_to_polygon=True
        )
        assert buf_limited.area <= buf_unlimited.area + 1e-6

    # ── MultiLineString ──────────────────────────────────────────────────────

    def test_multilinestring_centerline(self):
        """A forked centerline (MultiLineString) should produce a valid union."""
        poly = Polygon([(-50, -5), (50, -5), (50, 5), (-50, 5)])
        cl = MultiLineString([
            [(-50, 0), (0, 0)],
            [(0, 0), (50, 2)],
        ])
        buf = compute_proportional_buffer(poly, cl, buffer_ratio=0.5, sample_distance=1.0)
        assert not buf.is_empty

    # ── Smoothing ────────────────────────────────────────────────────────────

    def test_smooth_tolerance_produces_valid_geometry(self):
        poly, cl = self._rect_poly_and_cl()
        buf = compute_proportional_buffer(
            poly, cl,
            buffer_ratio=0.5,
            sample_distance=1.0,
            smooth_tolerance=2.0,
        )
        assert buf is not None
        assert not buf.is_empty

    # ── Automatic sample distance ────────────────────────────────────────────

    def test_auto_sample_distance_produces_valid_result(self):
        poly, cl = self._rect_poly_and_cl()
        buf = compute_proportional_buffer(poly, cl, buffer_ratio=0.5, sample_distance=None)
        assert not buf.is_empty

    # ── Error handling ───────────────────────────────────────────────────────

    def test_empty_polygon_raises(self):
        cl = LineString([(0, 0), (1, 0)])
        with pytest.raises(ValueError, match="empty"):
            compute_proportional_buffer(Polygon(), cl)

    def test_empty_centerline_raises(self):
        poly = Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])
        with pytest.raises(ValueError, match="empty"):
            compute_proportional_buffer(poly, LineString())


# ===========================================================================
# GeoDataFrame interface tests
# ===========================================================================


@requires_geopandas
class TestProcessGeoDataFrames:
    @staticmethod
    def _make_gdfs():
        """Create minimal GeoDataFrames for a single polygon / centerline pair."""
        from shapely.geometry import Polygon, LineString

        poly_gdf = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[Polygon([(-50, -5), (50, -5), (50, 5), (-50, 5)])],
            crs="EPSG:32650",
        )
        cl_gdf = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[LineString([(-50, 0), (50, 0)])],
            crs="EPSG:32650",
        )
        return poly_gdf, cl_gdf

    def test_returns_geodataframe(self):
        poly_gdf, cl_gdf = self._make_gdfs()
        out = process_geodataframes(poly_gdf, cl_gdf, buffer_ratio=0.5)
        assert isinstance(out, gpd.GeoDataFrame)

    def test_output_has_expected_columns(self):
        poly_gdf, cl_gdf = self._make_gdfs()
        out = process_geodataframes(poly_gdf, cl_gdf, buffer_ratio=0.5)
        for col in ("src_id", "buffer_ratio", "width_min_m", "width_max_m", "width_mean_m"):
            assert col in out.columns

    def test_output_crs_matches_input(self):
        poly_gdf, cl_gdf = self._make_gdfs()
        out = process_geodataframes(poly_gdf, cl_gdf, buffer_ratio=0.5)
        assert out.crs == poly_gdf.crs

    def test_width_statistics_are_positive(self):
        poly_gdf, cl_gdf = self._make_gdfs()
        out = process_geodataframes(poly_gdf, cl_gdf, buffer_ratio=0.5)
        assert len(out) >= 1
        # width_min_m can be 0 when the centerline touches the polygon boundary
        assert (out["width_min_m"] >= 0).all()
        assert (out["width_max_m"] > 0).all()
        assert (out["width_max_m"] >= out["width_min_m"]).all()
        assert (out["width_mean_m"] >= out["width_min_m"]).all()

    def test_buffer_ratio_stored_correctly(self):
        poly_gdf, cl_gdf = self._make_gdfs()
        out = process_geodataframes(poly_gdf, cl_gdf, buffer_ratio=0.75)
        assert (out["buffer_ratio"] == 0.75).all()

    def test_no_match_returns_empty(self):
        from shapely.geometry import Polygon, LineString

        poly_gdf = gpd.GeoDataFrame(
            geometry=[Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])],
            crs="EPSG:32650",
        )
        cl_gdf = gpd.GeoDataFrame(
            geometry=[LineString([(100, 0), (200, 0)])],  # far away
            crs="EPSG:32650",
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = process_geodataframes(poly_gdf, cl_gdf)
        assert len(out) == 0

    def test_crs_mismatch_auto_reprojects(self):
        """CRS mismatch between poly and centerline should be handled gracefully."""
        from shapely.geometry import Polygon, LineString
        import pyproj

        poly_gdf = gpd.GeoDataFrame(
            geometry=[Polygon([(-50, -5), (50, -5), (50, 5), (-50, 5)])],
            crs="EPSG:32650",
        )
        # Centerline in a different projected CRS – this will not align spatially,
        # but the function should not raise on CRS mismatch (it should reproject).
        cl_gdf = gpd.GeoDataFrame(
            geometry=[LineString([(-50, 0), (50, 0)])],
            crs="EPSG:32651",
        )
        try:
            out = process_geodataframes(poly_gdf, cl_gdf)
            # Result may be empty if geometries don't overlap after reprojection,
            # but no CRS exception should be raised.
            assert isinstance(out, gpd.GeoDataFrame)
        except Exception as exc:
            pytest.fail(f"CRS mismatch caused an unexpected exception: {exc}")
