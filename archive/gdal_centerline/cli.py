#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cli.py — Command-line interface for the open-source centerline tool.

Usage examples
--------------
    # Voronoi method (vector-based, default)
    python cli.py input.geojson output.geojson

    # Skeleton method (raster-based)
    python cli.py input.geojson output.geojson --method skeleton

    # Adjust densification and pruning
    python cli.py rivers.gpkg centerlines.gpkg --densify 0.5 --prune 5.0

    # Raster skeleton with Gaussian smoothing
    python cli.py roads.shp roads_cl.shp --method skeleton --smooth 2.0

Run `python cli.py --help` for all options.
"""

import argparse
import sys

from centerline import polygon_to_centerline, read_polygons, write_centerlines


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="centerline",
        description=(
            "Convert elongated polygon features to centerline polylines.\n\n"
            "Two methods are available:\n"
            "  voronoi  — vector-based Voronoi / Thiessen skeleton (default)\n"
            "  skeleton — raster-based morphological skeletonisation"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", help="Path to the input polygon vector file (GeoJSON, Shapefile, GPKG, …)")
    p.add_argument("output", help="Path for the output centerline vector file")
    p.add_argument(
        "--method",
        choices=["voronoi", "skeleton"],
        default="voronoi",
        help="Algorithm to use (default: voronoi)",
    )
    p.add_argument(
        "--densify",
        type=float,
        default=1.0,
        metavar="DISTANCE",
        help="Max vertex spacing for boundary densification [voronoi] or "
             "raster cell size [skeleton] in CRS units (default: 1.0)",
    )
    p.add_argument(
        "--prune",
        type=float,
        default=0.0,
        metavar="LENGTH",
        help="Prune skeleton branches shorter than this length [voronoi only] (default: 0 = no pruning)",
    )
    p.add_argument(
        "--smooth",
        type=float,
        default=0.0,
        metavar="SIGMA",
        help="Gaussian smoothing sigma before skeletonisation [skeleton only] (default: 0 = no smoothing)",
    )
    p.add_argument(
        "--resolution",
        type=float,
        default=None,
        metavar="CELLSIZE",
        help="Raster cell size override [skeleton only]; defaults to --densify value",
    )
    p.add_argument(
        "--driver",
        default=None,
        metavar="DRIVER",
        help="Output file driver (e.g. GeoJSON, 'ESRI Shapefile', GPKG). "
             "Auto-detected from extension if not specified.",
    )
    p.add_argument(
        "--multi-line",
        dest="multi_line",
        action="store_true",
        default=False,
        help="Return the full skeleton (may contain forks / branches). "
             "By default a single non-branching line is returned.",
    )
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    print(f"Reading input: {args.input}")
    gdf = read_polygons(args.input)
    print(f"  {len(gdf)} feature(s) read.")

    print(f"Computing centerlines (method={args.method}, densify={args.densify}) …")
    centerlines = polygon_to_centerline(
        gdf,
        method=args.method,
        densify_distance=args.densify,
        prune_threshold=args.prune,
        smooth_sigma=args.smooth,
        raster_resolution=args.resolution,
        single_line=not args.multi_line,
    )
    print(f"  {len(centerlines)} centerline(s) generated.")

    print(f"Writing output: {args.output}")
    write_centerlines(centerlines, args.output, driver=args.driver)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
