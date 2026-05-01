# -*- coding: utf-8 -*-
"""
cli.py
======
Command-line interface for the proportional buffer tool.

Usage
-----
::

    python cli.py --polygon  waterways.gpkg --polygon-layer  polygons  \\
                  --centerline waterways.gpkg --centerline-layer centerlines \\
                  --output  buffers.gpkg  --output-layer buffers             \\
                  --ratio 0.5

Run ``python cli.py --help`` for the full list of options.
"""

from __future__ import annotations

import argparse
import sys
import os


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="proportional_buffer",
        description=(
            "Compute variable-width proportional buffers along pre-existing "
            "centerlines.\n\n"
            "Inputs : (1) waterway polygon layer, (2) pre-computed centerline "
            "layer.\n"
            "Output : buffer polygons whose width at every cross-section equals "
            "RATIO × 2 × local_channel_width."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Input / output ───────────────────────────────────────────────────────
    p.add_argument(
        "--polygon",
        required=True,
        metavar="PATH",
        help="Path to the input polygon file (GeoJSON, Shapefile, GeoPackage, …).",
    )
    p.add_argument(
        "--polygon-layer",
        default=None,
        metavar="LAYER",
        help="Layer name (for multi-layer formats such as GeoPackage).",
    )
    p.add_argument(
        "--centerline",
        required=True,
        metavar="PATH",
        help="Path to the input centerline file.",
    )
    p.add_argument(
        "--centerline-layer",
        default=None,
        metavar="LAYER",
        help="Layer name for the centerline file.",
    )
    p.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Path for the output buffer file.",
    )
    p.add_argument(
        "--output-layer",
        default="proportional_buffer",
        metavar="LAYER",
        help="Layer name to write in the output file (default: proportional_buffer).",
    )

    # ── Algorithm parameters ─────────────────────────────────────────────────
    p.add_argument(
        "--ratio",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help=(
            "Buffer ratio: fraction of the local half-width used as buffer "
            "radius on each side.  0 < ratio ≤ 1.  Default: 0.5."
        ),
    )
    p.add_argument(
        "--sample-distance",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Centerline sampling interval (CRS units).  "
            "Omit for automatic selection."
        ),
    )
    p.add_argument(
        "--end-cap",
        choices=["round", "flat"],
        default="round",
        help="End-cap style at centerline termini (default: round).",
    )
    p.add_argument(
        "--no-clip",
        action="store_true",
        default=False,
        help=(
            "Do NOT clip the buffer to the original polygon boundary.  "
            "By default, clipping is applied."
        ),
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=0,
        metavar="N",
        help="Number of Chaikin smoothing iterations (0 = no smoothing).",
    )
    p.add_argument(
        "--width-min",
        type=float,
        default=0.0,
        metavar="FLOAT",
        help="Minimum total buffer width (CRS units).  Default: 0.",
    )
    p.add_argument(
        "--width-max",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Maximum total buffer width (CRS units).  Default: no limit.",
    )

    # ── Field names ──────────────────────────────────────────────────────────
    p.add_argument(
        "--poly-id-field",
        default=None,
        metavar="FIELD",
        help="Attribute field in the polygon layer to use as src_id in the output.",
    )
    p.add_argument(
        "--cl-id-field",
        default=None,
        metavar="FIELD",
        help="Attribute field in the centerline layer (used only for log messages).",
    )

    return p


def main(argv=None) -> int:
    """
    Entry point for the command-line interface.

    Returns
    -------
    int
        Exit code: 0 on success, non-zero on failure.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ── Lazy imports (keep startup fast) ────────────────────────────────────
    try:
        import geopandas as gpd
    except ImportError:
        print(
            "ERROR: geopandas is required.  Install it with:\n"
            "    pip install geopandas",
            file=sys.stderr,
        )
        return 1

    # Import the core module from the same directory as this script
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)

    from proportional_buffer import process_geodataframes

    # ── Read inputs ──────────────────────────────────────────────────────────
    print(f"Reading polygon layer:    {args.polygon}")
    try:
        poly_gdf = gpd.read_file(args.polygon, layer=args.polygon_layer)
    except Exception as exc:
        print(f"ERROR reading polygon file: {exc}", file=sys.stderr)
        return 1

    print(f"Reading centerline layer: {args.centerline}")
    try:
        cl_gdf = gpd.read_file(args.centerline, layer=args.centerline_layer)
    except Exception as exc:
        print(f"ERROR reading centerline file: {exc}", file=sys.stderr)
        return 1

    print(
        f"  Polygons  : {len(poly_gdf)} features  (CRS: {poly_gdf.crs})\n"
        f"  Centerlines: {len(cl_gdf)} features  (CRS: {cl_gdf.crs})"
    )

    # ── Run algorithm ────────────────────────────────────────────────────────
    width_max = args.width_max if args.width_max is not None else float("inf")

    print(
        f"\nParameters:\n"
        f"  buffer_ratio    = {args.ratio}\n"
        f"  sample_distance = {args.sample_distance or 'auto'}\n"
        f"  end_cap         = {args.end_cap}\n"
        f"  clip_to_polygon = {not args.no_clip}\n"
        f"  smooth          = {args.smooth} iteration(s)\n"
        f"  width_min       = {args.width_min}\n"
        f"  width_max       = {width_max}\n"
    )

    print("Computing proportional buffers …")
    try:
        out_gdf = process_geodataframes(
            poly_gdf=poly_gdf,
            cl_gdf=cl_gdf,
            poly_id_field=args.poly_id_field,
            cl_id_field=args.cl_id_field,
            buffer_ratio=args.ratio,
            sample_distance=args.sample_distance,
            end_cap=args.end_cap,
            clip_to_polygon=not args.no_clip,
            smooth_tolerance=float(args.smooth),
            width_min=args.width_min,
            width_max=width_max,
        )
    except Exception as exc:
        print(f"ERROR during processing: {exc}", file=sys.stderr)
        return 1

    if len(out_gdf) == 0:
        print("WARNING: No buffer features were produced.", file=sys.stderr)
        return 0

    # ── Write output ─────────────────────────────────────────────────────────
    print(f"Writing {len(out_gdf)} feature(s) to: {args.output}")
    try:
        out_gdf.to_file(args.output, layer=args.output_layer, driver=_infer_driver(args.output))
    except Exception as exc:
        print(f"ERROR writing output: {exc}", file=sys.stderr)
        return 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(
        f"\nDone.  {len(out_gdf)} buffer feature(s) written.\n"
        f"  Mean channel width range: "
        f"{out_gdf['width_min_m'].min():.1f} – {out_gdf['width_max_m'].max():.1f} "
        f"(CRS units)"
    )
    return 0


def _infer_driver(path: str) -> str:
    """Guess the OGR driver from the file extension."""
    ext = os.path.splitext(path)[1].lower()
    return {
        ".gpkg": "GPKG",
        ".geojson": "GeoJSON",
        ".json": "GeoJSON",
        ".shp": "ESRI Shapefile",
    }.get(ext, "GPKG")


if __name__ == "__main__":
    sys.exit(main())
