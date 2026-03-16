# -*- coding: utf-8 -*-
"""
Polygon_to_Centerline.pyt

An ArcGIS Python Toolbox that extracts centerlines from elongated polygon
features using a Voronoi (Thiessen) diagram approach.

Requirements:
    - ArcGIS Pro 2.x+ or ArcGIS Desktop 10.x with Standard/Advanced license
    - arcpy (bundled with ArcGIS)

Algorithm:
    1. Densify polygon boundary to increase vertex density along edges.
    2. Extract the densified boundary vertices as point features.
    3. Create Thiessen (Voronoi) polygons from those boundary points.
    4. Clip the Thiessen polygons to the extent of the original polygon.
    5. Convert the clipped Thiessen polygons to lines; select only the
       interior edges (shared between two Thiessen cells) — these form
       the skeleton / centerline of the original polygon.
    6. Optionally smooth the resulting centerline with the PAEK algorithm.
"""

import arcpy
import os


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the .pyt file)."""
        self.label = "Polygon to Centerline"
        self.alias = "PolygonToCenterline"
        self.tools = [PolygonToCenterline]


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

class PolygonToCenterline(object):
    def __init__(self):
        """Define the tool properties."""
        self.label = "Polygon to Centerline"
        self.description = (
            "Converts elongated polygon features to polyline centerlines "
            "using a Voronoi (Thiessen) diagram approach.\n\n"
            "The tool densifies the polygon boundary, creates Thiessen "
            "polygons from the boundary vertices, clips them to the original "
            "polygon, and extracts the interior Thiessen edges as the "
            "centerline."
        )
        self.canRunInBackground = False

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def getParameterInfo(self):
        """Define tool parameters."""

        # Input polygon feature class
        param_in = arcpy.Parameter(
            displayName="Input Polygon Features",
            name="in_features",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input",
        )
        param_in.filter.list = ["Polygon"]

        # Output polyline feature class
        param_out = arcpy.Parameter(
            displayName="Output Centerline Features",
            name="out_features",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Output",
        )

        # Densification distance (controls skeleton resolution)
        param_densify = arcpy.Parameter(
            displayName="Densification Distance",
            name="densify_distance",
            datatype="GPLinearUnit",
            parameterType="Optional",
            direction="Input",
        )
        param_densify.value = "1 Meters"

        # Optional PAEK smoothing tolerance
        param_smooth = arcpy.Parameter(
            displayName="Smoothing Tolerance (0 = no smoothing)",
            name="smooth_tolerance",
            datatype="GPLinearUnit",
            parameterType="Optional",
            direction="Input",
        )
        param_smooth.value = "0 Meters"

        return [param_in, param_out, param_densify, param_smooth]

    # ------------------------------------------------------------------
    # Licensing
    # ------------------------------------------------------------------

    def isLicensed(self):
        """Allow the tool to execute only if a Standard/Advanced license is available."""
        try:
            return arcpy.GetInstallInfo()["LicenseType"] in ("Standard", "Advanced")
        except Exception:
            return True  # Let execution surface the error naturally

    # ------------------------------------------------------------------
    # Validation hooks (unused but required by the interface)
    # ------------------------------------------------------------------

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, parameters, messages):
        """Run the tool."""
        in_features = parameters[0].valueAsText
        out_features = parameters[1].valueAsText
        densify_distance = (
            parameters[2].valueAsText if parameters[2].value else "1 Meters"
        )
        smooth_tolerance = (
            parameters[3].valueAsText if parameters[3].value else "0 Meters"
        )

        arcpy.env.overwriteOutput = True
        workspace = arcpy.env.scratchGDB

        try:
            _extract_centerline(
                in_features,
                out_features,
                densify_distance,
                smooth_tolerance,
                workspace,
                messages,
            )
        except arcpy.ExecuteError:
            messages.addErrorMessage(arcpy.GetMessages(2))
            raise
        except Exception as exc:
            messages.addErrorMessage(str(exc))
            raise

    def postExecute(self, parameters):
        """Called after the tool execution."""
        return


# ---------------------------------------------------------------------------
# Core algorithm (also importable / callable from other Python scripts)
# ---------------------------------------------------------------------------

def _extract_centerline(
    in_features,
    out_features,
    densify_distance="1 Meters",
    smooth_tolerance="0 Meters",
    workspace=None,
    messages=None,
):
    """
    Extract centerlines from polygon features.

    Parameters
    ----------
    in_features : str
        Path to the input polygon feature class or layer.
    out_features : str
        Path for the output polyline feature class.
    densify_distance : str
        Distance used to densify polygon boundaries, e.g. ``"1 Meters"``.
        Smaller values produce denser skeletons at the cost of longer
        processing time.
    smooth_tolerance : str
        PAEK smoothing tolerance for the output centerline, e.g.
        ``"5 Meters"``.  Set to ``"0 Meters"`` (the default) to skip
        smoothing.
    workspace : str or None
        Geodatabase path used for temporary datasets.  Defaults to
        ``arcpy.env.scratchGDB``.
    messages : arcpy.Messages or None
        ArcGIS tool messages object.  If *None* ``arcpy.AddMessage`` is
        used directly.
    """
    if workspace is None:
        workspace = arcpy.env.scratchGDB

    def log(msg):
        if messages is not None:
            messages.addMessage(msg)
        else:
            arcpy.AddMessage(msg)

    # Helper: build a unique temp path to avoid stale-data collisions
    def tmp(name):
        return os.path.join(workspace, "ptc_" + name)

    arcpy.env.overwriteOutput = True

    # ------------------------------------------------------------------
    # Step 1 – Copy input so the original is never modified
    # ------------------------------------------------------------------
    log("Step 1/7  Copying input polygon features …")
    temp_polygon = tmp("polygon")
    arcpy.CopyFeatures_management(in_features, temp_polygon)

    # ------------------------------------------------------------------
    # Step 2 – Densify boundary to increase vertex density
    # ------------------------------------------------------------------
    log("Step 2/7  Densifying polygon boundaries …")
    arcpy.Densify_edit(temp_polygon, "DISTANCE", densify_distance)

    # ------------------------------------------------------------------
    # Step 3 – Extract every boundary vertex as a point
    # ------------------------------------------------------------------
    log("Step 3/7  Extracting boundary vertices …")
    temp_points = tmp("points")
    arcpy.FeatureVerticesToPoints_management(temp_polygon, temp_points, "ALL")

    # ------------------------------------------------------------------
    # Step 4 – Build Thiessen (Voronoi) polygons from the boundary points
    # ------------------------------------------------------------------
    log("Step 4/7  Creating Thiessen polygons …")
    temp_thiessen = tmp("thiessen")

    # Restrict Thiessen extent to the polygon extent so we do not
    # generate huge, unnecessary tessellations outside the study area.
    desc = arcpy.Describe(temp_polygon)
    arcpy.env.extent = desc.extent
    arcpy.CreateThiessenPolygons_analysis(temp_points, temp_thiessen, "ALL")
    arcpy.env.extent = None  # restore

    # ------------------------------------------------------------------
    # Step 5 – Clip Thiessen polygons to the original polygon footprint
    # ------------------------------------------------------------------
    log("Step 5/7  Clipping Thiessen polygons to polygon boundary …")
    temp_thiessen_clipped = tmp("thiessen_clipped")
    arcpy.Clip_analysis(temp_thiessen, temp_polygon, temp_thiessen_clipped)

    # ------------------------------------------------------------------
    # Step 6 – Convert clipped Thiessen polygons to lines and keep only
    #           interior edges (shared between two cells).
    # ------------------------------------------------------------------
    log("Step 6/7  Extracting interior skeleton edges …")
    temp_thiessen_lines = tmp("thiessen_lines")
    arcpy.PolygonToLine_management(
        temp_thiessen_clipped,
        temp_thiessen_lines,
        "IDENTIFY_AND_STORE_POLYGON_NEIGHBOR_INFORMATION",
    )

    # Interior edges: LEFT_FID >= 0 AND RIGHT_FID >= 0
    # Boundary edges (no neighbor on one side) have FID = -1 on that side.
    arcpy.MakeFeatureLayer_management(temp_thiessen_lines, "ptc_lyr")
    try:
        arcpy.SelectLayerByAttribute_management(
            "ptc_lyr",
            "NEW_SELECTION",
            "LEFT_FID >= 0 AND RIGHT_FID >= 0",
        )
        count = int(arcpy.GetCount_management("ptc_lyr").getOutput(0))
        if count == 0:
            raise RuntimeError(
                "No centerline features were generated. "
                "Try using a smaller Densification Distance."
            )
        log("         {:,} centerline segment(s) found.".format(count))

        temp_skeleton = tmp("skeleton")
        arcpy.CopyFeatures_management("ptc_lyr", temp_skeleton)
    finally:
        arcpy.Delete_management("ptc_lyr")

    # ------------------------------------------------------------------
    # Step 7 – Optional PAEK smoothing
    # ------------------------------------------------------------------
    smooth_value = _parse_linear_unit_value(smooth_tolerance)
    if smooth_value > 0:
        log(
            "Step 7/7  Smoothing centerline (PAEK, tolerance = {}) …".format(
                smooth_tolerance
            )
        )
        arcpy.SmoothLine_cartography(
            temp_skeleton, out_features, "PAEK", smooth_tolerance
        )
    else:
        log("Step 7/7  Copying centerline to output …")
        arcpy.CopyFeatures_management(temp_skeleton, out_features)

    # ------------------------------------------------------------------
    # Clean up temporary datasets
    # ------------------------------------------------------------------
    log("Cleaning up temporary data …")
    for _tmp in [
        temp_polygon,
        temp_points,
        temp_thiessen,
        temp_thiessen_clipped,
        temp_thiessen_lines,
        temp_skeleton,
    ]:
        try:
            arcpy.Delete_management(_tmp)
        except Exception:
            pass

    log("Done.  Output saved to: {}".format(out_features))


def _parse_linear_unit_value(linear_unit_str):
    """Return the numeric part of an ArcGIS linear unit string (e.g. '5 Meters' → 5.0)."""
    try:
        return float(str(linear_unit_str).split()[0])
    except (ValueError, IndexError, AttributeError):
        return 0.0
