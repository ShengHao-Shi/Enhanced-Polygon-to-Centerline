@echo off
rem ============================================================
rem  install_dependencies.bat
rem  Helper script to install the open-source Python packages
rem  required by GDAL_Centerline.pyt inside ArcGIS Pro.
rem
rem  HOW TO RUN
rem  ----------
rem  Option A (recommended):
rem    Start > ArcGIS > ArcGIS Pro Python Command Prompt
rem    cd to the gdal_centerline folder and run:
rem      install_dependencies.bat
rem
rem  Option B:
rem    Double-click this file in Windows Explorer.
rem    (Requires ArcGIS Pro Python Command Prompt to be configured
rem     or the conda/python commands to be on your PATH.)
rem
rem  ARCGIS PRO NOTE
rem  ---------------
rem  The default "arcgispro-py3" conda environment shipped with
rem  ArcGIS Pro is read-only when ArcGIS is installed in
rem  "C:\Program Files\".  If the install below fails with a
rem  "Read-only" or "CondaError" message, follow these steps first:
rem
rem    1. Open the ArcGIS Pro Python Command Prompt.
rem    2. Clone the default environment:
rem         conda create --name arcgispro-py3-gdal --clone arcgispro-py3
rem    3. Activate it (only needed in this shell session):
rem         activate arcgispro-py3-gdal
rem    4. Re-run this script.
rem    5. In ArcGIS Pro go to:
rem         Project > Python > Python Environments
rem         Select "arcgispro-py3-gdal" and click OK.
rem       Then restart ArcGIS Pro.
rem ============================================================

setlocal

echo ============================================================
echo  GDAL Centerline — Dependency Installer
echo ============================================================
echo.

rem ---- Detect Python / Conda -----------------------------------------
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: 'python' was not found on PATH.
    echo Please run this script from the ArcGIS Pro Python Command Prompt.
    echo   Start ^> ArcGIS ^> ArcGIS Pro Python Command Prompt
    goto :end_error
)

where conda >nul 2>&1
if errorlevel 1 (
    echo WARNING: 'conda' was not found on PATH.
    echo Falling back to pip for installation.
    goto :pip_install
)

rem ---- Try conda install ---------------------------------------------
echo Installing required packages via conda (conda-forge channel) ...
echo This may take several minutes on the first run.
echo.
echo Required (always installed):
echo   shapely, geopandas, scipy, numpy, networkx
echo Optional (installed here for convenience; needed only for method=skeleton):
echo   scikit-image
echo.

conda install -c conda-forge -y ^
    "shapely>=2.0" ^
    "geopandas>=0.14" ^
    "scipy>=1.10" ^
    "numpy>=1.24" ^
    "networkx>=3.0" ^
    "scikit-image>=0.21"

if errorlevel 1 (
    echo.
    echo WARNING: conda install returned an error.
    echo This may be because the current environment is read-only.
    echo.
    echo To fix, open the ArcGIS Pro Python Command Prompt and run:
    echo   conda create --name arcgispro-py3-gdal --clone arcgispro-py3
    echo   activate arcgispro-py3-gdal
    echo   install_dependencies.bat
    echo.
    echo Then in ArcGIS Pro:
    echo   Project ^> Python ^> Python Environments ^> arcgispro-py3-gdal
    echo.
    goto :pip_fallback
)

echo.
echo conda install completed successfully.
goto :verify

:pip_install
echo.
echo Installing packages via pip ...
python -m pip install --upgrade ^
    "shapely>=2.0" ^
    "geopandas>=0.14" ^
    "scipy>=1.10" ^
    "numpy>=1.24" ^
    "networkx>=3.0" ^
    "scikit-image>=0.21"

if errorlevel 1 (
    echo.
    echo ERROR: pip install also failed.
    echo Please see the troubleshooting section in README.md.
    goto :end_error
)

echo.
echo pip install completed successfully.
goto :verify

:pip_fallback
echo Attempting pip install as fallback ...
python -m pip install --upgrade ^
    "shapely>=2.0" ^
    "geopandas>=0.14" ^
    "scipy>=1.10" ^
    "numpy>=1.24" ^
    "networkx>=3.0" ^
    "scikit-image>=0.21"

if errorlevel 1 (
    echo.
    echo ERROR: pip install also failed. Please clone the environment
    echo as described above and re-run this script.
    goto :end_error
)

:verify
echo ============================================================
echo  Verifying installation ...
echo ============================================================
python -c "import shapely; import geopandas; import scipy; import networkx; print('All required packages are available.')"
if errorlevel 1 (
    echo.
    echo WARNING: Verification failed — some packages may not have
    echo installed correctly.  See messages above for details.
    goto :end_error
)

echo.
echo ============================================================
echo  SUCCESS!  All dependencies are installed.
echo  You can now run the GDAL Centerline toolbox in ArcGIS Pro.
echo ============================================================
echo.
pause
exit /b 0

:end_error
echo.
echo ============================================================
echo  Installation incomplete.  Please read the messages above.
echo  For full instructions see: gdal_centerline/README.md
echo ============================================================
echo.
pause
exit /b 1
