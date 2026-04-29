@echo off
rem ============================================================
rem  install_dependencies.bat
rem  Helper script to install the open-source Python packages
rem  required by Degree_Centerline.pyt inside ArcGIS Pro.
rem
rem  PACKAGES REQUIRED
rem  -----------------
rem  networkx  — the only package that needs to be installed.
rem  numpy, scipy, and matplotlib ship with every ArcGIS Pro Python
rem  environment; no extra install is needed for those.
rem
rem  HOW TO RUN
rem  ----------
rem  Option A (recommended):
rem    Start > ArcGIS > ArcGIS Pro Python Command Prompt
rem    cd to the degree_centerline folder and run:
rem      install_dependencies.bat
rem
rem  Option B:
rem    Double-click this file in Windows Explorer.
rem
rem  ARCGIS PRO NOTE
rem  ---------------
rem  The default "arcgispro-py3" conda environment is read-only when
rem  ArcGIS is installed in "C:\Program Files\".  If the install
rem  fails, clone the environment first:
rem
rem    1. Open the ArcGIS Pro Python Command Prompt.
rem    2. conda create --name arcgispro-py3-degree --clone arcgispro-py3
rem    3. activate arcgispro-py3-degree
rem    4. Re-run this script.
rem    5. In ArcGIS Pro: Project > Python > Python Environments
rem       Select "arcgispro-py3-degree" and restart ArcGIS Pro.
rem ============================================================

setlocal

echo ============================================================
echo  Degree Centerline -- Dependency Installer
echo ============================================================
echo.
echo This toolbox needs only ONE third-party package: networkx
echo (numpy, scipy, and matplotlib are already present in ArcGIS Pro).
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: 'python' was not found on PATH.
    echo Please run this script from the ArcGIS Pro Python Command Prompt.
    goto :end_error
)

where conda >nul 2>&1
if errorlevel 1 (
    echo WARNING: 'conda' was not found on PATH.
    echo Falling back to pip.
    goto :pip_install
)

echo Installing networkx via conda (conda-forge) ...
conda install -c conda-forge -y "networkx>=3.0"

if errorlevel 1 (
    echo.
    echo WARNING: conda install returned an error.
    echo The current environment may be read-only.  Clone it first:
    echo   conda create --name arcgispro-py3-degree --clone arcgispro-py3
    echo   activate arcgispro-py3-degree
    echo   install_dependencies.bat
    goto :pip_fallback
)

echo.
echo conda install completed successfully.
goto :verify

:pip_install
echo Installing networkx via pip ...
python -m pip install --upgrade "networkx>=3.0"
if errorlevel 1 (
    echo ERROR: pip install failed.  See README.md for troubleshooting.
    goto :end_error
)
echo pip install completed successfully.
goto :verify

:pip_fallback
echo Attempting pip as fallback ...
python -m pip install --upgrade "networkx>=3.0"
if errorlevel 1 (
    echo ERROR: pip install also failed.
    goto :end_error
)

:verify
echo ============================================================
echo  Verifying installation ...
echo ============================================================
python -c "import numpy; import scipy; import networkx; print('All required packages are available.')"
if errorlevel 1 (
    echo WARNING: Verification failed.  See messages above.
    goto :end_error
)

echo.
echo ============================================================
echo  SUCCESS!  All dependencies are installed.
echo  You can now run the Degree Centerline toolbox in ArcGIS Pro.
echo ============================================================
echo.
pause
exit /b 0

:end_error
echo.
echo ============================================================
echo  Installation incomplete.  See: degree_centerline/README.md
echo ============================================================
echo.
pause
exit /b 1
