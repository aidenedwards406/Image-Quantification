@echo off
setlocal
cd /d "%~dp0"
title HUVEC Damage Analyzer - setup
echo.
echo  HUVEC Damage Analyzer - one-time setup
echo  ======================================
echo  Downloads Python and the libraries the app needs (about 1 GB, 5-15 minutes).
echo  Everything goes into this folder and your user account. No admin rights needed.
echo.

set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"
where uv >nul 2>nul
if errorlevel 1 (
  echo  Installing uv, a small Python installer from astral.sh ...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
)
where uv >nul 2>nul
if errorlevel 1 (
  echo.
  echo  Could not install uv. Check the internet connection and run install.bat again,
  echo  or ask whoever maintains the app for help.
  pause
  exit /b 1
)

echo.
echo  Setting up Python 3.12 for the app ...
uv venv .venv --python 3.12 --allow-existing
if errorlevel 1 goto :fail

echo.
echo  Installing libraries (PyTorch is the big one) ...
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
if errorlevel 1 goto :fail

echo.
echo  Checking the installation ...
.venv\Scripts\python.exe -c "import torch, streamlit, roifile, tifffile; print('  OK - PyTorch', torch.__version__, '/ Streamlit', streamlit.__version__)"
if errorlevel 1 goto :fail

echo.
echo  Adding a "HUVEC Damage Analyzer" shortcut to the desktop ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$d=[Environment]::GetFolderPath('Desktop'); $s=(New-Object -ComObject WScript.Shell).CreateShortcut(\"$d\HUVEC Damage Analyzer.lnk\"); $s.TargetPath='%~dp0start.bat'; $s.WorkingDirectory='%~dp0'; $s.IconLocation='%SystemRoot%\System32\imageres.dll,109'; $s.Save()" >nul 2>nul

echo.
echo  ============================================================
echo   Setup finished. Double-click "HUVEC Damage Analyzer" on the
echo   desktop (or start.bat in this folder) to open the app.
echo  ============================================================
pause
exit /b 0

:fail
echo.
echo  Setup did not finish. Scroll up for the error message and send a
echo  screenshot of this window to whoever maintains the app.
pause
exit /b 1
