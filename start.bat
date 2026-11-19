@echo off
cd /d "%~dp0"
title HUVEC Damage Analyzer
if not exist ".venv\Scripts\python.exe" (
  echo  The app isn't set up on this computer yet. Double-click install.bat first.
  pause
  exit /b 1
)
echo.
echo  Starting HUVEC Damage Analyzer - your browser opens in a moment.
echo  If it doesn't, go to  http://localhost:8501
echo.
echo  Keep this window open while you use the app; close it to stop the app.
echo  (Training and analysis that are already running keep going in the background.)
echo.
".venv\Scripts\python.exe" -m streamlit run app.py
pause
