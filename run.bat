@echo off
rem Double-click to start Simulatics Studio. "run.bat exe" runs the released exe, "run.bat web" the browser GUI.
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (py -3 run.py %*) else (python run.py %*)
if errorlevel 1 (
  echo.
  echo Simulatics stopped with an error. If Python is missing, install it from https://www.python.org/downloads/
  pause
)
