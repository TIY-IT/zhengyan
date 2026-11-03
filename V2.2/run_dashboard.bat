@echo off
rem Double-click to open the dashboard (uses the GPU Python from the README; edit if yours differs).
set GPU_PY=D:\DL_Environment\pytorch\.venv\Scripts\python.exe
cd /d "%~dp0"
if exist "%GPU_PY%" ( "%GPU_PY%" dashboard.py ) else ( python dashboard.py )
pause
