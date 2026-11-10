@echo off
rem First time only: makes a Python environment with the CPU version of PyTorch (~200 MB download).
cd /d "%~dp0"
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
echo.
echo Gata. Acum porneste start.bat
pause
