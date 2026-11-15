@echo off
cd /d "%~dp0"
rem Ignore global pip config (it points at an unreachable NVIDIA index)
set PIP_CONFIG_FILE=nul
set PIP_EXTRA_INDEX_URL=
set PIP_INDEX_URL=https://pypi.org/simple
echo === AVIS setup: creating Python environment (first run only) ===
if not exist .venv\Scripts\python.exe (
  py -3.12 -m venv .venv 2>nul || py -3.13 -m venv .venv 2>nul || py -3.11 -m venv .venv 2>nul || python -m venv .venv
)
if not exist .venv\Scripts\python.exe (
  echo ERROR: could not create a virtual environment. Is Python installed?
  pause
  exit /b 1
)
.venv\Scripts\python --version
echo === Installing packages (first run downloads PyTorch, a few minutes) ===
.venv\Scripts\python -m pip --isolated install -q -U pip --index-url https://pypi.org/simple
.venv\Scripts\python -m pip --isolated install -r requirements.txt --index-url https://pypi.org/simple
if errorlevel 1 (
  echo ERROR: package install failed. Copy the red text above and send it to Claude.
  pause
  exit /b 1
)
if "%~1"=="--setup-only" ( echo === Setup complete. === & exit /b 0 )
echo === Starting live camera feed. Click the AVIS window: E = enroll a face, Q = quit. ===
.venv\Scripts\python realtime_yolo.py --camera 0 %*
