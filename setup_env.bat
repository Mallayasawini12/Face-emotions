@echo off
REM Create virtual environment and install dependencies (cmd)
where python >nul 2>&1
if %errorlevel% neq 0 (
  echo Python not found. Install Python 3.10+ and add to PATH.
  exit /b 1
)
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo Virtualenv created and dependencies installed. Activate with: .venv\Scripts\activate.bat