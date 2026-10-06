@echo off
cd /d "%~dp0"

echo ==========================================
echo CritAlert Startup Script
echo ==========================================

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python is not installed or not added to PATH.
    echo Please install Python 3.10 or 3.11 and try again.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
)

call ".venv\Scripts\activate.bat"

echo Installing project requirements...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo Starting CritAlert server...
python app.py

pause
