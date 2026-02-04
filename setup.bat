@echo off
REM DeepFake Detector - Windows Setup Script
REM ==========================================

echo ============================================
echo   DeepFake Detector - Setup Script
echo ============================================
echo.

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.9 or higher from https://python.org
    pause
    exit /b 1
)

echo [1/5] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)

echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/5] Upgrading pip...
python -m pip install --upgrade pip

echo [4/5] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

echo [5/5] Creating directories...
if not exist "uploads" mkdir uploads
if not exist "models" mkdir models
if not exist "results" mkdir results
if not exist "static\heatmaps" mkdir static\heatmaps

echo.
echo ============================================
echo   Setup Complete!
echo ============================================
echo.
echo To start the application:
echo   1. Activate environment: venv\Scripts\activate
echo   2. Initialize models: python download_models.py
echo   3. Run application: python app.py
echo.
echo Then open http://localhost:5000 in your browser
echo.
pause
