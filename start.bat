@echo off
echo ============================================
echo   Crop Leaf Disease Generator
echo   Starting Streamlit App...
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed!
    echo Please install Python 3.10+ from python.org
    pause
    exit /b 1
)

REM Install dependencies if needed
echo Checking dependencies...
pip show streamlit >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo Starting app...
echo Press Ctrl+C to stop
echo.

cd /d "%~dp0"
python -m streamlit run app.py

pause
