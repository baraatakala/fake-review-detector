@echo off
cd /d "%~dp0"
echo Starting Fake Review Detector...
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH.
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

echo Installing required packages...
python -m pip install flask scikit-learn pandas numpy nltk

echo.
echo Starting Flask application...
echo Open your browser to: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

python app.py

pause
