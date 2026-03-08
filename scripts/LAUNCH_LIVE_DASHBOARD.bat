@echo off
echo ==========================================
echo  CYCLING PREDICT - LIVE DASHBOARD
echo ==========================================
echo.

REM Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Installing Streamlit...
    pip install streamlit pandas beautifulsoup4 requests -q
)

echo Starting Live Dashboard...
echo.
echo The dashboard will open in your browser
echo.

streamlit run live_race_dashboard.py

pause
