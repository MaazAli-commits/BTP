@echo off
title Team 33 - TVPReID Interactive Demonstration
echo ======================================================================
echo    Team 33 BTP - Video Person Re-Identification under Dual Degradation
echo    Interactive Demonstration Engine & Benchmark Dashboard
echo ======================================================================
echo.
echo Starting Flask Application Server...
start "" "http://localhost:5000"
e:\BTP\venv\Scripts\python.exe demo_server.py
pause
