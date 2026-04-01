@echo off
setlocal enabledelayedexpansion
title Kalshi AI Trading Bot
cd /d "%~dp0"

echo.
echo  ================================================
echo   KALSHI AI TRADING BOT
echo   Beast Mode + Lean Mode + Dashboard
echo  ================================================
echo.

REM ── Find a working Python ────────────────────────────────────────────────────
set PYTHON=

REM 1. Try py -3.12 first (installed version)
py -3.12 --version >nul 2>&1
if !errorlevel! == 0 ( set PYTHON=py -3.12 )

REM 2. Try virtual environment
if "!PYTHON!"=="" (
    if exist ".venv\Scripts\python.exe" (
        .venv\Scripts\python.exe --version >nul 2>&1
        if !errorlevel! == 0 ( set PYTHON=.venv\Scripts\python.exe )
    )
)
if "!PYTHON!"=="" (
    if exist "venv\Scripts\python.exe" (
        venv\Scripts\python.exe --version >nul 2>&1
        if !errorlevel! == 0 ( set PYTHON=venv\Scripts\python.exe )
    )
)

REM 3. Fall back to system Python
if "!PYTHON!"=="" (
    python --version >nul 2>&1
    if !errorlevel! == 0 ( set PYTHON=python )
)
if "!PYTHON!"=="" (
    python3 --version >nul 2>&1
    if !errorlevel! == 0 ( set PYTHON=python3 )
)

if "!PYTHON!"=="" (
    echo ERROR: No Python found. Install Python from https://python.org
    pause
    exit /b 1
)

echo  Python: !PYTHON!
echo.

REM ── Check .env exists ────────────────────────────────────────────────────────
if not exist ".env" (
    echo  WARNING: .env file not found.
    echo  Copy .env.template to .env and add your API keys first.
    echo.
    pause
    exit /b 1
)

REM ── Start Bot Orchestrator (Beast + Lean) in its own window ─────────────────
echo  Starting Bot Orchestrator ^(Beast + Lean Mode^)...
start "Kalshi Bot - Orchestrator" cmd /k "cd /d "%~dp0" && echo Bot Orchestrator starting... && !PYTHON! bot_orchestrator.py --beast --lean"

REM ── Wait for orchestrator to initialise ─────────────────────────────────────
echo  Waiting for bot to initialise ^(5 seconds^)...
timeout /t 5 /nobreak >nul

REM ── Start Streamlit Dashboard in its own window ──────────────────────────────
echo  Starting Streamlit Dashboard...
start "Kalshi Bot - Dashboard" cmd /k "cd /d "%~dp0" && echo Dashboard starting... && !PYTHON! -m streamlit run streamlit_dashboard.py --server.headless true"

REM ── Wait for Streamlit to bind to port 8501 ──────────────────────────────────
echo  Waiting for dashboard ^(6 seconds^)...
timeout /t 6 /nobreak >nul

REM ── Open browser ─────────────────────────────────────────────────────────────
echo  Opening dashboard in browser...
start http://localhost:8501

echo.
echo  ================================================
echo   Bot is running. Two windows should be open:
echo     - Kalshi Bot - Orchestrator
echo     - Kalshi Bot - Dashboard
echo   Dashboard: http://localhost:8501
echo   Close those windows to stop the bot.
echo  ================================================
echo.
endlocal
