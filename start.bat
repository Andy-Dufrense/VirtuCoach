@echo off
title VirtuCoach
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set HF_ENDPOINT=https://hf-mirror.com
set PYTHONPATH=E:\VirtuCoach-Lib;%PYTHONPATH%
cd /d E:\VirtuCoach

echo.
echo   ==========================================
echo     VirtuCoach  -  AI Guitar Coach
echo   ==========================================
echo.
echo   [*] Starting backend on port 7160 ...
echo.

start "VirtuCoachBackend" /B E:\python.exe run.py

echo   [*] Waiting for backend to be ready...
set RETRY=0
:waitloop
curl -s -o NUL http://localhost:7160/api/models/status 2>NUL
if %errorlevel% equ 0 goto ready
set /a RETRY+=1
if %RETRY% geq 60 (
    echo   [ERROR] Backend failed to start within 120s. Check the output above.
    pause >nul
    exit /b 1
)
ping 127.0.0.1 -n 2 >nul
goto waitloop

:ready
echo.
echo   ==========================================
echo     [OK] Backend is ready!
echo     Open: http://localhost:7160
echo   ==========================================
echo.
echo   Press any key to stop the server...
pause >nul
