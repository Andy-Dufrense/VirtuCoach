@echo off
title VirtuCoach-Graduate
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set HF_ENDPOINT=https://hf-mirror.com
cd /d F:\VirtuCoach-Graduate

echo.
echo   ==========================================
echo     VirtuCoach-Graduate  -  AI Guitar Coach
echo   ==========================================
echo.
echo   [*] Starting backend on port 1218 ...
echo.

start "VirtuCoachGraduate" /B F:\Python310\python.exe run.py

echo   [*] Waiting for backend to be ready...
set RETRY=0
:waitloop
curl -s -o NUL http://localhost:1218/api/models/status 2>NUL
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
echo     Open: http://localhost:1218
echo   ==========================================
echo.
echo   Press any key to stop the server...
pause >nul
