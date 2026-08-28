@echo off
title VirtuCoach-Graduate Regression Tests
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set HF_ENDPOINT=https://hf-mirror.com

rem Automatically move to the project root (parent of scripts\),
rem so it works on any drive letter (F:/G:/...) on any PC.
cd /d "%~dp0.."

rem Prefer the portable Python installed next to the project.
for %%I in ("%~dp0..\Python310\python.exe") do set "PY=%%~fI"
if not exist "%PY%" set PY=python

echo.
echo   ============================================
echo     VirtuCoach-Graduate  -  Regression Tests
echo     Chord detection / Strum classification /
echo     Score baselines
echo   ============================================
echo.

%PY% -m pytest tests/chord_regression/ tests/strum_regression/ tests/score_regression/ -v --tb=short

echo.
echo   Tests finished. Exit code: %errorlevel% (0 = all passed)
pause
