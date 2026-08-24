@echo off
title VirtuCoach-Graduate 回归测试
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set HF_ENDPOINT=https://hf-mirror.com
cd /d F:\VirtuCoach-Graduate

echo.
echo   ============================================
echo     VirtuCoach-Graduate 一键回归测试
echo     和弦检测 / 扫弦分类 / 评分基线 三套
echo   ============================================
echo.

set PY=F:\Python310\python.exe
if not exist %PY% set PY=python

%PY% -m pytest tests/chord_regression/ tests/strum_regression/ tests/score_regression/ -v --tb=short

echo.
echo   测试结束。退出码: %errorlevel% (0 = 全部通过)
pause
