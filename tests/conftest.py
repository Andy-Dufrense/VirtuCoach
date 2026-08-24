"""
VirtuCoach-Graduate 回归测试共享路径配置。

作用：
1. 把仓库根目录和 backend/ 加入 sys.path，测试可直接 import 后端模块
2. site-packages 解析与 run.py 同序：项目同级 VirtuCoach-Lib 优先，E:/VirtuCoach-Lib 兜底，
   最后退回解释器自身的 site-packages（答辩机无 E 盘依赖也能跑）

用法（一键回归，见 scripts/run_regression_tests.bat）：
    cd F:/VirtuCoach-Graduate
    python -m pytest tests/ -v
"""

import os
import sys
from pathlib import Path

# 与 start.bat 保持一致：模型本地缓存离线加载，避免测试时访问 HuggingFace
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"

for _p in (str(ROOT), str(BACKEND)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_lib = ROOT.parent / "VirtuCoach-Lib"
if not _lib.is_dir():
    _lib = Path("E:/VirtuCoach-Lib")
if _lib.is_dir():
    if str(_lib) not in sys.path:
        sys.path.insert(0, str(_lib))
else:
    import site
    for _sp in site.getsitepackages():
        if _sp not in sys.path:
            sys.path.insert(0, _sp)

# 基线视频统一放 tests/videos/
TESTS_VIDEO_DIR = ROOT / "tests" / "videos"
