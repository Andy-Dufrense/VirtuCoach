"""VirtuCoach launcher. Run from project root: `python run.py`"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "backend"))

os.environ["PYTHONIOENCODING"] = "utf-8"

# ---- auto-detect site-packages from the running Python ----
# 项目独立 Lib 目录优先，E:\lib\site-packages 作为兜底
_project_lib = str(BASE_DIR.parent / "VirtuCoach-Lib")
if os.path.isdir(_project_lib) and _project_lib not in sys.path:
    sys.path.insert(0, _project_lib)

_custom_sp = os.environ.get("VIRTUCOACH_SITE_PACKAGES", "")
if _custom_sp:
    if _custom_sp not in sys.path:
        sys.path.insert(0, _custom_sp)
else:
    import site
    for sp in site.getsitepackages():
        if sp not in sys.path:
            sys.path.insert(0, sp)

if __name__ == "__main__":
    import uvicorn
    from backend.config import HOST, PORT, validate
    from backend.logging_config import setup_logging, get_logger

    setup_logging()
    logger = get_logger("run")

    warnings = validate()
    for w in warnings:
        logger.warning(w)

    from backend.main import app

    logger.info("VirtuCoach - AI Music Coach")
    logger.info(f"http://localhost:{PORT}")

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
