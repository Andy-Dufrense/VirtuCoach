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
    from backend.config import (
        HOST,
        PORT,
        HTTPS_PORT,
        SSL_CERT_FILE,
        SSL_KEY_FILE,
        validate,
    )
    from backend.logging_config import setup_logging, get_logger

    setup_logging()
    logger = get_logger("run")

    warnings = validate()
    for w in warnings:
        logger.warning(w)

    from backend.main import app

    logger.info("VirtuCoach - AI Music Coach")
    if all(os.path.isfile(p) for p in (SSL_CERT_FILE, SSL_KEY_FILE)):
        # 双监听：HTTP 留在 1218（兼容旧书签/快捷方式/旧 SW），HTTPS 走 1443（PWA 安装入口）。
        # 教训：不要把老源的 HTTP 直接停掉，否则老源残留的 Service Worker 会把一切导航兜回缓存首页。
        import threading
        from uvicorn import Config, Server

        logger.info(
            f"HTTP  http://localhost:{PORT}  (兼容旧链接/快捷方式)"
        )
        logger.info(
            f"HTTPS https://localhost:{HTTPS_PORT}  (PWA 安装入口，手机需先信任 CA)"
        )
        http_server = Server(Config(app, host=HOST, port=PORT, log_level="info"))
        https_server = Server(
            Config(
                app,
                host=HOST,
                port=HTTPS_PORT,
                log_level="info",
                ssl_certfile=SSL_CERT_FILE,
                ssl_keyfile=SSL_KEY_FILE,
            )
        )
        threading.Thread(target=http_server.run, daemon=True).start()
        https_server.run()
    else:
        logger.info(f"http://localhost:{PORT}  (未找到证书，纯 HTTP 开发模式)")
        uvicorn.run(app, host=HOST, port=PORT, log_level="info")
