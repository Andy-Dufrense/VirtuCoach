"""VirtuCoach configuration. All paths are computed dynamically from project root.

Override any setting with environment variables:
  VIRTUCOACH_UPLOAD_DIR, VIRTUCOACH_MODEL_DIR, VIRTUCOACH_HOST, VIRTUCOACH_PORT,
  DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
"""

import os
import sys
from pathlib import Path

# ---- load .env before reading any env vars ----
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=str(_env_path))
except ImportError:
    pass

# ---- project root (2 levels above this file: backend/config.py -> VirtuCoach/) ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---- paths (computed from project root, overridable via env) ----
UPLOAD_DIR = os.environ.get("VIRTUCOACH_UPLOAD_DIR", str(PROJECT_ROOT / "uploads"))
MODEL_DIR = os.environ.get("VIRTUCOACH_MODEL_DIR", str(PROJECT_ROOT / "models"))

# ---- HTTPS (P0-4：局域网手机安装 PWA 需要安全上下文) ----
# 证书由 scripts/gen_https_certs.py 生成；存在则 HTTPS 启动，不存在则回退纯 HTTP。
SSL_CERT_FILE = os.environ.get(
    "VIRTUCOACH_SSL_CERT", str(PROJECT_ROOT / "certs" / "virtucoach-server.crt")
)
SSL_KEY_FILE = os.environ.get(
    "VIRTUCOACH_SSL_KEY", str(PROJECT_ROOT / "certs" / "virtucoach-server.key")
)
HTTPS_PORT = int(os.environ.get("VIRTUCOACH_HTTPS_PORT", "1443"))
CA_CERT_FILE = os.environ.get(
    "VIRTUCOACH_CA_CERT", str(PROJECT_ROOT / "certs" / "virtucoach-ca.crt")
)


def _resolve_ffmpeg_path() -> str:
    """定位 ffmpeg 可执行文件：环境变量 → PATH → imageio_ffmpeg 自带二进制。

    移动硬盘换电脑后 PATH 里不一定有 ffmpeg，而 runtime Python 的
    imageio_ffmpeg 包里通常自带一个完整 ffmpeg，用它兜底最稳妥。
    """
    env = os.environ.get("FFMPEG_PATH", "").strip()
    if env:
        return env
    try:
        import shutil
        if shutil.which("ffmpeg"):
            return "ffmpeg"
    except Exception:
        pass
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


FFMPEG_PATH = _resolve_ffmpeg_path()

# ---- DeepSeek API ----
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# ---- audio analysis ----
SAMPLE_RATE = 16000
FRAME_LENGTH_MS = 32
ONSET_THRESHOLD = 0.5

# ---- video analysis ----
FRAME_INTERVAL_SEC = 0.3  # ~3 fps — dense enough to catch sub-second hand issues
MAX_UNIFORM_KEYFRAMES = 200   # 均匀采样上限，超过则等比降采样
MAX_BONUS_KEYFRAMES = 80      # 错误+技巧附加上限
MAX_TOTAL_KEYFRAMES = 250     # 绝对安全上限
ERROR_FRAME_WINDOW = 0.5

# ---- JWT ----
# 真实密钥放 .env（gitignored，不会进仓库）。这里只留开发用回退值：
# 必须是固定值（随机密钥会导致每次重启 token 全失效），但不能把生产密钥写进代码库。
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "virtucoach-graduate-dev-fallback-2026")
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_HOURS = int(os.environ.get("JWT_EXPIRE_HOURS", "24"))

# ---- server ----
HOST = os.environ.get("VIRTUCOACH_HOST", "0.0.0.0")
PORT = int(os.environ.get("VIRTUCOACH_PORT", "1218"))

# ---- admin ----
ADMIN_PASSWORD = os.environ.get("VIRTUCOACH_ADMIN_PASSWORD", "andy0716")

# ---- admin analytics ----
ONLINE_WINDOW_MINUTES = 30        # last_activity 在此窗口内视为在线
ONLINE_TIMEOUT_SECONDS = 90       # 心跳超时：超过此时长无心跳即视为离线
WILLINGNESS_WINDOW_DAYS = 30      # 使用意愿统计的时间窗口
WILLINGNESS_HIGH = 70             # ≥ 此分数为高意愿
WILLINGNESS_LOW = 40              # < 此分数为低意愿

# ---- CORS ----
# Comma-separated list of allowed origins. Defaults to localhost for dev.
# Set VIRTUCOACH_CORS_ORIGINS="*" to allow all (dev only).
_cors_raw = os.environ.get("VIRTUCOACH_CORS_ORIGINS", "http://localhost:1218,http://localhost:3000")
CORS_ORIGINS = [o.strip() for o in _cors_raw.split(",") if o.strip()]


def validate() -> list[str]:
    """Check required configuration on startup. Returns list of warnings."""
    warnings = []
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "sk-your-deepseek-api-key-here":
        warnings.append(
            "DEEPSEEK_API_KEY not set. AI report generation will be unavailable. "
            "Set it in .env or as an environment variable."
        )
    if not os.path.isdir(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR, exist_ok=True)
    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
    return warnings
