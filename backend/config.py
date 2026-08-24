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
FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")

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
# 固定默认密钥：若用随机密钥，每次重启后端所有已登录用户的 token 都会失效
# （表现为"分析完成了但没保存练习记录"）。生产环境请用 JWT_SECRET_KEY 环境变量覆盖。
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "virtucoach-graduate-stable-secret-2026")
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
