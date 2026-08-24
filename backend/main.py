"""
VirtuCoach - AI Music Coaching System
FastAPI application entry point (thin routing layer).

Architecture:
  routers/   - route definitions (parameter parsing, response formatting)
  services/  - business orchestration (analysis pipeline, hand check)
  pipeline/  - testable filter stages (hand issues, scoring)
"""

import os
import sys
import io

# ── Windows 控制台 UTF-8 编码修复 ──
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ── 屏蔽第三方库的噪声警告（必须在其他 import 之前设置）──
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")       # 0=all 1=info 2=warning 3=error
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"                # 关闭 oneDNN 提示
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # HuggingFace 国内镜像

# Fix sys.path for E:\Python — site-packages not auto-detected when python.exe is at drive root
_site_pkg = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages")
if os.path.isdir(_site_pkg) and _site_pkg not in sys.path:
    sys.path.insert(0, _site_pkg)

from pathlib import Path

import uuid

from fastapi import FastAPI, HTTPException, File, Form, UploadFile, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from config import UPLOAD_DIR, HOST, PORT, CORS_ORIGINS, ADMIN_PASSWORD, validate
from logging_config import setup_logging, get_logger

# ---- logging ----
setup_logging()
logger = get_logger("main")

# ---- startup validation ----
warnings = validate()
for w in warnings:
    logger.warning(w)

# ---- global singletons (stateless, safe to create once) ----
from audio_separator import AudioSeparator
from audio_analyzer import AudioAnalyzer
from video_processor import VideoProcessor
from deepseek_agent import DeepSeekAgent
from vision_analyzer import VisionAnalyzer
from db.reference_db import get_db
from chord_analyzer import ChordAnalyzer
from db.feedback_db import init_feedback_db, get_db as get_feedback_db
from feedback_models import FeedbackUpdate, FeedbackResponse, StatsResponse
from db.user_db import init_user_db, get_db as get_user_db
from user_service import decode_access_token, get_user_by_id
from db.practice_db import init_practice_db, get_db as get_practice_db

audio_analyzer = AudioAnalyzer()
video_processor = VideoProcessor()
deepseek_agent = DeepSeekAgent()
audio_separator = AudioSeparator()
vision_analyzer = VisionAnalyzer()
reference_db = get_db()
chord_analyzer = ChordAnalyzer()

knowledge_db = None
try:
    from db.knowledge_db import knowledge_db as kdb
    knowledge_db = kdb
except Exception:
    logger.warning("Knowledge DB unavailable — RAG features disabled")

# ---- service layer (dependency injection) ----
from services.analysis_service import AnalysisService
from services.hand_check_service import HandCheckService

analysis_service = AnalysisService(
    audio_analyzer=audio_analyzer,
    video_processor=video_processor,
    deepseek_agent=deepseek_agent,
    audio_separator=audio_separator,
    vision_analyzer=vision_analyzer,
    reference_db=reference_db,
    knowledge_db=knowledge_db,
    practice_db_getter=get_practice_db,
)

hand_check_service = HandCheckService(
    chord_analyzer=chord_analyzer,
    video_processor=video_processor,
    vision_analyzer=vision_analyzer,
    deepseek_agent=deepseek_agent,
    reference_db=reference_db,
    upload_dir=UPLOAD_DIR,
)

# ---- shared state ----
tasks: dict = {}

# ---- paths ----
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# ---- FastAPI app ----
app = FastAPI(title="VirtuCoach API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---- feedback system ----
FEEDBACK_SCREENSHOT_DIR = Path(UPLOAD_DIR) / "feedback_screenshots"
FEEDBACK_SCREENSHOT_DIR.mkdir(exist_ok=True)
init_feedback_db()
init_user_db()
init_practice_db()

# ---- register routers ----
from routers.analysis import create_analysis_router
from routers.hand_check import create_hand_check_router
from routers.references import create_references_router
from routers.chat import create_chat_router
from routers.auth import create_auth_router
from routers.practice import create_practice_router
from routers.analytics import router as analytics_router

analysis_router = create_analysis_router(analysis_service, UPLOAD_DIR, str(FRONTEND_DIR))
hand_check_router = create_hand_check_router(hand_check_service, UPLOAD_DIR, get_practice_db)
references_router = create_references_router(reference_db, video_processor, vision_analyzer, UPLOAD_DIR)
chat_router = create_chat_router(deepseek_agent, knowledge_db)
auth_router = create_auth_router()
practice_router = create_practice_router()

analysis_router.tasks = tasks
chat_router.tasks = tasks

app.include_router(analysis_router)
app.include_router(hand_check_router)
app.include_router(references_router)
app.include_router(chat_router)
app.include_router(auth_router)
app.include_router(practice_router)
app.include_router(analytics_router)

logger.info("Routes registered: analysis, hand_check, references, chat, practice, analytics")


# ---- simple endpoints ----

@app.get("/favicon.ico")
async def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "VirtuCoach", "version": "2.1.0"}


@app.get("/api/models/status")
async def model_status():
    vision_name = None
    if vision_analyzer.available:
        vision_name = "qwen" if vision_analyzer.provider == "qwen" else "claude"
    return {
        "basic_pitch": audio_analyzer.model_loaded,
        "mediapipe_hands": video_processor.model_loaded,
        "vision_ai": vision_analyzer.available,
        "vision_provider": vision_name,
    }


@app.get("/uploads/{filename}")
async def get_upload(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")


SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")


@app.get("/snapshots/{filename}")
async def get_snapshot(filename: str):
    file_path = os.path.join(SNAPSHOT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")


# 页面类文件统一禁用浏览器缓存，避免"代码已修复但页面还是旧的"
NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}


def page_response(path) -> FileResponse:
    return FileResponse(str(path), headers=NO_CACHE_HEADERS)


@app.get("/")
@app.get("/index.html")
async def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return page_response(index_path)
    return {"error": "Frontend not found"}


@app.get("/admin")
def serve_admin():
    admin_path = FRONTEND_DIR / "admin.html"
    if admin_path.exists():
        return page_response(admin_path)
    raise HTTPException(status_code=404, detail="Admin panel not found")


@app.get("/dashboard.html")
def serve_dashboard():
    return page_response(FRONTEND_DIR / "dashboard.html")


@app.get("/analysis.html")
def serve_analysis():
    return page_response(FRONTEND_DIR / "analysis.html")


@app.get("/login.html")
def serve_login():
    return page_response(FRONTEND_DIR / "login.html")


@app.get("/register.html")
def serve_register():
    return page_response(FRONTEND_DIR / "register.html")


# ═══════════════════════════════════════════
# Feedback System
# ═══════════════════════════════════════════


def _verify_admin(token: str):
    # fail-closed：未配置管理密码（空）或 token 为空一律拒绝
    if not ADMIN_PASSWORD or not token or token != ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="Invalid admin token")


@app.get("/api/verify-admin")
async def verify_admin(token: str = Query(...)):
    _verify_admin(token)
    return {"ok": True}


@app.post("/api/feedbacks")
async def create_feedback(
    project: str = Form("VirtuCoach"),
    category: str = Form(...),
    severity: str = Form(...),
    title: str = Form(...),
    description: str = Form(""),
    steps_to_reproduce: str = Form(""),
    tester_name: str = Form(""),
    browser_info: str = Form(""),
    screenshot: UploadFile | None = None,
    authorization: str | None = Header(None),
):
    # Auto-collect user info from auth token
    user_id = None
    username = ""
    if authorization and authorization.startswith("Bearer "):
        try:
            payload = decode_access_token(authorization[7:])
            user = get_user_by_id(payload["user_id"])
            if user:
                user_id = user.id
                username = user.username
        except Exception:
            pass

    screenshot_name = ""
    if screenshot and screenshot.filename:
        ext = Path(screenshot.filename).suffix or ".png"
        screenshot_name = f"{uuid.uuid4().hex}{ext}"
        contents = await screenshot.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Screenshot too large (max 10MB)")
        (FEEDBACK_SCREENSHOT_DIR / screenshot_name).write_bytes(contents)

    conn = get_feedback_db()
    existing = conn.execute(
        """SELECT id FROM feedbacks
           WHERE title = ? AND description = ?
           AND datetime(created_at, '+5 seconds') >= datetime('now', 'localtime')
           LIMIT 1""",
        [title, description],
    ).fetchone()
    if existing:
        conn.close()
        raise HTTPException(status_code=429, detail="请勿重复提交，5秒内相同反馈已存在")

    conn.execute(
        """INSERT INTO feedbacks (project, category, severity, title, description,
           steps_to_reproduce, screenshot, tester_name, user_id, username, browser_info)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [project, category, severity, title, description,
         steps_to_reproduce, screenshot_name, tester_name, user_id, username, browser_info],
    )
    conn.commit()
    feedback_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    row = conn.execute("SELECT * FROM feedbacks WHERE id = ?", [feedback_id]).fetchone()
    conn.close()
    return FeedbackResponse.from_row(row)


@app.get("/api/feedbacks")
def list_feedbacks(
    project: str = "",
    category: str = "",
    status: str = "",
    search: str = "",
    sort: str = "newest",
    page: int = 1,
):
    conn = get_feedback_db()
    conditions = []
    params = []

    if project:
        conditions.append("project = ?")
        params.append(project)
    if category:
        conditions.append("category = ?")
        params.append(category)
    if status:
        conditions.append("status = ?")
        params.append(status)
    if search:
        conditions.append("(title LIKE ? OR description LIKE ?)")
        params.extend([f"%{search}%", f"%{search}%"])

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    order = "ORDER BY created_at DESC" if sort == "newest" else "ORDER BY created_at ASC"

    per_page = 20
    offset = (page - 1) * per_page

    rows = conn.execute(
        f"SELECT * FROM feedbacks {where} {order} LIMIT ? OFFSET ?",
        params + [per_page, offset],
    ).fetchall()

    total = conn.execute(
        f"SELECT COUNT(*) FROM feedbacks {where}", params
    ).fetchone()[0]

    conn.close()
    return {
        "items": [FeedbackResponse.from_row(r) for r in rows],
        "total": total,
        "page": page,
        "per_page": per_page,
    }


@app.get("/api/feedbacks/{feedback_id}")
def get_feedback(feedback_id: int, admin_token: str = Query(...)):
    _verify_admin(admin_token)
    conn = get_feedback_db()
    row = conn.execute("SELECT * FROM feedbacks WHERE id = ?", [feedback_id]).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Feedback not found")
    return FeedbackResponse.from_row(row)


@app.patch("/api/feedbacks/{feedback_id}")
def update_feedback(feedback_id: int, body: FeedbackUpdate, admin_token: str = Query(...)):
    _verify_admin(admin_token)
    conn = get_feedback_db()
    existing = conn.execute("SELECT * FROM feedbacks WHERE id = ?", [feedback_id]).fetchone()
    if not existing:
        conn.close()
        raise HTTPException(status_code=404, detail="Feedback not found")

    updates = {}
    if body.status is not None:
        updates["status"] = body.status
    if body.admin_note is not None:
        updates["admin_note"] = body.admin_note

    if updates:
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [feedback_id]
        conn.execute(f"UPDATE feedbacks SET {set_clause} WHERE id = ?", values)
        conn.commit()

    row = conn.execute("SELECT * FROM feedbacks WHERE id = ?", [feedback_id]).fetchone()
    conn.close()
    return FeedbackResponse.from_row(row)


@app.delete("/api/feedbacks/{feedback_id}")
def delete_feedback(feedback_id: int, admin_token: str = Query(...)):
    _verify_admin(admin_token)
    conn = get_feedback_db()
    existing = conn.execute("SELECT * FROM feedbacks WHERE id = ?", [feedback_id]).fetchone()
    if not existing:
        conn.close()
        raise HTTPException(status_code=404, detail="Feedback not found")
    conn.execute("DELETE FROM feedbacks WHERE id = ?", [feedback_id])
    conn.commit()
    conn.close()
    return {"ok": True}


@app.get("/api/screenshots/{filename}")
def get_feedback_screenshot(filename: str):
    path = FEEDBACK_SCREENSHOT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Screenshot not found")
    return FileResponse(str(path))


@app.get("/api/feedback-stats")
def get_feedback_stats(admin_token: str = Query(...)):
    _verify_admin(admin_token)
    conn = get_feedback_db()
    total = conn.execute("SELECT COUNT(*) FROM feedbacks").fetchone()[0]
    pending = conn.execute("SELECT COUNT(*) FROM feedbacks WHERE status='pending'").fetchone()[0]
    investigating = conn.execute("SELECT COUNT(*) FROM feedbacks WHERE status='investigating'").fetchone()[0]
    resolved = conn.execute("SELECT COUNT(*) FROM feedbacks WHERE status='resolved'").fetchone()[0]
    today = conn.execute(
        "SELECT COUNT(*) FROM feedbacks WHERE date(created_at) = date('now', 'localtime')"
    ).fetchone()[0]

    projects = conn.execute(
        "SELECT project, COUNT(*) as cnt FROM feedbacks GROUP BY project"
    ).fetchall()
    by_project = {r["project"]: r["cnt"] for r in projects}

    conn.close()
    return StatsResponse(
        total=total, pending=pending, investigating=investigating,
        resolved=resolved, today=today, by_project=by_project,
    )


@app.get("/api/admin/overview")
def get_admin_overview(admin_token: str = Query(...)):
    """Data overview for admin dashboard."""
    _verify_admin(admin_token)
    user_conn = get_user_db()
    practice_conn = get_practice_db()
    fb_conn = get_feedback_db()

    total_users = user_conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    today_users = user_conn.execute(
        "SELECT COUNT(*) FROM users WHERE date(created_at) = date('now','localtime')"
    ).fetchone()[0]
    total_sessions = practice_conn.execute("SELECT COUNT(*) FROM practice_sessions").fetchone()[0]
    today_sessions = practice_conn.execute(
        "SELECT COUNT(*) FROM practice_sessions WHERE date(created_at) = date('now','localtime')"
    ).fetchone()[0]
    total_feedbacks = fb_conn.execute("SELECT COUNT(*) FROM feedbacks").fetchone()[0]

    user_conn.close()
    practice_conn.close()
    fb_conn.close()

    return {
        "total_users": total_users,
        "today_users": today_users,
        "total_sessions": total_sessions,
        "today_sessions": today_sessions,
        "total_feedbacks": total_feedbacks,
    }


@app.get("/api/admin/users")
def get_admin_users(admin_token: str = Query(...)):
    """List all users with their practice statistics. Admin token required."""
    _verify_admin(admin_token)
    user_conn = get_user_db()
    practice_conn = get_practice_db()

    users = user_conn.execute(
        "SELECT id, username, email, created_at, last_login, last_activity, active_seconds "
        "FROM users ORDER BY created_at DESC"
    ).fetchall()

    result = []
    for u in users:
        row = dict(u)
        row["practice_count"] = practice_conn.execute(
            "SELECT COUNT(*) FROM practice_sessions WHERE user_id = ?", [row["id"]]
        ).fetchone()[0]
        row["last_practice"] = practice_conn.execute(
            "SELECT created_at FROM practice_sessions WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
            [row["id"]],
        ).fetchone()
        row["last_practice"] = row["last_practice"][0] if row["last_practice"] else None
        result.append(row)

    user_conn.close()
    practice_conn.close()
    return {"users": result}


@app.post("/api/admin/users/{user_id}/reset-password")
def admin_reset_user_password(
    user_id: int,
    body: dict,
    admin_token: str = Query(...),
):
    """Admin resets a user's password (bcrypt哈希存储，原密码不可查看)."""
    _verify_admin(admin_token)
    from user_service import admin_reset_password
    new_password = (body or {}).get("new_password", "")
    try:
        admin_reset_password(user_id, new_password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True}


@app.get("/{filename:path}")
async def serve_static(filename: str):
    """Serve frontend static files (CSS, JS, etc)."""
    # Only serve known static extensions; everything else is an API 404
    if not any(filename.endswith(ext) for ext in (".css", ".js", ".html", ".svg", ".png", ".jpg", ".ico", ".json", ".woff2")):
        raise HTTPException(status_code=404, detail="Not found")
    file_path = os.path.join(str(FRONTEND_DIR), filename)
    # Prevent directory traversal
    if not os.path.realpath(file_path).startswith(os.path.realpath(str(FRONTEND_DIR))):
        raise HTTPException(status_code=404, detail="Not found")
    if os.path.exists(file_path):
        # HTML/CSS/JS 禁缓存，图片/字体保持默认
        if filename.endswith((".html", ".css", ".js")):
            return page_response(file_path)
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")



# ---- startup ----
if __name__ == "__main__":
    import uvicorn
    logger.info(f"VirtuCoach v2.1.0  http://localhost:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
