"""Admin data analytics routes — 后端聚合计算，前端只负责画图。

指标函数均接收数据库连接作为参数，便于用内存库做单元测试。
所有端点需要 admin_token（与 feedback 管理接口同一校验模式）。
"""

import csv
import io
from datetime import date, datetime, timedelta

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from config import (
    ADMIN_PASSWORD,
    ONLINE_WINDOW_MINUTES,
    ONLINE_TIMEOUT_SECONDS,
    WILLINGNESS_HIGH,
    WILLINGNESS_LOW,
    WILLINGNESS_WINDOW_DAYS,
)
from db.user_db import get_db as get_user_db
from db.practice_db import get_db as get_practice_db
from db.feedback_db import get_db as get_feedback_db
from logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["analytics"])

MAX_DAYS = 365
TIER_LABELS = {"high": "高意愿", "mid": "中意愿", "low": "低意愿"}


def _verify_admin(token: str):
    if token != ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="Invalid admin token")


def _clamp_days(days: int) -> int:
    return max(1, min(days or 30, MAX_DAYS))


def _date_series(days: int, today: date | None = None) -> list[str]:
    """最近 days 天（含今天）的日期字符串，旧→新。"""
    today = today or date.today()
    return [(today - timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)]


def _fill_daily(rows, series: list[str]) -> list[int]:
    """把 (date_str, count) 行对齐到日期序列，缺失补 0。"""
    counts = {d: c for d, c in rows}
    return [counts.get(d, 0) for d in series]


# ═══════════════════════════════════════════
# 指标计算（纯函数，conn 参数注入）
# ═══════════════════════════════════════════

def compute_online(user_conn, now: datetime | None = None) -> dict:
    now = now or datetime.now()
    # 实时在线：online=1 且最近有心跳（与登录拦截口径一致）
    cutoff = (now - timedelta(seconds=ONLINE_TIMEOUT_SECONDS)).strftime("%Y-%m-%d %H:%M:%S")
    today = now.strftime("%Y-%m-%d")
    week_start = (now - timedelta(days=6)).strftime("%Y-%m-%d")

    current_online = user_conn.execute(
        "SELECT COUNT(*) FROM users "
        "WHERE online = 1 AND last_activity IS NOT NULL AND last_activity >= ?",
        [cutoff],
    ).fetchone()[0]
    today_active = user_conn.execute(
        "SELECT COUNT(*) FROM users WHERE date(last_login) = ?", [today]
    ).fetchone()[0]
    week_active = user_conn.execute(
        "SELECT COUNT(*) FROM users WHERE date(last_activity) >= ?", [week_start]
    ).fetchone()[0]
    top = user_conn.execute(
        "SELECT username, active_seconds, last_activity FROM users "
        "WHERE active_seconds > 0 ORDER BY active_seconds DESC LIMIT 10"
    ).fetchall()

    return {
        "current_online": current_online,
        "today_active": today_active,
        "week_active": week_active,
        "online_window_minutes": ONLINE_WINDOW_MINUTES,
        "top_active": [dict(r) for r in top],
    }


def compute_logins(user_conn, days: int) -> dict:
    series = _date_series(days)
    start = series[0]
    rows = user_conn.execute(
        "SELECT date(login_at) d, COUNT(*) c FROM login_events "
        "WHERE date(login_at) >= ? GROUP BY d",
        [start],
    ).fetchall()
    daily = _fill_daily(rows, series)
    ranking = user_conn.execute(
        "SELECT u.username, COUNT(*) cnt, MAX(le.login_at) last_login_at "
        "FROM login_events le JOIN users u ON u.id = le.user_id "
        "WHERE date(le.login_at) >= ? GROUP BY le.user_id "
        "ORDER BY cnt DESC, last_login_at DESC LIMIT 20",
        [start],
    ).fetchall()
    return {
        "days": days,
        "dates": series,
        "daily": daily,
        "total": sum(daily),
        "ranking": [dict(r) for r in ranking],
    }


def compute_practice(user_conn, practice_conn, days: int) -> dict:
    series = _date_series(days)
    start = series[0]
    rows = practice_conn.execute(
        "SELECT date(created_at) d, COUNT(*) c FROM practice_sessions "
        "WHERE date(created_at) >= ? GROUP BY d",
        [start],
    ).fetchall()
    daily = _fill_daily(rows, series)

    names = {
        r["id"]: r["username"]
        for r in user_conn.execute("SELECT id, username FROM users")
    }
    per_user = practice_conn.execute(
        "SELECT user_id, COUNT(*) cnt, "
        "ROUND(AVG(CASE WHEN overall_score > 0 THEN overall_score END), 1) avg_score "
        "FROM practice_sessions WHERE date(created_at) >= ? GROUP BY user_id "
        "ORDER BY cnt DESC LIMIT 20",
        [start],
    ).fetchall()
    ranking = [
        {
            "username": names.get(r["user_id"], f"user#{r['user_id']}"),
            "count": r["cnt"],
            "avg_score": r["avg_score"],
        }
        for r in per_user
    ]

    tracks = practice_conn.execute(
        "SELECT chord_or_track name, COUNT(*) cnt FROM practice_sessions "
        "WHERE date(created_at) >= ? AND chord_or_track != '' "
        "GROUP BY chord_or_track ORDER BY cnt DESC LIMIT 10",
        [start],
    ).fetchall()
    track_distribution = [{"name": r["name"], "count": r["cnt"]} for r in tracks]

    scores = practice_conn.execute(
        "SELECT overall_score FROM practice_sessions "
        "WHERE date(created_at) >= ? AND overall_score > 0",
        [start],
    ).fetchall()
    buckets = {"0-60": 0, "60-75": 0, "75-90": 0, "90+": 0}
    for r in scores:
        s = r["overall_score"]
        if s < 60:
            buckets["0-60"] += 1
        elif s < 75:
            buckets["60-75"] += 1
        elif s < 90:
            buckets["75-90"] += 1
        else:
            buckets["90+"] += 1

    return {
        "days": days,
        "dates": series,
        "daily": daily,
        "total": sum(daily),
        "ranking": ranking,
        "track_distribution": track_distribution,
        "score_buckets": buckets,
    }


def compute_growth(user_conn, days: int) -> dict:
    series = _date_series(days)
    start = series[0]
    rows = user_conn.execute(
        "SELECT date(created_at) d, COUNT(*) c FROM users "
        "WHERE date(created_at) >= ? GROUP BY d",
        [start],
    ).fetchall()
    daily = _fill_daily(rows, series)

    base = user_conn.execute(
        "SELECT COUNT(*) FROM users WHERE date(created_at) < ?", [start]
    ).fetchone()[0]
    cumulative = []
    running = base
    for c in daily:
        running += c
        cumulative.append(running)

    total = user_conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    return {
        "days": days,
        "dates": series,
        "daily_registrations": daily,
        "cumulative": cumulative,
        "total_users": total,
    }


def compute_willingness(
    user_conn, practice_conn, feedback_conn,
    now: datetime | None = None, window_days: int | None = None,
) -> dict:
    """综合使用意愿评分（0-100）。

    维度权重：登录频率 30% + 练习频率 30% + 在线时长 20% + 反馈参与 20%。
    每个维度在全体用户间 min-max 归一化后加权。
    """
    window_days = window_days or WILLINGNESS_WINDOW_DAYS
    now = now or datetime.now()
    start = (now - timedelta(days=window_days - 1)).strftime("%Y-%m-%d")

    users = user_conn.execute(
        "SELECT id, username, active_seconds FROM users ORDER BY id"
    ).fetchall()
    if not users:
        return {
            "window_days": window_days, "users": [],
            "tiers": {"high": 0, "mid": 0, "low": 0},
        }

    logins = {
        r["user_id"]: r["cnt"] for r in user_conn.execute(
            "SELECT user_id, COUNT(*) cnt FROM login_events "
            "WHERE date(login_at) >= ? GROUP BY user_id", [start])
    }
    practices = {
        r["user_id"]: r["cnt"] for r in practice_conn.execute(
            "SELECT user_id, COUNT(*) cnt FROM practice_sessions "
            "WHERE date(created_at) >= ? GROUP BY user_id", [start])
    }
    feedbacks = {
        r["user_id"]: r["cnt"] for r in feedback_conn.execute(
            "SELECT user_id, COUNT(*) cnt FROM feedbacks "
            "WHERE user_id IS NOT NULL AND date(created_at) >= ? GROUP BY user_id",
            [start])
    }

    dims = [
        {
            "username": u["username"],
            "logins": logins.get(u["id"], 0),
            "practices": practices.get(u["id"], 0),
            "active_seconds": u["active_seconds"] or 0,
            "feedbacks": feedbacks.get(u["id"], 0),
        }
        for u in users
    ]

    def norm(values: list) -> list[float]:
        hi = max(values)
        if hi <= 0:
            return [0.0] * len(values)
        return [v / hi for v in values]

    n_logins = norm([d["logins"] for d in dims])
    n_practices = norm([d["practices"] for d in dims])
    n_active = norm([d["active_seconds"] for d in dims])
    n_feedbacks = norm([d["feedbacks"] for d in dims])

    result = []
    for i, d in enumerate(dims):
        score = round(
            (0.3 * n_logins[i] + 0.3 * n_practices[i]
             + 0.2 * n_active[i] + 0.2 * n_feedbacks[i]) * 100,
            1,
        )
        tier = "high" if score >= WILLINGNESS_HIGH else (
            "mid" if score >= WILLINGNESS_LOW else "low"
        )
        result.append({**d, "score": score, "tier": tier})

    result.sort(key=lambda x: x["score"], reverse=True)
    tiers = {t: sum(1 for r in result if r["tier"] == t) for t in ("high", "mid", "low")}
    return {"window_days": window_days, "users": result, "tiers": tiers}


# ═══════════════════════════════════════════
# 端点
# ═══════════════════════════════════════════

@router.get("/admin/analytics/online")
def analytics_online(admin_token: str = Query(...)):
    _verify_admin(admin_token)
    conn = get_user_db()
    try:
        return compute_online(conn)
    finally:
        conn.close()


@router.get("/admin/analytics/logins")
def analytics_logins(days: int = Query(30), admin_token: str = Query(...)):
    _verify_admin(admin_token)
    days = _clamp_days(days)
    conn = get_user_db()
    try:
        return compute_logins(conn, days)
    finally:
        conn.close()


@router.get("/admin/analytics/practice")
def analytics_practice(days: int = Query(30), admin_token: str = Query(...)):
    _verify_admin(admin_token)
    days = _clamp_days(days)
    user_conn, practice_conn = get_user_db(), get_practice_db()
    try:
        return compute_practice(user_conn, practice_conn, days)
    finally:
        user_conn.close()
        practice_conn.close()


@router.get("/admin/analytics/growth")
def analytics_growth(days: int = Query(30), admin_token: str = Query(...)):
    _verify_admin(admin_token)
    days = _clamp_days(days)
    conn = get_user_db()
    try:
        return compute_growth(conn, days)
    finally:
        conn.close()


@router.get("/admin/analytics/willingness")
def analytics_willingness(admin_token: str = Query(...)):
    _verify_admin(admin_token)
    user_conn, practice_conn, feedback_conn = get_user_db(), get_practice_db(), get_feedback_db()
    try:
        return compute_willingness(user_conn, practice_conn, feedback_conn)
    finally:
        user_conn.close()
        practice_conn.close()
        feedback_conn.close()


# ═══════════════════════════════════════════
# CSV 导出（utf-8-sig 带 BOM，Excel 直接打开不乱码）
# ═══════════════════════════════════════════

def _csv_response(header: list[str], rows: list[list], filename: str) -> Response:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    writer.writerows(rows)
    content = "\ufeff" + buf.getvalue()  # BOM：Excel 识别 UTF-8 中文
    return Response(
        content=content.encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/admin/analytics/export")
def analytics_export(
    type: str = Query(...),
    days: int = Query(30),
    admin_token: str = Query(...),
):
    _verify_admin(admin_token)
    days = _clamp_days(days)
    start = _date_series(days)[0]
    stamp = date.today().isoformat()

    user_conn, practice_conn, feedback_conn = get_user_db(), get_practice_db(), get_feedback_db()
    try:
        names = {
            r["id"]: r["username"]
            for r in user_conn.execute("SELECT id, username FROM users")
        }

        if type == "logins":
            rows = user_conn.execute(
                "SELECT u.username, le.login_at FROM login_events le "
                "JOIN users u ON u.id = le.user_id "
                "WHERE date(le.login_at) >= ? ORDER BY le.login_at",
                [start],
            ).fetchall()
            return _csv_response(
                ["用户名", "登录时间"],
                [[r["username"], r["login_at"]] for r in rows],
                f"virtucoach_logins_{stamp}.csv",
            )

        if type == "practice":
            rows = practice_conn.execute(
                "SELECT user_id, created_at, instrument, skill_level, chord_or_track, "
                "overall_score, audio_score, hand_score, mode "
                "FROM practice_sessions WHERE date(created_at) >= ? ORDER BY created_at",
                [start],
            ).fetchall()
            return _csv_response(
                ["用户名", "时间", "乐器", "水平", "曲目/和弦", "综合分", "音频分", "手型分", "模式"],
                [
                    [
                        names.get(r["user_id"], f"user#{r['user_id']}"),
                        r["created_at"], r["instrument"], r["skill_level"],
                        r["chord_or_track"], r["overall_score"], r["audio_score"],
                        r["hand_score"], r["mode"],
                    ]
                    for r in rows
                ],
                f"virtucoach_practice_{stamp}.csv",
            )

        if type == "users":
            rows = user_conn.execute(
                "SELECT id, username, email, created_at, last_login, last_activity, "
                "active_seconds FROM users ORDER BY created_at"
            ).fetchall()
            login_counts = {
                r["user_id"]: r["cnt"] for r in user_conn.execute(
                    "SELECT user_id, COUNT(*) cnt FROM login_events "
                    "WHERE date(login_at) >= ? GROUP BY user_id", [start])
            }
            practice_counts = {
                r["user_id"]: r["cnt"] for r in practice_conn.execute(
                    "SELECT user_id, COUNT(*) cnt FROM practice_sessions "
                    "WHERE date(created_at) >= ? GROUP BY user_id", [start])
            }
            return _csv_response(
                ["用户名", "邮箱", "注册时间", "上次登录", "最近活跃", "累计在线秒数",
                 f"近{days}天登录次数", f"近{days}天练习次数"],
                [
                    [
                        r["username"], r["email"], r["created_at"],
                        r["last_login"] or "", r["last_activity"] or "",
                        r["active_seconds"] or 0,
                        login_counts.get(r["id"], 0),
                        practice_counts.get(r["id"], 0),
                    ]
                    for r in rows
                ],
                f"virtucoach_users_{stamp}.csv",
            )

        if type == "willingness":
            data = compute_willingness(user_conn, practice_conn, feedback_conn)
            return _csv_response(
                ["用户名", "意愿评分", "档位", f"近{data['window_days']}天登录次数",
                 f"近{data['window_days']}天练习次数", "累计在线秒数", "反馈次数"],
                [
                    [
                        u["username"], u["score"], TIER_LABELS[u["tier"]],
                        u["logins"], u["practices"], u["active_seconds"], u["feedbacks"],
                    ]
                    for u in data["users"]
                ],
                f"virtucoach_willingness_{stamp}.csv",
            )

        raise HTTPException(status_code=400, detail="type 必须是 logins/practice/users/willingness")
    finally:
        user_conn.close()
        practice_conn.close()
        feedback_conn.close()
