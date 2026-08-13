"""Practice history router — factory function pattern."""

from fastapi import APIRouter, HTTPException, Depends

from practice_db import get_db
from practice_models import PracticeSessionResponse, PracticeStatsResponse
from auth_deps import require_user


def create_practice_router():
    router = APIRouter(prefix="/api", tags=["practice"])

    @router.get("/practice/stats")
    async def get_stats(current_user=Depends(require_user)):
        conn = get_db()
        try:
            rows = conn.execute(
                """SELECT overall_score, mode FROM practice_sessions
                   WHERE user_id = ? ORDER BY created_at DESC""",
                [current_user.id],
            ).fetchall()

            total = len(rows)
            scores = [r["overall_score"] for r in rows if r["overall_score"] > 0]
            recent = scores[:10][::-1]  # chronological for chart

            mode_counts = {}
            for r in rows:
                m = r["mode"]
                mode_counts[m] = mode_counts.get(m, 0) + 1

            return PracticeStatsResponse(
                total_sessions=total,
                average_score=round(sum(scores) / len(scores), 1) if scores else 0,
                highest_score=round(max(scores), 1) if scores else 0,
                lowest_score=round(min(scores), 1) if scores else 0,
                recent_scores=recent,
                by_mode=mode_counts,
            )
        finally:
            conn.close()

    @router.get("/practice/sessions")
    async def list_sessions(
        page: int = 1,
        per_page: int = 20,
        mode: str = "",
        current_user=Depends(require_user),
    ):
        conn = get_db()
        try:
            conditions = ["user_id = ?"]
            params = [current_user.id]
            if mode:
                conditions.append("mode = ?")
                params.append(mode)

            where = f"WHERE {' AND '.join(conditions)}"
            total = conn.execute(
                f"SELECT COUNT(*) FROM practice_sessions {where}", params
            ).fetchone()[0]

            offset = (page - 1) * per_page
            rows = conn.execute(
                f"SELECT * FROM practice_sessions {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                params + [per_page, offset],
            ).fetchall()

            return {
                "items": [PracticeSessionResponse.from_row(r) for r in rows],
                "total": total,
                "page": page,
                "per_page": per_page,
            }
        finally:
            conn.close()

    @router.get("/practice/sessions/{session_id}")
    async def get_session(session_id: int, current_user=Depends(require_user)):
        conn = get_db()
        try:
            row = conn.execute(
                "SELECT * FROM practice_sessions WHERE id = ? AND user_id = ?",
                [session_id, current_user.id],
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="记录不存在")
            return PracticeSessionResponse.from_row(row)
        finally:
            conn.close()

    @router.delete("/practice/sessions/{session_id}")
    async def delete_session(session_id: int, current_user=Depends(require_user)):
        conn = get_db()
        try:
            row = conn.execute(
                "SELECT * FROM practice_sessions WHERE id = ? AND user_id = ?",
                [session_id, current_user.id],
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="记录不存在")
            conn.execute("DELETE FROM practice_sessions WHERE id = ?", [session_id])
            conn.commit()
            return {"ok": True}
        finally:
            conn.close()

    return router
