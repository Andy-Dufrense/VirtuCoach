"""视频分析路由"""
import os
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter(prefix="/api", tags=["analysis"])

MAX_TASKS = 100  # 最大任务数上限
TASK_TTL_HOURS = 2  # 任务保留时间（小时）


def _cleanup_old_tasks(tasks: dict):
    """清理过期任务和超出上限的任务。"""
    now = datetime.now()
    # 1. 删除超过 TTL 的已完成/失败任务
    expired = []
    for tid, t in tasks.items():
        created = t.get("created_at", "")
        try:
            created_dt = datetime.fromisoformat(created)
            if now - created_dt > timedelta(hours=TASK_TTL_HOURS):
                expired.append(tid)
        except (ValueError, TypeError):
            continue
    for tid in expired:
        _remove_task(tid, tasks)

    # 2. 超出上限时删除最旧的已完成任务
    if len(tasks) > MAX_TASKS:
        completed = [(tid, t) for tid, t in tasks.items()
                     if t.get("status") in ("completed", "failed")]
        completed.sort(key=lambda x: x[1].get("created_at", ""))
        excess = len(tasks) - MAX_TASKS
        for tid, _ in completed[:max(excess, 1)]:
            _remove_task(tid, tasks)


def _remove_task(task_id: str, tasks: dict):
    """安全删除任务，同时清理对应视频文件。"""
    task = tasks.get(task_id)
    if task:
        video_path = task.get("video_path", "")
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except OSError:
                pass
    tasks.pop(task_id, None)


def create_analysis_router(analysis_service, upload_dir, frontend_dir):
    """工厂函数：注入依赖后返回路由"""

    @router.post("/analyze")
    async def analyze_video(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        instrument: str = Form("piano"),
        level: str = Form("beginner"),
        title: Optional[str] = Form(None),
        capo: int = Form(0),
    ):
        task_id = str(uuid.uuid4())
        file_ext = Path(file.filename).suffix if file.filename else ".mp4"
        video_path = os.path.join(upload_dir, f"{task_id}{file_ext}")

        # 清理旧任务，防止内存泄漏
        _cleanup_old_tasks(router.tasks)

        content = await file.read()
        with open(video_path, "wb") as f:
            f.write(content)

        router.tasks[task_id] = {
            "id": task_id,
            "status": "uploaded",
            "progress": 0,
            "message": "文件已上传，开始分析...",
            "video_path": video_path,
            "instrument": instrument,
            "level": level,
            "title": title or os.path.splitext(file.filename or "untitled")[0],
            "created_at": __import__("datetime").datetime.now().isoformat(),
            "result": None,
        }

        background_tasks.add_task(
            analysis_service.run, task_id, video_path, instrument, level,
            router.tasks[task_id]["title"], router.tasks, capo
        )
        return {"task_id": task_id, "status": "uploaded"}

    @router.get("/task/{task_id}")
    async def get_task_status(task_id: str):
        task = router.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        resp = {
            "id": task["id"], "status": task["status"], "progress": task["progress"],
            "message": task["message"], "title": task["title"],
            "instrument": task["instrument"], "level": task["level"],
            "created_at": task["created_at"],
        }
        if task.get("timing"):
            resp["timing"] = task["timing"]
        if task["result"]:
            resp["result"] = task["result"]
        if task.get("error_detail"):
            resp["error_detail"] = task["error_detail"]
        return resp

    @router.get("/tasks")
    async def list_tasks():
        summary = [
            {"id": tid, "status": t["status"], "progress": t["progress"],
             "title": t["title"], "created_at": t["created_at"]}
            for tid, t in router.tasks.items()
        ]
        return {"tasks": summary}

    return router


# tasks 字典放在 router 上（共享状态）
router.tasks = {}
