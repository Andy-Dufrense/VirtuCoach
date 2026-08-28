"""视频分析路由"""
import os
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException, Depends, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, Response

from auth_deps import get_current_user

router = APIRouter(prefix="/api", tags=["analysis"])

MAX_TASKS = 100  # 最大任务数上限
TASK_TTL_HOURS = 2  # 任务保留时间（小时）


def _parse_range(spec: str, size: int):
    """解析单段 bytes Range：'a-b' / 'a-' / '-suffix'，返回 (start, end)。

    非法或越界抛 ValueError（由调用方转 416）。多段 Range 只取第一段——
    浏览器对单段 206 响应完全可接受。
    """
    spec = spec.strip()
    if "," in spec:
        spec = spec.split(",")[0].strip()
    if spec.startswith("-"):
        suffix = int(spec[1:])  # 非法字符抛 ValueError
        if suffix <= 0:
            raise ValueError("empty suffix range")
        return max(0, size - suffix), size - 1
    start_str, _, end_str = spec.partition("-")
    start = int(start_str)
    if start >= size:
        raise ValueError("range start beyond file size")
    if end_str == "":
        return start, size - 1
    end = min(int(end_str), size - 1)
    if end < start:
        raise ValueError("range end before start")
    return start, end


async def _stream_file(path: str, start: int, end: int):
    """按 1MB 块流式读取文件 [start, end] 区间（aiofiles，不阻塞事件循环）。"""
    import aiofiles
    chunk_size = 1024 * 1024
    async with aiofiles.open(path, "rb") as f:
        await f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = await f.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


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
    """安全删除任务，同时清理对应视频文件和回放代理。"""
    task = tasks.get(task_id)
    if task:
        for key in ("video_path", "proxy_path"):
            path = task.get(key, "")
            if path and os.path.exists(path):
                try:
                    os.remove(path)
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
        current_user=Depends(get_current_user),
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
            "user_id": current_user.id if current_user else None,
        }

        background_tasks.add_task(
            analysis_service.run_queued, task_id, video_path, instrument, level,
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
        # 可播放文件存在时给出回放地址（代理优先：HEVC 原片浏览器解不了）
        playable = task.get("proxy_path") or task.get("video_path") or ""
        if task["status"] == "completed" and playable and os.path.isfile(playable):
            resp["video_url"] = f"/api/task/{task_id}/video"
        return resp

    @router.get("/task/{task_id}/video")
    async def get_task_video(task_id: str, request: Request):
        """流式提供任务视频（H.264 代理优先，否则原片），支持 HTTP Range。

        前端「原视频回放」靠它跳转错误时间点：starlette 0.27 的 FileResponse
        不支持 Range（206），<video> 无法拖动/跳转，所以这里手写分块流。
        安全：只通过 tasks 字典查文件路径，绝不从 URL 输入拼路径。
        """
        task = router.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        path = task.get("proxy_path") or task.get("video_path") or ""
        if not path or not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="视频文件不存在（可能已被清理）")
        size = os.path.getsize(path)

        range_header = (request.headers.get("range") or "").strip()
        if not range_header.startswith("bytes="):
            return StreamingResponse(
                _stream_file(path, 0, size - 1),
                media_type="video/mp4",
                headers={"Accept-Ranges": "bytes", "Content-Length": str(size)},
            )
        try:
            start, end = _parse_range(range_header[6:], size)
        except (ValueError, TypeError):
            return Response(
                status_code=416,
                headers={"Content-Range": f"bytes */{size}"},
            )
        return StreamingResponse(
            _stream_file(path, start, end),
            status_code=206,
            media_type="video/mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Range": f"bytes {start}-{end}/{size}",
                "Content-Length": str(end - start + 1),
            },
        )

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
