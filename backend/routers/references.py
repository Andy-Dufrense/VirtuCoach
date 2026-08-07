"""参考图管理路由"""
import os
import json
import base64
from typing import Optional

import cv2
import numpy as np
import httpx
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

from logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["references"])


def create_references_router(reference_db, video_processor, vision_analyzer, upload_dir):
    """工厂函数"""

    @router.get("/references")
    async def list_references(
        keyword: Optional[str] = None, hand: Optional[str] = None,
        technique: Optional[str] = None, body_part: Optional[str] = None,
    ):
        kwargs = {}
        if hand: kwargs["hand"] = hand
        if technique: kwargs["technique"] = technique
        if body_part: kwargs["body_part"] = body_part
        if keyword:
            kwargs["keywords"] = [k.strip() for k in keyword.split(",") if k.strip()]
        results = reference_db.search(**kwargs)
        for r in results:
            r["image_url"] = reference_db.image_url(r)
        return {"references": results}

    @router.post("/references")
    async def upload_reference(
        image: UploadFile = File(...),
        instrument: str = Form("guitar"), hand: str = Form("双手"),
        technique: str = Form("basic_fretting"), body_part: str = Form("full_hand"),
        view_direction: str = Form("前"), description: str = Form(...),
        tags: str = Form(""),
    ):
        ext = os.path.splitext(image.filename or ".jpg")[1].lower()
        if ext not in (".jpg", ".jpeg", ".png"):
            raise HTTPException(status_code=400, detail="仅支持 JPG/PNG 格式")
        content = await image.read()
        if len(content) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="图片不能超过 5MB")

        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        metadata = {
            "instrument": instrument, "hand": hand, "technique": technique,
            "body_part": body_part, "view_direction": view_direction,
            "description": description, "tags": tag_list,
        }
        ref_id = reference_db.add(metadata, content, ext)
        try:
            ref_image_path = os.path.join(reference_db.images_dir, f"{ref_id}{ext}")
            vec = video_processor.extract_landmark_vector(ref_image_path)
            if vec:
                reference_db.update_landmark_vector(ref_id, vec)
        except Exception as e:
            logger.warning(f"关键点提取失败: {e}")

        ref = reference_db.get_by_id(ref_id)
        ref["image_url"] = reference_db.image_url(ref)
        return {"id": ref_id, "reference": ref}

    @router.post("/references/analyze")
    async def analyze_reference_image(image: UploadFile = File(...)):
        ext = os.path.splitext(image.filename or ".jpg")[1].lower()
        if ext not in (".jpg", ".jpeg", ".png"):
            raise HTTPException(status_code=400, detail="仅支持 JPG/PNG 格式")
        content = await image.read()
        if len(content) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="图片不能超过 5MB")
        if not vision_analyzer.available:
            raise HTTPException(status_code=503, detail="视觉 AI 未配置")

        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            h, w = img.shape[:2]
            if w > 800:
                scale = 800.0 / w
                img = cv2.resize(img, (800, int(h * scale)))
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
            image_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        else:
            image_b64 = base64.b64encode(content).decode("utf-8")

        prompt = """你是一位资深吉他教师。请分析这张正确手型参考图，提取以下元数据。
返回严格 JSON：
{
  "hand": "左手/右手/双手",
  "technique": "以下之一: basic_fretting, natural_harmonics, artificial_harmonics, pm, am, tapping, muting, bend, slide, vibrato, hammer_on, pull_off",
  "body_part": "wrist/thumb/index_finger/middle_finger/ring_finger/pinky/full_hand",
  "view_direction": "前(正面)/侧(侧面)/上(上方)",
  "description": "一句话描述图中正确手型的要点（中文，30字以内）",
  "tags": ["标签1", "标签2", "标签3"]
}
只返回 JSON，不要 markdown 代码块"""

        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=os.getenv("DASHSCOPE_API_KEY", ""),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                timeout=httpx.Timeout(60.0, connect=15.0, read=60.0, write=15.0),
            )
            resp = client.chat.completions.create(
                model=os.getenv("VISION_MODEL", "qwen-vl-max"),
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": prompt},
                ]}],
                temperature=0.3, max_tokens=1024,
                timeout=httpx.Timeout(60.0, connect=15.0, read=60.0),
            )
            text = resp.choices[0].message.content
            import re as _re
            parsed = None
            try:
                parsed = json.loads(text)
            except:
                m = _re.search(r'\{[\s\S]*\}', text)
                if m:
                    try:
                        parsed = json.loads(m.group(0))
                    except:
                        pass
            if not parsed:
                return {"error": "AI 分析失败，请手动填写元数据", "raw": text[:500]}
            return {
                "suggested": {
                    "hand": parsed.get("hand", "双手"),
                    "technique": parsed.get("technique", "basic_fretting"),
                    "body_part": parsed.get("body_part", "full_hand"),
                    "view_direction": parsed.get("view_direction", "前"),
                    "description": parsed.get("description", ""),
                    "tags": parsed.get("tags", []),
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"视觉分析失败: {str(e)}")

    @router.post("/references/backfill-vectors")
    async def backfill_vectors():
        unvec = reference_db.get_unvectorized()
        if not unvec:
            return {"status": "ok", "message": "所有参考图已有向量", "count": 0}
        success = failed = 0
        for ref in unvec:
            img_path = os.path.join(reference_db.images_dir, ref["image_path"])
            if not os.path.exists(img_path):
                failed += 1
                continue
            try:
                vec = video_processor.extract_landmark_vector(img_path)
                if vec:
                    reference_db.update_landmark_vector(ref["id"], vec)
                    success += 1
                else:
                    failed += 1
            except Exception:
                failed += 1
        return {"status": "ok", "total": len(unvec), "success": success, "failed": failed}

    @router.delete("/references/{ref_id}")
    async def delete_reference(ref_id: str):
        ok = reference_db.delete(ref_id)
        if not ok:
            raise HTTPException(status_code=404, detail="参考图不存在")
        return {"status": "deleted"}

    @router.get("/reference-images/{filename}")
    async def get_reference_image(filename: str):
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="非法文件名")
        file_path = os.path.join(reference_db.images_dir, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="图片不存在")
        return FileResponse(file_path)

    return router
