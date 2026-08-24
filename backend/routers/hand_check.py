"""手型检查路由（和弦 + 技巧）"""
import os
import uuid

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import FileResponse

from auth_deps import get_current_user

router = APIRouter(prefix="/api", tags=["hand_check"])

CHORDS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "..", "knowledge", "chords")


def _norm_score(value) -> float:
    """手型/图片检查的 LLM 评分是 0-10 分制，练习记录统一存 0-100，
    否则与视频分析的百分制混在一起会拉垮平均分/趋势/管理端统计。"""
    v = float(value or 0)
    return v * 10 if v <= 10 else v


def _build_report_text(result: dict) -> str:
    """把手型检查结果拼成可读报告文本（summary + 问题明细 + 练习建议）。"""
    parts = []
    summary = str(result.get("summary") or "").strip()
    if summary:
        parts.append(summary)
    issues = result.get("issues") or []
    if issues:
        parts.append("\n【问题明细】")
        for i, it in enumerate(issues, 1):
            desc = str(it.get("description") or "").strip()
            sug = str(it.get("suggestion") or "").strip()
            line = f"{i}. {desc}"
            if sug:
                line += f"\n   建议：{sug}"
            parts.append(line)
    tips = result.get("practice_tips") or []
    if tips:
        parts.append("\n【练习建议】")
        for t in tips:
            if t:
                parts.append(f"· {t}")
    return "\n".join(parts).strip()


def create_hand_check_router(hand_check_service, upload_dir, practice_db_getter=None):
    """工厂函数"""

    @router.post("/check-chord")
    async def check_chord(
        video: UploadFile = File(...),
        chord: str = Form(""),
        instrument: str = Form("guitar"),
        mode: str = Form("chord"),
        technique: str = Form(""),
        current_user=Depends(get_current_user),
    ):
        ext = os.path.splitext(video.filename or ".webm")[1].lower()
        if ext not in (".mp4", ".mov", ".webm", ".avi", ".mkv"):
            raise HTTPException(status_code=400, detail="仅支持视频格式 (MP4/MOV/WEBM等)")

        content = video.file.read()
        if len(content) > 100 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="视频不能超过 100MB")

        video_id = str(uuid.uuid4())[:12]
        video_filename = f"chord_{video_id}{ext}"
        video_path = os.path.join(upload_dir, video_filename)
        with open(video_path, "wb") as f:
            f.write(content)

        result = hand_check_service.check(video_path, chord, instrument, mode, technique)

        if current_user and practice_db_getter:
            try:
                conn = practice_db_getter()
                conn.execute(
                    """INSERT INTO practice_sessions
                       (user_id, video_filename, instrument, skill_level,
                        chord_or_track, overall_score, audio_score, hand_score,
                        report_text, mode)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [current_user.id, video_filename, instrument, "beginner",
                     chord or technique,
                     _norm_score(result.get("overall_score", 0) or result.get("overall_hand_score", 0)),
                     0,
                     _norm_score(result.get("overall_score", 0) or result.get("overall_hand_score", 0)),
                     _build_report_text(result),
                     f"chord_check" if mode == "chord" else "technique_check"],
                )
                conn.commit()
                conn.close()
            except Exception:
                pass

        return result

    @router.get("/chords")
    async def list_chords():
        chords = []
        chords_dir = os.path.normpath(CHORDS_DIR)
        if os.path.isdir(chords_dir):
            for fname in sorted(os.listdir(chords_dir)):
                if not fname.endswith(".md"):
                    continue
                fpath = os.path.join(chords_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()
                    fm = {}
                    if content.startswith("---"):
                        parts = content.split("---", 2)
                        if len(parts) >= 3:
                            for line in parts[1].strip().split("\n"):
                                if ":" in line:
                                    k, v = line.split(":", 1)
                                    fm[k.strip()] = v.strip()
                    title = fm.get("chord_name", "")
                    if not title:
                        for line in content.split("\n"):
                            if line.startswith("# "):
                                title = line[2:].strip()
                                break
                    chord_id = fm.get("id", "").replace("chord-", "")
                    chords.append({
                        "id": chord_id, "name": title or fname.replace(".md", ""),
                        "difficulty": fm.get("difficulty", ""),
                        "tags": fm.get("tags", ""),
                    })
                except Exception:
                    pass
        return {"chords": chords}

    @router.post("/analyze-hand-posture")
    async def analyze_hand_posture(
        image: UploadFile = File(...),
        instrument: str = Form("guitar"),
        check_area: str = Form(""),
        current_user=Depends(get_current_user),
    ):
        ext = os.path.splitext(image.filename or ".jpg")[1].lower()
        if ext not in (".jpg", ".jpeg", ".png"):
            raise HTTPException(status_code=400, detail="仅支持 JPG/PNG 格式")
        content = await image.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="图片不能超过 10MB")

        if not hand_check_service.vision_analyzer.available:
            raise HTTPException(status_code=503, detail="视觉 AI 未配置，无法分析手型")

        image_id = str(uuid.uuid4())[:12]
        image_filename = f"hand_check_{image_id}{ext}"
        image_path = os.path.join(upload_dir, image_filename)
        with open(image_path, "wb") as f:
            f.write(content)

        # RAG 向量搜索参考图
        user_vec = hand_check_service.video_processor.extract_landmark_vector(image_path)
        reference_db = hand_check_service.reference_db
        if user_vec:
            matched_refs = reference_db.vector_search(user_vec, top_k=3)
            if not matched_refs:
                matched_refs = reference_db.search(technique="basic_fretting")
        elif check_area:
            keywords = [k.strip() for k in check_area.replace("，", ",").split(",") if k.strip()]
            matched_refs = reference_db.search(keywords=keywords)
        else:
            matched_refs = reference_db.search(technique="basic_fretting")
            if not matched_refs:
                matched_refs = reference_db.list_all()[:3]

        ref_image_paths = []
        reference_images_data = []
        for ref in matched_refs[:3]:
            ref_path = os.path.join(reference_db.images_dir, ref["image_path"])
            if os.path.exists(ref_path):
                ref_image_paths.append(ref_path)
            ref_tags = ref.get("tags", [])
            reference_images_data.append({
                "id": ref["id"], "description": ref["description"],
                "body_part": ref["body_part"], "hand": ref["hand"],
                "technique": ref["technique"], "tags": ref["tags"],
                "image_url": reference_db.image_url(ref),
                "chord_name": "",
                "chord_matched": False,
                "is_topdown": "俯视图" in str(ref_tags),
                "is_front": "正视图" in str(ref_tags),
            })

        result = hand_check_service.vision_analyzer.analyze_hand_posture_image(
            image_path=image_path, instrument=instrument,
            reference_paths=ref_image_paths if ref_image_paths else None,
            check_area=check_area,
        )

        response = {
            "success": result.get("success", False),
            "user_image_url": f"/uploads/{image_filename}",
            "hands_detected": result.get("hands_detected", False),
            "handedness": result.get("handedness", ""),
            "overall_score": result.get("overall_hand_score", 0),
            "issues": result.get("issues", []),
            "reference_images": reference_images_data,
            "provider": result.get("provider", ""),
        }

        # 图片检查是一次性诊断，不计入练习记录（避免10分制数据混入百分制统计）
        return response

    return router
