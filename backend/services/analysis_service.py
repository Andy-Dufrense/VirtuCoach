"""
视频分析服务：编排完整的分析流水线。

从 main.py 的 run_analysis() 提取，原来 530 行单体函数，
重构为 AnalysisService 类，每个步骤有明确边界。
"""

import os
import json
import time
import numpy as np
import asyncio
from typing import Dict, Optional

from services.report_postprocessor import (
    fix_placeholders,
    force_hand_section,
    force_audio_section,
    force_audio_section_nosound,
)
from services.audio_diagnosis import diagnose
from config import UPLOAD_DIR
from pipeline import (
    FilterContext,
    create_freeplay_pipeline,
    merge_vision_issues,
    dedup_by_time,
)
from logging_config import get_logger

logger = get_logger(__name__)


def _image_dhash(image_path: str, hash_size: int = 8) -> str:
    """计算图片的差异哈希（dhash），用于图片内容去重。

    算法：缩放到 hash_size+1 × hash_size 灰度图，
    逐行比较相邻像素，左>右得1，生成十六进制字符串。
    """
    from PIL import Image
    img = Image.open(image_path).convert("L")
    img = img.resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = list(img.getdata())
    hash_bits = []
    for row in range(hash_size):
        row_start = row * (hash_size + 1)
        for col in range(hash_size):
            left = pixels[row_start + col]
            right = pixels[row_start + col + 1]
            hash_bits.append("1" if left > right else "0")
    hash_int = int("".join(hash_bits), 2)
    return f"{hash_int:016x}"


def _hamming_distance(h1: str, h2: str) -> int:
    """两个十六进制哈希的汉明距离。"""
    if len(h1) != len(h2):
        return 999
    diff = int(h1, 16) ^ int(h2, 16)
    return bin(diff).count("1")


class AnalysisService:
    """视频分析流水线编排器。依赖注入所有底层模块。"""

    # 七和弦→三和弦参考图回退（初学者七和弦参考量少）
    # E7/A7/B7 是属七和弦，回退到对应的大三和弦（E/A/B），不是小和弦
    SEVENTH_TO_TRIAD = {
        "chord-Bm7": "chord-Bm", "chord-Cmaj7": "chord-C",
        "chord-Am7": "chord-Am", "chord-Dm7": "chord-Dm",
        "chord-D7": "chord-D", "chord-E7": "chord-E",
        "chord-G7": "chord-G", "chord-A7": "chord-A",
        "chord-B7": "chord-B", "chord-C7": "chord-C",
        "chord-Emaj7": "chord-E", "chord-Fmaj7": "chord-F",
        "chord-Gmaj7": "chord-G", "chord-Amaj7": "chord-Am",
        "chord-Dmaj7": "chord-D",
    }

    def __init__(self, audio_analyzer, video_processor, deepseek_agent,
                 audio_separator, vision_analyzer, reference_db, knowledge_db=None,
                 practice_db_getter=None):
        self.audio_analyzer = audio_analyzer
        self.video_processor = video_processor
        self.deepseek_agent = deepseek_agent
        self.audio_separator = audio_separator
        self.vision_analyzer = vision_analyzer
        self.reference_db = reference_db
        self.knowledge_db = knowledge_db
        self.practice_db_getter = practice_db_getter

    async def run(self, task_id: str, video_path: str, instrument: str,
                  level: str, title: str, tasks: dict, capo: int = 0) -> None:
        """执行完整分析流水线，结果写入 tasks[task_id]。"""
        try:
            tasks[task_id]["status"] = "processing"
            tasks[task_id]["message"] = "正在分析音频..."
            tasks[task_id]["progress"] = 8
            tasks[task_id]["timing"] = {"audio": 0, "hand": 0, "ai": 0, "total": 0}

            t_start = time.time()

            # ===== Step 1.5: 快速手区估算（先于音频分析，用于品位分配） =====
            hand_zone_center = await self._run_quick_hand_zone(video_path, tasks, task_id)

            # ===== Step 2: 音频分析（始终用原始视频） =====
            # 不做事先人声分离：Demucs 分离质量不可控，会破坏吉他谐波导致
            # chroma 和弦检测严重退化。弹唱视频中的人声干扰由下游
            # _resolve_chord_intent + _filter_notes_by_chords 处理——
            # 人声音符不属于任何和弦 pcset，天然会被标记为幻音过滤。
            audio_result = await self._run_audio_analysis(
                video_path, instrument, tasks, task_id,
                hand_zone_center=hand_zone_center, capo=capo,
            )
            t_audio_end = time.time()
            tasks[task_id]["timing"]["audio"] = round(t_audio_end - t_start, 1)

            # ===== Step 3: 处理无音频情况 =====
            if self._is_no_audio(audio_result):
                self._handle_no_audio(task_id, video_path, instrument, audio_result, tasks)
                return

            # ===== Step 4: 和弦检测 & 技巧推断 + 品位修正 =====
            capo = audio_result.get("detected_capo", 0)
            logger.info(f"和弦检测: detected_capo={capo}, notes={len(audio_result.get('notes', []))}个")
            detected_chords, capo = self._detect_chords(audio_result, capo)
            # 用检测到的和弦修正不合理的品位分配
            notes = audio_result.get("notes", [])
            if notes and detected_chords:
                self._correct_frets_by_chords(notes, detected_chords, capo)
            technique_hints = self._detect_technique_hints_from_audio(audio_result)

            # ===== Step 5: 视频手型检测 =====
            tasks[task_id]["progress"] = 35
            tasks[task_id]["message"] = "正在检测手型..."
            # 代理视频留存路径：HEVC/高分辨率原片浏览器解不了，分析用的 H.264
            # 代理顺带保留供结果页回放（零额外转码成本），随任务 TTL 一起清理
            proxy_dest = os.path.join(UPLOAD_DIR, f"{task_id}_proxy.mp4")
            video_data = await self._run_hand_detection(
                video_path, audio_result, instrument, technique_hints,
                detected_chords=detected_chords,
                proxy_dest=proxy_dest,
            )
            tasks[task_id]["proxy_path"] = video_data.get("proxy_path") or ""

            # ===== Step 5.2: Qwen VL 指板视觉和弦检测 =====
            tasks[task_id]["message"] = "正在用AI视觉识别和弦..."
            vl_chords = await self._detect_vl_chords(
                video_data, detected_chords, instrument
            )
            if vl_chords:
                logger.info(f"VL和弦验证: {len(vl_chords)}个: "
                            f"{[(c.get('vl_chosen','?'), c.get('time',0)) for c in vl_chords]}")

            # ===== Step 5.5: 视频手型和弦检测 & 融合 =====
            detected_chords = self._fuse_chord_detection(
                detected_chords, video_data, audio_result, capo, vl_chords=vl_chords
            )
            # ===== Step 5.6: 视觉和弦交叉验证（AirStrum风格landmark→和弦）=====
            detected_chords = self._cross_validate_chords_visual(
                detected_chords, video_data
            )
            # 用融合后的和弦重新修正品位分配
            if notes and detected_chords:
                self._correct_frets_by_chords(notes, detected_chords, capo)

            # Am7 → Am 映射：Am7 参考图通常也是 Am 的指法，避免匹错参考图
            for c in detected_chords:
                if c.get("chord_id") == "Am7":
                    c["chord_id"] = "Am"
                    c["chord_name"] = c.get("chord_name", "Am7和弦").replace("Am7", "Am")
                    logger.info(f"和弦映射: Am7 → Am (t={c.get('time',0):.1f}s)")

            # ===== Step 6: 手型过滤（统一管道）=====
            first_note_time = self._get_first_note_time(audio_result)
            hand_filter = create_freeplay_pipeline()
            filter_ctx = FilterContext(
                mode="freeplay",
                first_note_time=first_note_time,
                technique_segments=audio_result.get("technique_segments", []),
                user_level=level,
            )
            raw_issues = video_data.get("hand_issues", [])
            # 诊断：追踪「太直/太竖直」类问题在各阶段的存留
            diag_straight = [h for h in raw_issues if "太直" in h.get("问题","") or "太竖直" in h.get("问题","")]
            diag_straight_ts = [(h.get("时间",0), h.get("问题","")[:50]) for h in diag_straight]
            logger.info(f"手型管道输入: {len(raw_issues)}条, 其中太直/太竖直={len(diag_straight)}条 {diag_straight_ts}")
            logger.info(f"  first_note={first_note_time:.1f}s, level={level}")
            deduped = hand_filter.apply(raw_issues, filter_ctx)
            diag_straight_out = [h for h in deduped if "太直" in h.get("问题","") or "太竖直" in h.get("问题","")]
            diag_straight_out_ts = [(h.get("时间",0), h.get("问题","")[:60]) for h in diag_straight_out]
            logger.info(f"手型管道输出: {len(deduped)}条, 其中太直/太竖直={len(diag_straight_out)}条 {diag_straight_out_ts}")
            logger.info(f"  过滤日志: {hand_filter._filter_log}")
            video_data["hand_issues"] = deduped
            tasks[task_id]["timing"]["hand"] = round(time.time() - t_audio_end, 1)

            # ===== Step 6.5a: 过滤不相关手指问题 =====
            deduped = self._filter_irrelevant_fingers(deduped, detected_chords)
            diag_after_finger = [h for h in deduped if "太直" in h.get("问题","") or "太竖直" in h.get("问题","")]
            logger.info(f"finger_filter后: {len(deduped)}条, 太直/太竖直={len(diag_after_finger)}条")
            video_data["hand_issues"] = deduped

            # ===== Step 6.5b: 和弦感知分组（同一和弦段内的问题合并）=====
            deduped = self._group_by_chord(deduped, detected_chords)
            # ===== Step 6.5c: 问题类型去重（同类型问题只留1条，合并时间点）=====
            deduped = self._dedup_by_problem_type(deduped)
            diag_after_dedup = [h for h in deduped if "太直" in h.get("问题","") or "太竖直" in h.get("问题","")]
            logger.info(f"group+dedup后: {len(deduped)}条, 太直/太竖直={len(diag_after_dedup)}条")
            video_data["hand_issues"] = deduped

            # ===== Step 6.6: AI 审核节点（Phase 2d）=====
            ai_dismissed = []
            if deduped and self.knowledge_db:
                try:
                    deduped, ai_dismissed = self._ai_review_hand_issues(deduped, detected_chords, level)
                    diag_after_ai = [h for h in deduped if "太直" in h.get("问题","") or "太竖直" in h.get("问题","")]
                    logger.info(f"AI审核后: {len(deduped)}条, 太直/太竖直={len(diag_after_ai)}条, dismissed={len(ai_dismissed)}")
                    video_data["hand_issues"] = deduped
                except Exception as e:
                    logger.warning(f"AI审核失败，保持原始检测结果: {e}")

            # ===== Step 7: 截图标注（每个问题类型最多1张截图）=====
            video_data = self._annotate_snapshots(video_data, audio_result, deduped, first_note_time)

            # ===== Step 8: 参考图匹配 & Vision AI =====
            reference_images = []
            if deduped and self.vision_analyzer.available:
                reference_images = self._match_references(deduped, detected_chords, technique_hints, video_data, audio_result, vl_chords=vl_chords)
                all_snaps = video_data.get("snapshots", [])
                if all_snaps:
                    tasks[task_id]["progress"] = 40
                    tasks[task_id]["message"] = "AI 视觉分析手型中..."
                    vision_result = self._run_vision_analysis(all_snaps, instrument, reference_images, deduped, video_data)
                    if vision_result.get("vision_issues"):
                        video_data = self._merge_vision_results(video_data, deduped, vision_result)

            # ===== Step 9: 手型状态判定 =====
            video_data = self._determine_hand_status(video_data)
            video_data["detected_chords"] = detected_chords

            # ===== Step 10: AI 报告生成 =====
            tasks[task_id]["progress"] = 75
            tasks[task_id]["message"] = "AI 老师评估中..."

            # 运行音频诊断引擎（Layer 2：信号 → 结构化诊断）
            audio_diagnosis = diagnose(audio_result)

            report = self.deepseek_agent.generate_report(
                audio_result=audio_result,
                video_data=video_data,
                instrument=instrument,
                level=level,
                title=title,
                mode="freeplay",
                audio_diagnosis=audio_diagnosis,
                ai_dismissed=ai_dismissed,
            )
            # 保存原始输入，供流式报告端点使用
            tasks[task_id]["_report_inputs"] = {
                "audio_result": audio_result,
                "video_data": video_data,
                "instrument": instrument,
                "level": level,
                "title": title,
                "audio_diagnosis": audio_diagnosis,
            }
            tasks[task_id]["timing"]["ai"] = round(time.time() - t_audio_end - tasks[task_id]["timing"]["hand"], 1)
            tasks[task_id]["timing"]["total"] = round(time.time() - t_start, 1)

            # ===== Step 11: 后处理（评分锚定 + 章节覆盖）=====
            result = self._postprocess_report(
                report, audio_result, video_data, detected_chords, reference_images, audio_diagnosis
            )
            tasks[task_id]["progress"] = 100
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["message"] = "分析完成！"
            tasks[task_id]["result"] = result

            # 已登录用户自动保存练习记录
            # 注意：result 结构为 {"score": {overall/pitch/rhythm/technique}, "report_markdown": ...}
            # report_text 必须存字符串（report 本身是 dict，直接绑定会 InterfaceError）
            user_id = tasks[task_id].get("user_id")
            if user_id and self.practice_db_getter:
                try:
                    scores = result.get("score", {}) if isinstance(result, dict) else {}
                    overall = scores.get("overall") or 0
                    pitch = scores.get("pitch") or 0
                    rhythm = scores.get("rhythm") or 0
                    hand = scores.get("technique") or 0
                    report_md = ""
                    if isinstance(result, dict):
                        report_md = result.get("report_markdown") or result.get("summary") or ""
                    conn = self.practice_db_getter()
                    conn.execute(
                        """INSERT INTO practice_sessions
                           (user_id, video_filename, instrument, skill_level,
                            chord_or_track, overall_score, audio_score, hand_score,
                            report_text, mode)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        [user_id, os.path.basename(video_path), instrument, level,
                         title or "",
                         float(overall),
                         float(round((pitch + rhythm) / 2)),  # 音频分 = 音准与节奏的均值
                         float(hand),
                         str(report_md), "video_analysis"],
                    )
                    conn.commit()
                    conn.close()
                    logger.info(f"练习记录已保存: user_id={user_id}, overall={overall}")
                except Exception:
                    logger.warning("保存练习记录失败", exc_info=True)

        except Exception as e:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["message"] = f"分析失败: {str(e)}"
            import traceback
            tasks[task_id]["error_detail"] = traceback.format_exc()

    # ====== 私有步骤方法 ======

    async def _run_quick_hand_zone(self, video_path, tasks, task_id):
        """快速估算左手在指板上的位置"""
        try:
            loop = asyncio.get_running_loop()
            zone = await loop.run_in_executor(
                None, self.video_processor.estimate_hand_zone, video_path, 8
            )
            if zone is not None:
                logger.debug(f"快速手区估算完成: ~{zone:.0f}品")
            return zone
        except Exception as e:
            logger.warning(f"快速手区估算失败: {e}")
            return None

    async def _run_audio_analysis(self, video_path, instrument, tasks, task_id,
                                   hand_zone_center=None, capo=0):
        """异步执行音频分析 + 进度更新"""

        def update_progress(p, msg):
            tasks[task_id]["progress"] = p
            tasks[task_id]["message"] = msg

        async def run_audio():
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self.audio_analyzer.analyze, video_path, instrument, capo, hand_zone_center
            )

        async def progress_reporter():
            for p, msg in [(12, "提取音频中..."), (18, "分析音符中..."),
                            (25, "检测节奏..."), (30, "对比演奏错误...")]:
                await asyncio.sleep(2)
                if tasks[task_id]["progress"] < p:
                    update_progress(p, msg)

        audio_task = asyncio.create_task(run_audio())
        progress_task = asyncio.create_task(progress_reporter())
        result = await audio_task
        progress_task.cancel()
        return result

    def _is_no_audio(self, audio_result):
        errors = audio_result.get("errors", [])
        stats = audio_result.get("stats", {})
        return (
            any(e.get("type") in ("audio_extraction_failed", "no_notes_detected") for e in errors)
            or stats.get("total_notes", 0) == 0
        )

    def _handle_no_audio(self, task_id, video_path, instrument, audio_result, tasks):
        tasks[task_id]["progress"] = 30
        tasks[task_id]["message"] = "未检测到音频，尝试检测手型..."
        try:
            proxy_dest = os.path.join(UPLOAD_DIR, f"{task_id}_proxy.mp4")
            video_data = self.video_processor.process(
                video_path, [], instrument, force_hand="left", proxy_dest=proxy_dest)
            tasks[task_id]["proxy_path"] = video_data.get("proxy_path") or ""
            hand_issues_raw = video_data.get("hand_issues", [])
            analysis_parsed = {}
            try:
                analysis_parsed = json.loads(video_data.get("frame_analysis_data", "{}"))
            except:
                pass
            hands_detected = analysis_parsed.get("hands_found", False)
            if hands_detected and not hand_issues_raw:
                hand_issues_raw = [{"哪只手": "双手", "问题": "手部已检测到，手型基本正常", "时间": 0}]
            hand_status = "nHand" if not hands_detected else ("issues" if hand_issues_raw else "ok")
            snapshots = video_data.get("snapshots", [])
        except Exception as e:
            logger.warning(f"nSound下手型检测失败: {e}")
            hand_issues_raw, hand_status, snapshots = [], "nHand", []
            hands_detected = False

        score_tech = None if hand_status == "nHand" else 70
        tasks[task_id].update({
            "status": "completed", "progress": 100,
            "message": "分析完成（未检测到音频" + ("，手型已检测" if hands_detected else "，无手部画面") + "）",
            "result": {
                "score": {"overall": None, "pitch": None, "rhythm": None, "technique": score_tech},
                "audio_errors": [], "audio_status": "nSound",
                "hand_issues": hand_issues_raw, "hand_status": hand_status,
                "snapshots": snapshots,
                "report_markdown": (
                    "## ⚠️ 未检测到音频\n\n> 您上传的视频没有声音或音频轨道为空。\n>\n> 请确认：\n> - 录制的视频包含清晰的乐器声音\n> - 麦克风未被遮挡或关闭\n\n"
                    + ("## ✋ 手型\n\n> 📷 本次未捕捉到手部画面\n" if hand_status == "nHand"
                       else "## ✋ 手型\n\n> ✅ 手部检测正常，但因为没有音频，无法做完整评估。请重新录制带声音的视频。\n")
                ),
                "practice_tips": [], "summary": "视频无音频" + ("，手型已检测" if hands_detected else "，无手部画面"),
                "audio_stats": audio_result.get("stats", {}), "reference_images": [],
                "content_type": "uncertain",
            }
        })

    def _detect_chords(self, audio_result, capo):
        """检测和弦，优先使用 chroma 识别结果，回退到 basic-pitch 方案。

        Returns (chords, effective_capo).
        """
        # 优先使用 audio_analyzer 的 chroma 和弦识别结果
        chroma_chords = audio_result.get("detected_chords", [])
        if chroma_chords and len(chroma_chords) > 0:
            names = [f"{c['chord_name']}({c['confidence']:.2f})" for c in chroma_chords[:6]]
            logger.info(f"和弦检测(chroma): {len(chroma_chords)}个, {names}")
            return chroma_chords, capo

        # 回退到 basic-pitch → Jaccard 拼装
        try:
            from chord_analyzer import ChordAnalyzer
            notes = audio_result.get("notes", [])
            if notes and len(notes) >= 2:
                chords = ChordAnalyzer.detect_chords_from_notes(notes, capo)
                if chords:
                    names = [f"{c['chord_name']}({c['confidence']:.2f})" for c in chords[:6]]
                    logger.info(f"和弦检测({len(notes)}个音符, capo={capo}): {names}")

                    # 只在capo=0时尝试推断（capo>0说明_detect_capo已工作，比推断更可靠）
                    if capo == 0:
                        inferred = ChordAnalyzer.infer_capo_from_chords(notes, chords)
                        logger.info(f"和弦名推断: capo推断={inferred} (音频检测={capo})")
                        if inferred > 0:
                            logger.info(f"以推断值重新检测 capo={inferred}...")
                            chords2 = ChordAnalyzer.detect_chords_from_notes(notes, inferred, min_confidence=0.70)
                            if chords2:
                                names2 = [f"{c['chord_name']}({c['confidence']:.2f})" for c in chords2[:6]]
                                logger.info(f"和弦重检测: {names2}")
                                return chords2, inferred
                            else:
                                logger.warning(f"capo={inferred}重检测无结果，保持capo={capo}的结果")
                    return chords, capo
                else:
                    logger.info(f"和弦检测({len(notes)}个音符): 未匹配到和弦")
        except Exception as e:
            logger.warning(f"和弦检测失败: {e}")
        return [], capo

    def _consolidate_chord_windows(self, merged, all_detected):
        """窗口化合并：将 chroma 逐帧 flicker 段合并为 2s 窗口的主导和弦。

        分解和弦下每帧只含 1-2 个音，chroma 会在共享音高的和弦间快速切换
        （如 G/Em/G7/Cmaj7）。此方法按 ~2s 窗口取主导和弦，再合并相邻同和弦窗口。
        """
        if len(merged) < 5:
            return merged

        # 构建 chord_id → chord_name 映射
        cid_to_name = {}
        for c in all_detected:
            cid = c.get("chord_id", "")
            name = c.get("chord_name", "")
            if cid and name and cid not in cid_to_name:
                cid_to_name[cid] = name

        WIN = 2.0  # 窗口大小（秒）

        # 找到全局时间范围
        all_times = [c["time"] for c in merged] + [c.get("end_time", c["time"]) for c in merged]
        t_min, t_max = min(all_times), max(all_times)
        if t_max - t_min < WIN:
            return merged

        # 将 merged 段分配到窗口
        n_win = max(1, int((t_max - t_min) / WIN) + 1)
        win_chords = {}  # win_idx -> {chord_id: weighted_score}
        for c in merged:
            t0 = c["time"]
            t1 = c.get("end_time", t0 + 0.3)
            full_dur = max(0.1, t1 - t0)
            conf = c.get("confidence", 0.5)
            # 该段可能跨多个窗口
            w0 = int((t0 - t_min) / WIN)
            w1 = int((t1 - t_min) / WIN)
            for wi in range(w0, w1 + 1):
                if wi < 0 or wi >= n_win:
                    continue
                if wi not in win_chords:
                    win_chords[wi] = {}
                cid = c["chord_id"]
                # 计算该和弦在本窗口内的实际时长（而非完整时长）
                win_start = t_min + wi * WIN
                win_end = win_start + WIN
                overlap = max(0.0, min(t1, win_end) - max(t0, win_start))
                overlap_dur = max(0.1, overlap) if overlap > 0 else 0.0
                if overlap_dur > 0:
                    win_chords[wi][cid] = win_chords[wi].get(cid, 0) + conf * overlap_dur

        # 每个窗口选主导和弦
        win_dominant = {}  # wi -> (chord_id, avg_conf)
        for wi, counts in win_chords.items():
            if not counts:
                continue
            best_cid = max(counts, key=counts.get)
            total_weight = counts[best_cid]
            # 归一化置信度
            win_start = t_min + wi * WIN
            win_end = min(t_max, win_start + WIN)
            norm_conf = min(0.95, total_weight / (win_end - win_start + 0.01))
            win_dominant[wi] = (best_cid, round(norm_conf, 2))

        if not win_dominant:
            return merged

        # 将连续同和弦窗口合并为段
        result = []
        sorted_wins = sorted(win_dominant.keys())
        cur_cid, cur_conf = win_dominant[sorted_wins[0]]
        cur_start = t_min + sorted_wins[0] * WIN
        cur_end = cur_start + WIN

        for wi in sorted_wins[1:]:
            cid, conf = win_dominant[wi]
            ws = t_min + wi * WIN
            if cid == cur_cid:
                cur_end = ws + WIN
                cur_conf = round(max(cur_conf, conf), 2)
            else:
                result.append({
                    "time": round(cur_start, 2),
                    "end_time": round(cur_end, 2),
                    "chord_id": cur_cid,
                    "chord_name": cid_to_name.get(cur_cid, cur_cid),
                    "confidence": cur_conf,
                    "_source": "audio+consolidated",
                })
                cur_cid, cur_conf = cid, conf
                cur_start = ws
                cur_end = ws + WIN

        # 最后一个
        result.append({
            "time": round(cur_start, 2),
            "end_time": round(cur_end, 2),
            "chord_id": cur_cid,
            "chord_name": cid_to_name.get(cur_cid, cur_cid),
            "confidence": cur_conf,
            "_source": "audio+consolidated",
        })

        # 回填 chord_name 和相关元数据（从原始检测中查找）
        meta_cache = {}
        for ac in all_detected:
            cid = ac.get("chord_id", "")
            if cid not in meta_cache:
                meta_cache[cid] = {
                    "chord_name": ac.get("chord_name", cid),
                    "related_problems": ac.get("related_problems", []),
                    "related_techniques": ac.get("related_techniques", []),
                }
        for c in result:
            meta = meta_cache.get(c["chord_id"], {})
            c["chord_name"] = meta.get("chord_name", c["chord_id"])
            c["related_problems"] = meta.get("related_problems", [])
            c["related_techniques"] = meta.get("related_techniques", [])

        logger.info(f"窗口合并: {len(merged)}段 → {len(result)}段 (窗口={WIN}s)")
        return result

    def _fuse_chord_detection(self, detected_chords, video_data, audio_result, capo,
                               vl_chords: list = None):
        """融合音频、视频、VL三模态的和弦检测结果。

        策略（VL优先）：
        - VL 识别到和弦 → 以 VL 的 root+type 为基准（VL 看得准）
        - 音频 chroma + basic-pitch notes → 验证色彩音（7th, sus4, dim 等）
        - VL 不可用 → 回退到原有的 MediaPipe 手型 + 音频融合逻辑
        """
        import re

        def _root_note(chord_id):
            if not chord_id:
                return ""
            m = re.match(r'^([A-G][#b]?)', str(chord_id))
            return m.group(1) if m else ""

        vl_chords = vl_chords or []

        try:
            from chord_analyzer import ChordAnalyzer

            # ── 音频和弦预处理 ──
            threshold = 0.48 if len(detected_chords) >= 8 else 0.50
            merged = self._merge_adjacent_chords(detected_chords, threshold)
            if len(detected_chords) >= 20 and len(merged) > 10:
                merged = self._consolidate_chord_windows(merged, detected_chords)

            # ── 直接使用 chroma 和弦结果 ──
            # VL 证人模式和手型融合已移除：VL 识别准确率不足，证人不准等于引入噪声
            for mc in merged:
                mc["_source"] = "chroma"
            result = sorted(merged, key=lambda c: c.get("time", 0))
            names = [f"{c.get('chord_name','?')}({c.get('confidence',0):.2f})" for c in result[:8]]
            logger.info(f"和弦检测(chroma直出): {len(detected_chords)}个 → {len(result)}个: {names}")
            return result

            # === 以下为旧 VL 证人/手型融合代码，已废弃 ===
            # VL 作为辅助证人，不独立判断和弦，不覆盖 chroma
            # 只提供三个信号：横按确认、根音交叉验证、barre/open 矛盾标记
            #
            # 新增：vl_signals（video_processor 的三道判断题）
            # barre/zone/floating → 辅助验证和弦候选
            vl_signals = video_data.get("vl_signals", {}) if video_data else {}
            if vl_chords:
                high_conf_vl = [vc for vc in vl_chords
                                if vc.get("confidence", 0) >= 0.50]
                if high_conf_vl:
                    BARRE_CHORDS = {
                        "F", "Fm", "chord-F", "chord-Fm",
                        "F#", "F#m", "Gb", "Gbm", "chord-F#", "chord-F#m",
                        "B", "Bm", "chord-B", "chord-Bm",
                        "Bb", "Bbm", "chord-Bb", "chord-Bbm",
                        "B7", "chord-B7",
                    }
                    OPEN_CHORDS = {
                        "C", "Am", "Em", "E", "A", "D", "Dm", "G",
                        "chord-C", "chord-Am", "chord-Em", "chord-E",
                        "chord-A", "chord-D", "chord-Dm", "chord-G",
                    }

                    fused = []
                    used_vl = set()

                    for ac in merged:
                        at = ac.get("time", 0)
                        ac_cid = str(ac.get("chord_id", ""))
                        aconf = ac.get("confidence", 0)

                        best_vl = None
                        best_diff = float("inf")
                        best_vi = -1
                        for i, vc in enumerate(high_conf_vl):
                            if i in used_vl:
                                continue
                            diff = abs(at - vc.get("time", 0))
                            if diff < 2.0 and diff < best_diff:
                                best_diff = diff
                                best_vl = vc
                                best_vi = i

                        if best_vl is None:
                            ac["_source"] = "audio"
                            fused.append(dict(ac))
                            continue

                        used_vl.add(best_vi)
                        vl_cid = str(best_vl.get("chord_id", ""))
                        vl_name = str(best_vl.get("chord_name", ""))
                        vl_conf = best_vl.get("confidence", 0)
                        vl_is_barre = best_vl.get("is_barre", False)
                        vl_root = _root_note(vl_cid)
                        aroot = _root_note(ac_cid)

                        ac_is_barre = ac_cid in BARRE_CHORDS
                        ac_is_open = ac_cid in OPEN_CHORDS

                        # ── VL 证人模式：只确认/质疑，不覆盖 ──
                        if vl_root and aroot and vl_root == aroot:
                            # 根音一致 → VL 确认，小幅加置信度
                            ac["_source"] = "audio+vl_confirm"
                            boost = 0.05
                            if vl_is_barre and ac_is_barre:
                                boost = 0.10  # 横按双确认
                            ac["confidence"] = round(min(0.95, aconf + boost), 2)
                        else:
                            # 根音不一致 → 标记冲突，降权，但不覆盖
                            penalty = 0.80  # chroma 置信度打八折
                            if vl_is_barre and ac_is_open:
                                penalty = 0.70  # 横按视觉 vs 开放和弦 → 更强矛盾
                            elif not vl_is_barre and ac_is_barre:
                                penalty = 0.75  # 非横按视觉 vs 横按和弦

                            # VL 证人判断题增强：barre/zone/floating 信号
                            if vl_signals:
                                # barre 信号强但和弦是开放的 → 加大矛盾
                                if vl_signals.get("barre") and vl_signals["barre_conf"] > 0.6 and ac_is_open:
                                    penalty = min(penalty, 0.65)
                                    logger.debug(f"VL证人barre信号: {ac_cid}看起来是横按但和弦是开放和弦")
                                # barre 信号明确为否但和弦是横按 → 矛盾
                                if not vl_signals.get("barre") and vl_signals["barre_conf"] > 0.7 and ac_is_barre:
                                    penalty = min(penalty, 0.70)
                                    logger.debug(f"VL证人非barre: {ac_cid}看起来不是横按但和弦需要横按")
                                # 高把位 + 开放和弦 → 矛盾
                                if vl_signals.get("zone") == "high" and vl_signals["zone_conf"] > 0.6 and ac_is_open:
                                    penalty = min(penalty, 0.70)
                                    logger.debug(f"VL证人高把位: {ac_cid}在高把位但识别为开放和弦")

                            ac["confidence"] = round(aconf * penalty, 2)
                            ac["_vl_conflict"] = True
                            ac["_vl_says"] = vl_cid
                            ac["_source"] = "audio_vl_conflict"

                            # 仅当 chroma 置信度极低且 VL 很确信时，才考虑用 VL
                            if aconf < 0.40 and vl_conf >= 0.70:
                                ac["chord_id"] = vl_cid
                                ac["chord_name"] = vl_name
                                ac["confidence"] = round(vl_conf * 0.80, 2)
                                ac["_source"] = "vl_override_low_audio"
                                logger.info(f"VL低置信覆盖: audio={ac_cid}({aconf:.2f}) → VL={vl_cid}({vl_conf:.2f}) @{at:.1f}s")

                        fused.append(dict(ac))

                    result = sorted(fused, key=lambda c: c.get("time", 0))
                    names = [f"{c.get('chord_name','?')}({c.get('confidence',0):.2f}|{c.get('_source','?')})"
                             for c in result[:8]]
                    logger.info(f"和弦检测(VL证人): {len(vl_chords)}VL + {len(merged)}音频 → {len(result)}个: {names}")
                    return result

            # ── VL 不可用 → 回退到手型+音频融合 ──
            all_hand_landmarks = video_data.get("hand_landmarks", [])
            mirror = getattr(self.video_processor, '_mirror_detected', False)
            hand_zone = video_data.get("hand_zone_center", None)

            hand_chords = ChordAnalyzer.detect_chords_from_landmarks(
                all_hand_landmarks, mirror_detected=mirror, hand_zone=hand_zone
            )

            if not hand_chords:
                for mc in merged:
                    mc["_source"] = "audio"
                result = sorted(merged, key=lambda c: c.get("time", 0))
                names = [f"{c.get('chord_name','?')}({c.get('confidence',0):.2f})" for c in result[:8]]
                logger.info(f"和弦检测(纯音频): {len(detected_chords)}个 → {len(result)}个: {names}")
                return result

            if not merged:
                high_conf = [hc for hc in hand_chords if hc.get("confidence", 0) >= 0.75]
                for hc in high_conf:
                    hc["_source"] = "video"
                logger.info(f"和弦检测(纯视频): {len(high_conf)}个")
                return high_conf

            fused = []
            used_hand = set()

            for ac in merged:
                at = ac.get("time", 0)
                acid = ac.get("chord_id", "")
                aroot = _root_note(acid)
                aconf = ac.get("confidence", 0)

                best_hand = None
                best_hidx = -1
                best_diff = float("inf")
                for i, hc in enumerate(hand_chords):
                    if i in used_hand:
                        continue
                    hconf = hc.get("confidence", 0)
                    if hconf < 0.65:
                        continue
                    diff = abs(at - hc.get("time", 0))
                    if diff < 1.5 and diff < best_diff:
                        best_diff = diff
                        best_hand = hc
                        best_hidx = i

                if best_hand is None:
                    ac["_source"] = "audio"
                    fused.append(dict(ac))
                    continue

                used_hand.add(best_hidx)
                hroot = _root_note(best_hand.get("chord_id", ""))
                hconf = best_hand.get("confidence", 0)

                if hroot and aroot and hroot == aroot:
                    ac["_source"] = "audio+hand"
                    ac["confidence"] = round(min(0.95, aconf + 0.10), 2)
                elif hconf >= 0.85 and aconf < 0.55:
                    best_hand["_source"] = "hand_override"
                    best_hand["confidence"] = round(hconf * 0.85, 2)
                    fused.append(dict(best_hand))
                    continue
                else:
                    ac["_source"] = "audio_primary"
                    if hconf >= 0.65 and aconf >= 0.50 and hroot != aroot:
                        ac["_conflict"] = True
                        ac["_conflict_hand_chord"] = best_hand.get("chord_id", "")
                        ac["_conflict_hand_name"] = best_hand.get("chord_name", "")
                        ac["_conflict_hand_root"] = hroot
                        ac["_conflict_hand_conf"] = round(hconf, 2)
                        ac["_conflict_audio_chord"] = ac.get("chord_id", "")
                        ac["_conflict_audio_name"] = ac.get("chord_name", "")
                        ac["_conflict_audio_root"] = aroot
                        ac["_conflict_audio_conf"] = round(aconf, 2)
                        logger.info(f"和弦冲突: 手型={best_hand.get('chord_name','?')}({hconf:.2f}) vs 音频={ac.get('chord_name','?')}({aconf:.2f}) @{at:.1f}s")
                fused.append(dict(ac))

            for i, hc in enumerate(hand_chords):
                if i not in used_hand and hc.get("confidence", 0) >= 0.80:
                    hc["_source"] = "hand_supplementary"
                    hc["confidence"] = round(hc.get("confidence", 0) * 0.80, 2)
                    fused.append(dict(hc))

            result = sorted(fused, key=lambda c: c.get("time", 0))
            names = [f"{c.get('chord_name','?')}({c.get('confidence',0):.2f}|{c.get('_source','?')})" for c in result[:8]]
            logger.info(f"和弦检测(音频为主): {len(merged)}音频 + {len(hand_chords)}手型 → {len(result)}个: {names}")
            return result

        except Exception as e:
            logger.warning(f"和弦融合失败: {e}")
            return detected_chords

    def _cross_validate_chords_visual(self, detected_chords, video_data):
        """横按检测交叉验证：视觉食指伸直度 → barre/open约束音频和弦。

        食指 PIP 夹角 < 22° → 横按 (84%准确率)
        横按→候选限定{F,Fmaj7,B,B7,Bm,Bm7}, 开放→排除横按
        矛盾时降低音频置信度-0.10
        """
        if not detected_chords or not video_data:
            return detected_chords

        hand_landmarks = video_data.get("hand_landmarks", [])
        if len(hand_landmarks) < 5:
            return detected_chords

        try:
            from chord_classifier import LandmarkChordClassifier
            from db.reference_db import ReferenceDB
        except ImportError:
            return detected_chords

        if not hasattr(self, '_visual_chord_clf'):
            rdb = ReferenceDB()
            self._visual_chord_clf = LandmarkChordClassifier(rdb)

        clf = self._visual_chord_clf
        if not clf.is_ready:
            return detected_chords

        vp = getattr(self, 'video_processor', None)
        if vp is None:
            return detected_chords

        left_hand_frames = [lm for lm in hand_landmarks if lm.get("hand") == "Left"]
        if len(left_hand_frames) < 3:
            return detected_chords

        confirmed = 0
        conflicts = 0
        for i, chord in enumerate(detected_chords):
            t_chord = chord.get("time", 0)
            audio_chord_id = chord.get("chord_id", "")

            best_frame = None
            best_dt = 999.0
            for lf in left_hand_frames:
                dt = abs(lf.get("time", 0) - t_chord)
                if dt < 0.3 and dt < best_dt:
                    best_dt = dt
                    best_frame = lf

            if best_frame is None:
                continue

            lm_list = best_frame.get("landmarks", [])
            if len(lm_list) < 21:
                continue

            vec = vp.landmark_list_to_vector(lm_list)
            if vec is None:
                continue

            result = clf.constraint_chord_candidates(vec, audio_chord_id)
            barre_info = result.get("barre", {})

            chord["_visual_cv"] = {
                "is_barre": barre_info.get("is_barre", False),
                "barre_score": barre_info.get("barre_score", 0),
                "index_angle": barre_info.get("index_angle", 0),
                "constraint": result["constraint"],
                "consistent": result["consistent"],
            }

            if result["consistent"] and result["constraint"] != "uncertain":
                chord["confidence"] = min(1.0, chord.get("confidence", 0.5) + 0.05)
                confirmed += 1
            elif not result["consistent"]:
                chord["confidence"] = max(0.3, chord.get("confidence", 0.5) - 0.10)
                conflicts += 1
                logger.info(f"横按冲突: {result['detail']} (t={t_chord:.1f}s)")

        if confirmed > 0 or conflicts > 0:
            logger.info(f"横按验证: {confirmed}确认, {conflicts}冲突 / {len(detected_chords)}和弦")
        return detected_chords

    # ====== Qwen VL 指板视觉和弦检测 (Step 5.2) ======

    @staticmethod
    @staticmethod
    def _build_vl_verification_prompt(candidates: list) -> str:
        """Build a prompt asking VL to choose among chroma's top candidates.

        VL is constrained to verify, not invent — it can only pick from the list.
        """
        names = [c.get("chord_name", c.get("chord_id", "?")) for c in candidates]
        name_str = "、".join(names)
        return (
            "你是一位吉他教师。看这张吉他指板近景截图。\n\n"
            f"音频分析认为当前可能是在弹以下和弦之一：{name_str}\n\n"
            "请根据手指在指板上的按弦位置，判断手型最符合以上哪一个和弦。\n"
            "注意：只观察手指的按弦位置（哪根手指按在几弦几品），不要受其他因素影响。\n\n"
            "返回严格 JSON（不要 markdown 代码块）：\n"
            '{"chord": "Am", "confidence": 0.8}  // 从候选列表中选一个\n'
            '{"chord": "uncertain", "confidence": 0.0}  // 如果无法从图中判断\n\n'
            "只返回 JSON，不要任何其他文字。"
        )

    @staticmethod
    def _build_chord_detection_prompt() -> str:
        """Old-style prompt: identify finger positions on strings/frets + barre detection."""
        return (
            "你是一位吉他教师。看这张吉他指板近景截图。\n\n"
            "请完成两项任务：\n\n"
            "1. 识别左手手指的按弦位置\n"
            "   吉他弦编号：最粗的是6弦（低音E），最细的是1弦（高音E）。\n"
            "   品位从琴头开始数：最靠近琴头的是1品，依次类推。\n"
            "   手指名称：食指、中指、无名指、小指。\n"
            "   只识别正在按压琴弦的手指（悬空或未触弦不返回）。\n\n"
            "2. 判断食指是否在横按（barre）\n"
            '   "barre": true 表示食指同时按下多根弦在同一品上\n'
            '   "barre": false 表示食指只按一根弦或未按弦\n\n'
            "返回严格 JSON（不要 markdown 代码块）：\n"
            '{"fingers": {"食指": {"string": 2, "fret": 1}, "中指": {"string": 3, "fret": 2}}, "barre": false}\n'
            "如果某手指按在两根弦之间（横按），string 填被按的第一根弦。\n"
            '如果没有检测到任何手指在按弦，返回 {"hands_detected": false}\n\n'
            "只返回 JSON，不要任何其他文字。"
        )

    @staticmethod
    def _find_chord_change_moments(detected_chords: list, max_moments: int = 8) -> list:
        """从检测到的和弦列表中识别切换时刻（去重相邻同和弦）。

        Returns:
            [(time, chord_id, chord_name), ...] 每个切换时刻一条，最多 max_moments 条
        """
        if not detected_chords:
            return []

        sorted_chords = sorted(detected_chords, key=lambda c: c.get("time", 0))
        moments = []
        last_cid = None

        for c in sorted_chords:
            cid = c.get("chord_id", "")
            if cid and cid != last_cid:
                moments.append((
                    c.get("time", 0),
                    cid,
                    c.get("chord_name", cid),
                ))
                last_cid = cid

        # Limit to max_moments, preferring evenly spaced samples
        if len(moments) > max_moments:
            step = len(moments) / max_moments
            sampled = [moments[int(i * step)] for i in range(max_moments)]
            return sampled[:max_moments]

        return moments[:max_moments]

    async def _detect_vl_chords(self, video_data: dict, detected_chords: list,
                                 instrument: str = "guitar") -> list:
        """Step 5.2: 用 Qwen VL 识别指板手指位置→匹配合弦，同时检测横按。

        Returns:
            [{time, chord_id, chord_name, confidence, is_barre, _source: "vl"}, ...]
        """
        if not self.vision_analyzer.available:
            logger.info("VL跳过: Vision AI 不可用")
            return []

        if self.vision_analyzer.provider != "qwen":
            logger.info("VL跳过: 仅 Qwen VL 支持")
            return []

        snapshots = video_data.get("snapshots", [])
        if not snapshots:
            logger.info("VL跳过: 无截图")
            return []

        moments = self._find_chord_change_moments(detected_chords, max_moments=8)
        if not moments:
            logger.info("VL跳过: 无和弦切换时刻")
            return []

        UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "snapshots")
        snap_times = []
        for s in snapshots:
            fname = os.path.basename(s.get("url", ""))
            fpath = os.path.join(UPLOAD_DIR, fname)
            if os.path.exists(fpath):
                snap_times.append((s.get("timestamp", 0), fpath))
        if not snap_times:
            return []
        snap_times.sort(key=lambda x: x[0])

        prompt = self._build_chord_detection_prompt()
        vl_results = []
        seen_chord_ids = set()

        for moment_time, moment_cid, moment_name in moments:
            if moment_cid in seen_chord_ids:
                continue
            seen_chord_ids.add(moment_cid)

            best_path = None
            best_diff = float("inf")
            for st, sp in snap_times:
                diff = abs(st - moment_time)
                if diff < best_diff:
                    best_diff = diff
                    best_path = sp
            if best_path is None or best_diff > 5.0:
                continue

            try:
                image_data, mime = self.vision_analyzer._compress_image_b64(best_path)
                text = self.vision_analyzer._call_qwen(prompt, image_data, mime, max_tokens=512)
                parsed = self._parse_vl_finger_response(text)
                if not parsed or not parsed.get("fingers"):
                    logger.info(f"VL: @{moment_time:.1f}s 未检测到手型")
                    continue

                from chord_analyzer import ChordAnalyzer
                match = ChordAnalyzer.finger_positions_to_chord_id(parsed.get("fingers", {}))
                if match:
                    match["time"] = round(moment_time, 1)
                    match["_source"] = "vl"
                    match["_vl_snapshot"] = os.path.basename(best_path)
                    match["is_barre"] = parsed.get("barre", False)
                    vl_results.append(match)
                    logger.info(f"VL: @{moment_time:.1f}s → {match['chord_name']} "
                                f"(conf={match['confidence']:.2f}, barre={match.get('is_barre')})")
                else:
                    logger.info(f"VL: @{moment_time:.1f}s 手指位置 {parsed.get('fingers')} 无法映射")

            except Exception as e:
                logger.warning(f"VL失败 @{moment_time:.1f}s: {e}")
                continue

        return vl_results

    @staticmethod
    def _build_chroma_candidates(detected_chords: list, moment_time: float) -> list:
        """Build top-3 chroma candidates near a given time point."""
        window = 3.0
        nearby = [c for c in detected_chords
                  if abs(c.get("time", 0) - moment_time) < window]

        # Deduplicate by chord_id, keep highest confidence
        best = {}
        for c in nearby:
            cid = c.get("chord_id", "")
            if not cid:
                continue
            if cid not in best or c.get("confidence", 0) > best[cid].get("confidence", 0):
                best[cid] = c

        # Sort by confidence descending, take top 3
        return sorted(best.values(), key=lambda c: c.get("confidence", 0), reverse=True)[:3]

    @staticmethod
    def _parse_vl_verification_response(text: str, candidates: list) -> tuple:
        """Parse VL verification response: {"chord": "Am", "confidence": 0.8}

        Returns:
            (chord_id or None, confidence)
        """
        if not text:
            return None, 0

        import re
        text = re.sub(r'^```(?:json)?\s*\n?', '', text.strip())
        text = re.sub(r'\n?```\s*$', '', text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return None, 0
            else:
                return None, 0

        vl_chord_name = str(data.get("chord", "")).strip()
        vl_conf = float(data.get("confidence", 0))

        if not vl_chord_name or vl_chord_name.lower() == "uncertain":
            return None, 0

        # Normalize: match VL's response to one of the candidate chord_ids
        candidate_map = {}
        for c in candidates:
            cid = str(c.get("chord_id", ""))
            cname = str(c.get("chord_name", ""))
            # Normalize for matching
            cid_normalized = cid.replace("chord-", "").lower()
            cname_normalized = cname.replace("和弦", "").lower()
            candidate_map[cid_normalized] = cid
            candidate_map[cname_normalized] = cid
            # Also try without "major" suffix
            if cid_normalized.endswith("major"):
                candidate_map[cid_normalized.replace("major", "")] = cid
            if "maj" in cid_normalized:
                # e.g., Cmaj7 -> try C and C-major
                pass

        vl_normalized = vl_chord_name.lower().replace("和弦", "")
        # Direct match
        if vl_normalized in candidate_map:
            return candidate_map[vl_normalized], vl_conf

        # Try fuzzy: see if any candidate name is a substring or vice versa
        for cid_norm, cid_orig in candidate_map.items():
            if vl_normalized in cid_norm or cid_norm in vl_normalized:
                return cid_orig, vl_conf

        logger.info(f"VL chose '{vl_chord_name}' which is not in candidates "
                    f"{[c.get('chord_name') for c in candidates]}")
        return vl_chord_name, vl_conf

    @staticmethod
    @staticmethod
    def _parse_vl_finger_response(text: str) -> dict:
        """Parse Qwen VL response with fingers + barre fields.

        New format:
            {"fingers": {"食指": {"string": 2, "fret": 1}}, "barre": false}
        Old format (fallback):
            {"食指": {"string": 2, "fret": 1}, ...}

        Returns:
            {"fingers": {...}, "barre": bool} or {}
        """
        if not text:
            return {}

        import re
        text = re.sub(r'^```(?:json)?\s*\n?', '', text.strip())
        text = re.sub(r'\n?```\s*$', '', text)
        text = re.sub(r',\s*(\]|\})', r'\1', text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    return {}
            else:
                return {}

        if not isinstance(data, dict):
            return {}

        if data.get("hands_detected") is False:
            return {}

        def _validate_finger(fdata):
            if isinstance(fdata, dict) and "string" in fdata:
                try:
                    s = int(fdata.get("string", 0))
                    f = int(fdata.get("fret", 0))
                    if 1 <= s <= 6 and 0 <= f <= 20:
                        return {"string": s, "fret": f}
                except (ValueError, TypeError):
                    pass
            return None

        # New format: {"fingers": {...}, "barre": bool}
        if "fingers" in data:
            fingers_data = data.get("fingers", {})
            valid_fingers = {}
            if isinstance(fingers_data, dict):
                for fname, fdata in fingers_data.items():
                    vf = _validate_finger(fdata)
                    if vf:
                        valid_fingers[fname] = vf
            barre = bool(data.get("barre", False))
            return {"fingers": valid_fingers, "barre": barre} if valid_fingers else {}

        # Old format fallback: flat {"食指": {...}, "中指": {...}}
        valid_fingers = {}
        for fname, fdata in data.items():
            if fname in ("hands_detected", "barre"):
                continue
            vf = _validate_finger(fdata)
            if vf:
                valid_fingers[fname] = vf
        return {"fingers": valid_fingers, "barre": False} if valid_fingers else {}

    def _refine_clusters_with_hand_motion(self, onset_clusters, hand_landmarks, right_hand=None):
        """用手腕运动幅度 + 逐次扫弦事件修正扫弦/柱式误判。

        扫弦时右手腕上下摆动幅度大（Y轴标准差 > 阈值），柱式/分解时幅度小。
        仅将 strumming → block_chord 降级，不反向升级。

        Args:
            right_hand: video_processor._compute_right_hand_strumming() 的输出，
                        包含 strum_events（逐次扫弦方向+速度）
        """
        right_hand = right_hand or {}
        if not onset_clusters or not hand_landmarks:
            return onset_clusters

        # 找到右手（拨弦手）的 landmarks
        right_hand_frames = [f for f in hand_landmarks if f.get("hand") == "Right"]
        if len(right_hand_frames) < 10:
            return onset_clusters

        # 提取右手腕 (landmark 0) 的 Y 坐标时间序列
        wrist_y = []
        for f in right_hand_frames:
            lm = f.get("landmarks", [])
            if len(lm) > 0:
                wrist_y.append((f["time"], lm[0][1]))  # (time, y)

        if len(wrist_y) < 10:
            return onset_clusters

        # 计算全局手腕 Y 标准差（归一化坐标 0-1）
        all_y = [y for _, y in wrist_y]
        global_std = np.std(all_y) if len(all_y) > 1 else 0.0

        # 逐次扫弦事件（AirStrum 风格交叉验证）
        strum_events = right_hand.get("strum_events", [])
        strum_motion_score = right_hand.get("strumming_motion_score", 0)

        refined = []
        for cluster in onset_clusters:
            ctype = cluster.get("type", "")
            if ctype != "strumming":
                refined.append(cluster)
                continue

            t0 = cluster.get("start_time", 0)
            t1 = cluster.get("end_time", 0)
            if t1 <= t0:
                refined.append(cluster)
                continue

            # 该 cluster 时间范围内的手腕 Y 值
            cluster_y = [y for t, y in wrist_y if t0 <= t <= t1]
            if len(cluster_y) < 5:
                refined.append(cluster)
                continue

            cluster_std = np.std(cluster_y)

            # 判断：局部波动大 或 全局波动大 → 真实扫弦
            is_significant_motion = (cluster_std > 0.015 or global_std > 0.02)

            # 右手分析增强：strumming_motion_score > 0.5 说明手腕有明显扫弦动作
            if strum_motion_score > 0.5:
                is_significant_motion = True

            # AirStrum 交叉验证：统计时间窗口内视觉扫弦事件数
            visual_strums_in_window = 0
            if strum_events:
                for se in strum_events:
                    if t0 - 0.05 <= se["time"] <= t1 + 0.05:
                        visual_strums_in_window += 1

                # 视觉确认有扫弦动作 → 增强置信度
                if visual_strums_in_window >= 1 and not is_significant_motion:
                    is_significant_motion = True
                    cluster = dict(cluster)
                    cluster["_visual_strum_cross_validated"] = True
                elif visual_strums_in_window == 0 and strum_motion_score < 0.3:
                    # 视觉没有任何扫弦且全局评分低 → 音频很可能是假阳性
                    is_significant_motion = False

            if not is_significant_motion:
                cluster = dict(cluster)
                cluster["type"] = "block_chord"
                cluster["type_confidence"] = round(cluster.get("type_confidence", 0.5) * 0.7, 2)
                cluster["_hand_motion_override"] = True
                reason = f"wrist_std={cluster_std:.4f}, global_std={global_std:.4f}"
                if strum_events:
                    reason += f", visual_strums={visual_strums_in_window}"
                logger.info(f"手部运动修正: strumming→block_chord ({reason})")
            elif visual_strums_in_window >= 1:
                logger.info(f"视听交叉验证: strumming 确认 "
                            f"(visual_strums={visual_strums_in_window}, "
                            f"t={t0:.2f}-{t1:.2f})")

            refined.append(cluster)

        return refined

    def _merge_adjacent_chords(self, chords, threshold):
        """合并相邻同和弦段 + 过滤低置信度。"""
        merged = []
        for c in chords:
            conf = c.get("confidence", 0)
            if conf < threshold:
                continue
            cid = c.get("chord_id", "")
            if merged and merged[-1]["chord_id"] == cid \
                    and c.get("time", 0) - merged[-1].get("end_time", 0) < 3.0:
                merged[-1]["end_time"] = c.get("end_time", merged[-1]["end_time"])
                merged[-1]["confidence"] = round(max(merged[-1]["confidence"], conf), 2)
            else:
                merged.append(dict(c))
        return merged

    def _detect_technique_hints_from_audio(self, audio_result):
        """从音频特征推断可能的技巧类型"""
        hints = []
        segments = audio_result.get("technique_segments", [])
        if segments:
            for seg in segments:
                if seg.get("confidence", 0) > 0.5 and seg["technique_id"] not in hints:
                    hints.append(seg["technique_id"])
            if hints:
                logger.info(f"technique detect (model output): {hints}")
                return hints

        af = audio_result.get("audio_features", {})
        if not af:
            return hints

        attack = af.get("attack_time_avg_ms", 30)

        if attack < 15 and af.get("inharmonicity_ratio", 0.3) > 0.4:
            hints.append("pm-technique")

        notes = audio_result.get("notes", [])
        if len(notes) >= 2:
            legato_count = sum(
                1 for i in range(1, min(len(notes), 20))
                if 0 < notes[i].get("start_time", 0) - notes[i-1].get("end_time", 0) < 0.03
            )
            if legato_count >= 3:
                hints.append("hammer-on")
        return hints

    @staticmethod
    def _load_chord_templates():
        """加载所有和弦知识模板，返回 {chord_name_lower: {chord_name, strings, expected_midi}}"""
        import ast
        knowledge_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "knowledge", "chords"
        )
        templates = {}
        if not os.path.isdir(knowledge_dir):
            return templates
        GUITAR_STRINGS = [(1, 64), (2, 59), (3, 55), (4, 50), (5, 45), (6, 40)]
        for fname in sorted(os.listdir(knowledge_dir)):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(knowledge_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                if not content.startswith("---"):
                    continue
                parts = content.split("---", 2)
                if len(parts) < 3:
                    continue
                fm = {}
                for line in parts[1].strip().split("\n"):
                    line = line.strip()
                    if ":" in line:
                        k, v = line.split(":", 1)
                        fm[k.strip()] = v.strip()
                strings_raw = fm.get("strings", "")
                if not strings_raw:
                    continue
                try:
                    # x 不是合法 Python 字面量，替换为 -1 后再解析
                    cleaned = strings_raw.replace("x", "-1").replace("X", "-1")
                    strings = ast.literal_eval(cleaned)
                except Exception:
                    continue
                if not isinstance(strings, list) or len(strings) != 6:
                    continue
                # strings 顺序: [6弦..1弦], 计算每弦期望 MIDI 音高
                expected_midi = {}
                for i, s_val in enumerate(strings):
                    if isinstance(s_val, (int, float)) and s_val < 0:
                        continue  # x → -1, 不弹
                    if isinstance(s_val, (int, float)) and s_val >= 0:
                        string_num = 6 - i
                        open_midi = GUITAR_STRINGS[string_num - 1][1]
                        fret = int(s_val)
                        midi = open_midi + fret
                        expected_midi[midi] = (string_num, fret)
                if expected_midi:
                    chord_name = fm.get("chord_name", fname[:-3])
                    chord_id = fm.get("id", "")
                    key = chord_name.replace("和弦", "").strip().lower()
                    templates[key] = {
                        "chord_name": chord_name, "chord_id": chord_id,
                        "strings": strings, "expected_midi": expected_midi,
                    }
            except Exception:
                continue
        return templates

    @staticmethod
    def _correct_frets_by_chords(notes, detected_chords, capo=0):
        """用检测到的和弦约束修正不合理品位分配。"""
        if not notes or not detected_chords:
            return notes
        templates = AnalysisService._load_chord_templates()
        if not templates:
            return notes

        corrected = 0
        cn = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六"}
        for chord in detected_chords:
            chord_name = chord.get("chord_name", "").replace("和弦", "").strip().lower()
            template = templates.get(chord_name)
            if not template:
                for key, t in templates.items():
                    if chord_name in key or key in chord_name:
                        template = t
                        break
            if not template:
                continue

            expected = template["expected_midi"]
            chord_time = chord.get("time", 0)
            WINDOW = 0.3

            for note in notes:
                nt = note.get("start_time", 0)
                if abs(nt - chord_time) > WINDOW:
                    continue
                pitch = note.get("pitch", 0)
                if pitch in expected:
                    exp_str, exp_fret = expected[pitch]
                    cur_str = note.get("string", 0)
                    if cur_str != exp_str and abs(cur_str - exp_str) <= 2:
                        note["string"] = exp_str
                        note["fret"] = exp_fret
                        actual_fret = exp_fret - capo if capo > 0 and exp_fret >= capo else exp_fret
                        if capo > 0 and exp_fret >= capo:
                            note["guitar_position"] = f"{cn[exp_str]}弦{actual_fret}品（变调夹第{capo}品）"
                        else:
                            note["guitar_position"] = f"{cn[exp_str]}弦{actual_fret}品"
                        corrected += 1

        if corrected > 0:
            logger.info(f"和弦品位修正: 修正了 {corrected} 个音符")
        return notes

    async def _run_hand_detection(self, video_path, audio_result, instrument,
                                    technique_hints, detected_chords=None,
                                    proxy_dest=None):
        loop = asyncio.get_running_loop()
        technique_segments = audio_result.get("technique_segments", [])
        from functools import partial
        process_fn = partial(
            self.video_processor.process,
            video_path,
            audio_result.get("error_timestamps", []),
            instrument,
            technique_hints if technique_hints else None,
            technique_segments if technique_segments else None,
            force_hand="left",
            detected_chords=detected_chords if detected_chords else None,
            proxy_dest=proxy_dest,
        )
        return await loop.run_in_executor(None, process_fn)

    def _get_first_note_time(self, audio_result):
        notes = audio_result.get("notes", [])
        if notes:
            first = min(n.get("start_time", 999) for n in notes)
            if first < 999:
                return max(0.0, first)
        return 0.0

    @staticmethod
    def _short_label(detail_text, max_len=24):
        """从原始检测描述生成智能摘要，多根手指同问题时归纳为整体表述。

        策略（参考主流运动分析App的截图标注方式）:
        - 多根手指同一问题 → 「整体手指偏直，放松指根关节」
        - 不同问题并存 → 「无名指、中指太直，食指塌陷」
        - 单个问题 → 「无名指太直，放松指根关节」
        目标长度 14-24 字，信息密度与参考图描述一致。
        """
        if not detail_text:
            return ""
        # 清理前缀符号和多余空白
        cleaned = detail_text.strip()
        for ch in ["⚠️", "💡", "✅", "❌", "🔴", "🟡", "⚠", "→"]:
            cleaned = cleaned.replace(ch, "")
        cleaned = cleaned.strip()

        # 按 "；" 或 "。" 拆分所有子问题
        segments = []
        for part in cleaned.replace("；", "\n").replace("。", "\n").split("\n"):
            part = part.strip()
            if part:
                segments.append(part)
        if not segments:
            return cleaned[:max_len] if cleaned else "手型问题"

        # 快速解析每个子问题 → {finger, problem}
        parsed = []
        FINGERS = ["无名指", "食指", "中指", "小指", "拇指", "手腕"]
        PROBLEM_MAP = {
            "太直": "太直", "太竖直": "太直", "太垂直": "太直", "站得太直": "太直",
            "太弯": "太弯", "弯得有点多": "太弯", "弯太多": "太弯",
            "紧张": "紧张", "太紧张": "紧张", "捏得太紧": "捏太紧", "捏太紧": "捏太紧",
            "翘起": "翘起", "没立起来": "没立起", "塌": "塌陷",
            "趴": "趴指", "缩": "没打开",
            "力度过大": "太用力", "太用力": "太用力",
        }

        # 辅助：检查 main 是否包含某个关键词（忽略末尾"了"）
        def _has_kw(main_text, kw):
            if kw in main_text:
                return True
            if (kw + "了") in main_text:
                return True
            return False

        for seg in segments:
            # 提取问题主体（在"——"、"——"、"："、"："之前）
            main = seg.split("——")[0].split("——")[0].split("：")[0].split(":")[0].strip()

            finger = ""
            for f in FINGERS:
                if f in main:
                    finger = f
                    break

            prob_short = ""
            for kw, label in PROBLEM_MAP.items():
                if _has_kw(main, kw):
                    prob_short = label
                    break
            if not prob_short:
                # 尝试匹配 "太X" 或 "X得" 模式 — 使用 \S 匹配中文字符
                import re
                m = re.search(r'(太\S{1,3})(?:了)?', main)
                if m:
                    prob_short = m.group(1)
                else:
                    prob_short = main[:6]

            if finger:
                parsed.append(f"{finger}{prob_short}")
            else:
                parsed.append(prob_short)

        # ── 生成最终标签 ──
        if len(parsed) == 1:
            return parsed[0][:max_len]

        # 去重：移除完全相同的条目
        seen = set()
        unique = []
        for p in parsed:
            if p not in seen:
                seen.add(p)
                unique.append(p)

        if len(unique) >= 3:
            return "整体手型需调整"

        result = "，".join(unique)
        return result[:max_len]

    def _filter_irrelevant_fingers(self, deduped, detected_chords):
        """过滤不相关手指的手型问题：如果 issue 描述的手指不在当前和弦的 required_fingers 中，则过滤。

        例如 Em 和弦只用食指+中指，针对无名指/小指的「太直」「太趴」等检测被过滤；
        拇指、手腕、虎口等问题永远保留。

        使用 ChordAnalyzer._load_chord_templates_with_finger_data() 获取和弦→手指映射。
        """
        if len(deduped) <= 1 or not detected_chords:
            return deduped

        try:
            from chord_analyzer import ChordAnalyzer
            templates = ChordAnalyzer._load_chord_templates_with_finger_data()
        except Exception:
            return deduped

        if not templates:
            return deduped

        # 手指名关键词（中文 → 映射到 template fingers 的 key）
        FINGER_KW = ["食指", "中指", "无名指", "小指", "拇指"]
        # 总是保留的关键词（和弦无关的全局问题）
        ALWAYS_KEEP = ["拇指", "手腕", "虎口", "捏", "手掌", "右手", "坐姿", "手肘",
                       "太紧张", "打品", "杂音", "力度",
                       "没立起来", "翘起", "太弯", "太弯了", "紧张", "僵硬"]
        # 和弦相关的关键词（需要结合当前和弦判断是否合理）
        CHORD_DEPENDENT = ["太直", "太竖直", "太趴"]

        sorted_chords = sorted(detected_chords, key=lambda c: c.get("time", 0))

        def _find_chord_at(t):
            # 严格区间包含优先：时间戳落在和弦 [start, end] 内的优先于任何邻近和弦
            # 避免短和弦（如 fmaj7 dur=1s）被相邻长和弦（如 Am dur=2.7s）淹没
            containing = [c for c in sorted_chords
                          if c.get("time", 0) <= t <= c.get("end_time", c.get("time", 0) + 2.0)]
            if containing:
                containing.sort(key=lambda c: c.get("confidence", 0), reverse=True)
                return containing[0]
            # 次选：窗口匹配（±0.5s），跳过极短且低置信度的段（chroma 噪声）
            best_match = None
            best_dur = 0
            for c in sorted_chords:
                c_start = c.get("time", 0)
                c_end = c.get("end_time", c_start + 2.0)
                if c_start - 0.5 <= t <= c_end + 0.5:
                    dur = c_end - c_start
                    conf = c.get("confidence", 0.5)
                    if dur < 0.8 and conf < 0.5:
                        if best_match is None:
                            best_match = c
                        continue
                    if dur > best_dur:
                        best_match = c
                        best_dur = dur
            if best_match:
                return best_match
            # 兜底：最近和弦（容忍检测漏段，最多 5s）
            if sorted_chords:
                nearest = min(sorted_chords, key=lambda c: abs(c.get("time", 0) - t))
                if abs(nearest.get("time", 0) - t) <= 5.0:
                    return nearest
            return None

        result = []
        filtered_count = 0
        for h in deduped:
            issue_text = h.get("问题", "")
            t = h.get("时间", h.get("timestamp", 0))

            # 找到对应和弦模板（提前查找，ALWAYS_KEEP 和 CHORD_DEPENDENT 都需要）
            ch = _find_chord_at(t)
            ch_id = ch.get("chord_id", "") if ch else ""
            kb_id = f"chord-{ch_id}" if ch_id and not ch_id.startswith("chord-") else ch_id
            template = templates.get(kb_id) if kb_id else None
            if not template and ch:
                ch_name = ch.get("chord_name", "").replace("和弦", "").strip()
                for tid, tpl in templates.items():
                    tpl_name = tpl.get("chord_name", "").replace("和弦", "").strip()
                    if ch_name == tpl_name or ch_name in tpl_name or tpl_name in ch_name:
                        template = tpl
                        break
            used_fingers = set(template.get("fingers", {}).keys()) if template else set()

            # 全局问题 + 严重问题永远保留
            if any(kw in issue_text for kw in ALWAYS_KEEP) or "⚠️" in issue_text:
                result.append(h)
                continue

            # 和弦相关关键词：检查 KB 中是否有"正常表现"说明允许这种行为
            if any(kw in issue_text for kw in CHORD_DEPENDENT) and template:
                normal_section = template.get("_normal_section", "")
                error_section = template.get("_error_section", "")
                if normal_section:
                    issue_fingers_here = {f for f in FINGER_KW if f in issue_text}
                    problem_kw = [kw for kw in CHORD_DEPENDENT if kw in issue_text]
                    # 精确匹配: 只有当 KB 正常表现中明确说「某手指 + 某行为是正常的」才过滤
                    # 例如 KB 写「无名指自然弯曲悬浮是正常的」→ 不过滤「无名指太直」（弯曲≠直）
                    # 如果 KB 写「食指可能看起来较直是正常的」→ 过滤「食指太直」
                    is_normal = False
                    normal_sentences = [s.strip() for s in normal_section.replace("；", "。").split("。")]
                    error_sentences = [s.strip() for s in error_section.replace("；", "。").split("。")] if error_section else []
                    for finger in issue_fingers_here:
                        for kw in problem_kw:
                            # 手指+关键词在同一句中 = KB 确认这是正常表现
                            for sent in normal_sentences:
                                if finger in sent and kw in sent:
                                    # 确认不在 error_section 中（双重保险）
                                    in_error = any(finger in es and kw in es for es in error_sentences)
                                    if not in_error:
                                        is_normal = True
                                        break
                            if is_normal:
                                break
                        if is_normal:
                            break
                    if is_normal:
                        logger.info(
                            f"finger_filter: 过滤和弦相关issue（KB正常表现）"
                            f" at t={t:.1f}s, chord={ch_id}: {issue_text[:60]}"
                        )
                        filtered_count += 1
                        continue

            # 提问题中提到的手指
            issue_fingers = {f for f in FINGER_KW if f in issue_text}
            if not issue_fingers:
                result.append(h)  # 没提到具体手指，保留
                continue

            if not ch:
                result.append(h)  # 找不到对应和弦，保留
                continue

            if not template:
                result.append(h)  # 找不到模板，保留
                continue

            if not used_fingers:
                result.append(h)
                continue

            # 检查是否有不相关手指
            irrelevant = issue_fingers - used_fingers
            if irrelevant:
                logger.info(
                    f"finger_filter: 过滤 {issue_fingers}（和弦 {ch_id} 只需要 {used_fingers}）"
                    f" at t={t:.1f}s: {issue_text[:60]}"
                )
                filtered_count += 1
                continue

            result.append(h)

        if filtered_count > 0:
            logger.info(f"finger_filter: {len(deduped)} → {len(result)} ({filtered_count} irrelevant)")
        return result

    def _ai_review_hand_issues(self, deduped, detected_chords, level):
        """Phase 2d: AI 审核节点。将候选问题 + KB 上下文发送给 LLM 做最终判定。

        输入：经过 KB 规则 + MediaPipe 置信度预筛选的候选问题列表
        输出：LLM 审核后的问题列表（确认/降级/驳回）
        """
        if not deduped or not self.deepseek_agent or not self.knowledge_db:
            return deduped, []

        try:
            # 收集 KB 上下文
            kb_context_parts = []
            for h in deduped[:5]:
                issue_text = h.get("问题", "")
                # 查找问题对应的 KB problem
                for pid, rules in self.knowledge_db.get_problem_thresholds().items():
                    # 简单关键词匹配找到相关 problem
                    kw_map = {
                        "flat-fingers": ["趴", "太趴", "没立起来", "指腹", "扁平"],
                        "too-vertical-fingers": ["太竖直", "太直", "站得太直"],
                        "pinky-flying": ["小指", "翘", "飞"],
                        "collapsed-wrist": ["手腕", "塌腕", "往里扣"],
                        "thumb-too-high": ["拇指", "捏太紧", "虎口"],
                        "thumb-no-support": ["拇指", "没支撑"],
                        "too-curved-fingers": ["太弯", "紧张", "蜷缩"],
                        "excessive-pressure": ["太用力", "力度"],
                    }
                    for pid_key, keywords in kw_map.items():
                        if any(kw in issue_text for kw in keywords):
                            entry = self.knowledge_db.search(pid_key.replace("-", " "), top_k=1)
                            if entry:
                                kb_context_parts.append(f"[{pid_key}]\n{entry[0].body[:300]}")
                            break

            # 和弦角度标准
            for ch in (detected_chords or [])[:3]:
                cid = ch.get("chord_id", "")
                if cid:
                    angles = self.knowledge_db.get_chord_angle_standards(cid)
                    if angles:
                        angle_str = ", ".join(
                            f"{f}: PIP={a.get('pip_ideal','?')}° MCP={a.get('mcp_ideal','?')}°"
                            for f, a in angles.items()
                        )
                        kb_context_parts.append(f"和弦 {ch.get('chord_name',cid)} 理想角度: {angle_str}")

            kb_context = "\n\n".join(kb_context_parts[:3]) if kb_context_parts else ""

            # 构建审核 prompt
            issues_json = json.dumps([
                {"id": i, "问题": h.get("问题", ""), "时间": h.get("时间", 0),
                 "哪只手": h.get("哪只手", ""), "严重度": h.get("诊断优先级", 3)}
                for i, h in enumerate(deduped[:8])
            ], ensure_ascii=False)

            review_prompt = f"""你是吉他教学专家。审核以下手型检测结果，判断哪些是真正的问题。

学员水平: {level or 'intermediate'}

知识库标准:
{kb_context or '（使用通用生物力学标准）'}

候选问题列表:
{issues_json}

请对每个问题判定:
- "confirmed": 确实是问题
- "downgraded": 存在但轻微，可降级为提示
- "dismissed": 误报，应排除

**severity 判定标准（必须严格遵守）**:
- "high": 影响演奏根本的问题——手指太直/太竖直（指根关节锁死）、手腕塌陷/往里扣、拇指完全错位。这类问题不改无法正常按弦
- "medium": 影响手型美观和效率的问题——手指微趴（没完全立起来）、手指偏紧张、小指轻微翘起
- "low": 轻微可忽略的问题——力度稍大、个别手指偶尔不到位

返回 JSON 格式:
{{"reviewed": [{{"id": 0, "verdict": "confirmed", "adjusted_text": "修正后的问题描述", "severity": "high"}}]}}

仅返回 JSON，不要其他文字。"""

            resp = self.deepseek_agent._call_api(
                [{"role": "user", "content": review_prompt}],
                temperature=0.0, json_mode=True,
            )
            if not resp:
                return deduped, []

            # 解析 LLM 响应
            try:
                resp = resp.strip()
                if resp.startswith("```"):
                    resp = resp.split("\n", 1)[1].rsplit("\n```", 1)[0]
                reviewed = json.loads(resp).get("reviewed", [])
            except (json.JSONDecodeError, KeyError):
                logger.warning(f"AI审核响应解析失败: {resp[:200]}")
                return deduped, []

            # 应用审核结果
            reviewed_map = {r["id"]: r for r in reviewed}
            result = []
            dismissed_list = []
            for i, h in enumerate(deduped):
                rv = reviewed_map.get(i, {})
                verdict = rv.get("verdict", "confirmed")
                if verdict == "dismissed":
                    logger.info(f"AI审核驳回: [{i}] {h.get('问题','')[:40]}")
                    dismissed_list.append(h.get("问题", "")[:80])
                    continue
                if verdict == "downgraded":
                    h["问题"] = "💡 " + h.get("问题", "").lstrip("⚠️💡")
                if rv.get("adjusted_text"):
                    h["问题"] = rv["adjusted_text"]
                if rv.get("severity"):
                    h["_ai_severity"] = rv["severity"]
                else:
                    # LLM 未返回 severity → 从问题类型推断严重度
                    text = h.get("问题", "")
                    prio = h.get("诊断优先级")
                    # 严重问题（影响演奏核心）：手指太直/太竖直、手腕塌陷
                    if any(kw in text for kw in ["太直", "太竖直", "站得太直", "手腕往里扣", "塌腕"]):
                        h["_ai_severity"] = "high"
                    elif text.startswith("⚠️") or (prio is not None and prio <= 1):
                        h["_ai_severity"] = "medium"
                    elif text.startswith("💡") or (prio is not None and prio == 2):
                        h["_ai_severity"] = "low"
                    else:
                        h["_ai_severity"] = "medium"
                result.append(h)

            logger.info(f"AI审核完成: {len(deduped)} → {len(result)} "
                        f"(确认{sum(1 for r in reviewed if r.get('verdict')=='confirmed')}, "
                        f"降级{sum(1 for r in reviewed if r.get('verdict')=='downgraded')}, "
                        f"驳回{sum(1 for r in reviewed if r.get('verdict')=='dismissed')})")
            return (result if result else deduped), dismissed_list

        except Exception as e:
            logger.warning(f"AI审核异常: {e}")
            return deduped, []

    def _group_by_chord(self, deduped, detected_chords):
        """和弦感知分组：同一和弦段内、同一问题类型合并为一条。

        例如：「食指太直」和「中指太直」→ 合并为一条，「中指太弯」→ 另保留一条。
        避免将「太直」和「太弯」这两类矛盾描述塞进一条，导致 AI 审核误驳回。
        """
        if len(deduped) <= 1 or not detected_chords:
            return deduped

        sorted_chords = sorted(detected_chords, key=lambda c: c.get("time", 0))

        def _find_chord_at(t):
            # 严格区间包含优先：时间戳落在和弦 [start, end] 内的优先于任何邻近和弦
            # 避免短和弦（如 fmaj7 dur=1s）被相邻长和弦（如 Am dur=2.7s）淹没
            containing = [c for c in sorted_chords
                          if c.get("time", 0) <= t <= c.get("end_time", c.get("time", 0) + 2.0)]
            if containing:
                containing.sort(key=lambda c: c.get("confidence", 0), reverse=True)
                return containing[0]
            # 次选：窗口匹配（±0.5s），跳过极短且低置信度的段（chroma 噪声）
            best_match = None
            best_dur = 0
            for c in sorted_chords:
                c_start = c.get("time", 0)
                c_end = c.get("end_time", c_start + 2.0)
                if c_start - 0.5 <= t <= c_end + 0.5:
                    dur = c_end - c_start
                    conf = c.get("confidence", 0.5)
                    if dur < 0.8 and conf < 0.5:
                        if best_match is None:
                            best_match = c
                        continue
                    if dur > best_dur:
                        best_match = c
                        best_dur = dur
            if best_match:
                return best_match
            # 兜底：最近和弦（容忍检测漏段，最多 5s）
            if sorted_chords:
                nearest = min(sorted_chords, key=lambda c: abs(c.get("time", 0) - t))
                if abs(nearest.get("time", 0) - t) <= 5.0:
                    return nearest
            return None

        def _problem_category(desc):
            """提取问题类别，用于分组（同类别合并，不同类别分开）。"""
            if "手腕" in desc:
                if "太直" in desc:
                    return "wrist-straight"
                if "太弯" in desc or "往里扣" in desc or "内扣" in desc or "塌" in desc:
                    return "wrist-collapsed"
                return "wrist"
            if "太竖直" in desc or "太直" in desc:
                return "too-straight"
            if "太弯" in desc or "紧张" in desc or "蜷" in desc:
                return "too-curved"
            if "趴" in desc or "塌" in desc or "没立起来" in desc:
                return "too-flat"
            if "飞" in desc or "翘" in desc:
                return "flying"
            if "捏" in desc:
                return "gripping"
            return "other"

        # 按 (和弦, 问题类别) 分组
        groups = {}  # {(chord_id, category): [issue, ...]}
        ungrouped = []
        for h in deduped:
            t = h.get("时间", h.get("timestamp", 0))
            ch = _find_chord_at(t)
            if ch:
                cid = ch.get("chord_id", ch.get("chord_name", ""))
                cat = _problem_category(h.get("问题", ""))
                groups.setdefault((cid, cat), []).append(h)
            else:
                ungrouped.append(h)

        result = []
        for (cid, cat), issues in groups.items():
            if len(issues) <= 1:
                result.extend(issues)
                continue
            # 合并同类别问题：取最早时间, 汇总所有问题描述
            issues.sort(key=lambda x: x.get("时间", x.get("timestamp", 0)))
            base = dict(issues[0])
            all_problems = []
            seen = set()
            for h in issues:
                p = h.get("问题", "")
                if p and p not in seen:
                    seen.add(p)
                    all_problems.append(p)
            if len(all_problems) > 1:
                base["问题"] = "；".join(all_problems)
                all_times = sorted(set(h.get("时间", h.get("timestamp", 0)) for h in issues))
                base["时间"] = all_times[0]
                base["所有检测时间点"] = all_times
                base["出现次数"] = len(all_times)
            result.append(base)

        result.extend(ungrouped)
        result.sort(key=lambda x: x.get("时间", x.get("timestamp", 0)))
        if len(result) < len(deduped):
            logger.info(f"和弦分组: {len(deduped)} → {len(result)} 条")
        return result

    @staticmethod
    def _dedup_by_problem_type(issues):
        """按问题类型去重：同手指+同问题类型的多时间点只保留一条。

        例如「食指太趴」在 2s、5s、9s 各出现一次 → 合并为一条，
        「中指太直」在 3s、7s 出现 → 另保留一条。
        每条代表性问题保留最早时间点，合并所有出现时间点。
        """
        if len(issues) <= 1:
            return issues

        FINGER_KW = ["食指", "中指", "无名指", "小指", "拇指"]
        GENERIC_KW = ["手腕", "虎口", "手掌", "坐姿", "手肘"]

        def _extract_category(desc):
            """从问题描述中提取分类 key。"""
            for fk in FINGER_KW:
                if fk in desc:
                    if "太直" in desc or "太竖直" in desc:
                        return f"{fk}-太直"
                    elif "太弯" in desc:
                        return f"{fk}-太弯"
                    elif "趴" in desc or "塌" in desc or "没立起来" in desc:
                        return f"{fk}-太趴"
                    elif "紧张" in desc or "僵硬" in desc:
                        return f"{fk}-紧张"
                    elif "翘起" in desc or "飞" in desc:
                        return f"{fk}-翘起"
                    elif "捏" in desc:
                        return f"{fk}-捏太紧"
                    elif "缩" in desc or "挤" in desc:
                        return f"{fk}-缩一起"
                    else:
                        return f"{fk}-其他"
            for gk in GENERIC_KW:
                if gk in desc:
                    if gk == "手腕":
                        if "太直" in desc:
                            return "手腕-太直"
                        if "太弯" in desc or "往里扣" in desc or "塌" in desc:
                            return "手腕-内扣"
                    return gk
            return desc[:12] if desc else "unknown"

        categories = {}
        result = []
        for h in issues:
            desc = h.get("问题", "")
            cat = _extract_category(desc)
            if cat not in categories:
                categories[cat] = h
                result.append(h)
            else:
                existing = categories[cat]
                existing_all_times = existing.get("所有检测时间点",
                    [existing.get("时间", existing.get("timestamp", 0))])
                new_t = h.get("时间", h.get("timestamp", 0))
                all_times = sorted(set(
                    list(existing_all_times) + [new_t] +
                    h.get("所有检测时间点", [])
                ))
                existing["所有检测时间点"] = all_times
                existing["出现次数"] = len(all_times)
                existing["时间"] = all_times[0]

        if len(result) < len(issues):
            logger.info(f"问题类型去重: {len(issues)} → {len(result)}")
        return result

    def _annotate_snapshots(self, video_data, audio_result, deduped, first_note_time):
        """截图：每个手型问题配最接近的截图。基于图片内容哈希去重，相同截图合并问题。"""
        raw_snaps = video_data.get("snapshots", [])
        UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "snapshots")

        # 过滤演奏开始前的截图（与 filter_pre_music 逻辑一致）
        if first_note_time > 0:
            cutoff = max(0.5, first_note_time - 0.5)
            raw_snaps = [s for s in raw_snaps if s.get("time", 0) >= cutoff]

        # 计算每张截图的 dhash (key=url 避免同秒截图覆盖)
        snap_hashes = {}  # snap_url -> dhash hex string
        for s in raw_snaps:
            fname = os.path.basename(s["url"])
            fpath = os.path.join(UPLOAD_DIR, fname)
            if os.path.exists(fpath):
                try:
                    snap_hashes[s["url"]] = _image_dhash(fpath)
                except Exception:
                    snap_hashes[s["url"]] = ""

        # {time -> [(severity, label, full_text), ...]}
        issue_index = {}
        for h in deduped:
            full_text = h.get("问题", "")
            label = self._short_label(full_text)
            if not label:
                continue
            sev = 0 if "⚠️" in full_text else 1
            # 优先用"所有检测时间点"中最接近截图的时刻，而非盲目选第一个
            all_times = h.get("所有检测时间点", [])
            if all_times:
                best_t = None
                best_dist = float("inf")
                for at in all_times:
                    at = round(at, 1)
                    if at <= 0:
                        continue
                    for s in raw_snaps:
                        dist = abs(round(s.get("time", 0), 1) - at)
                        if dist < best_dist:
                            best_dist = dist
                            best_t = at
                t = best_t if best_t is not None else round(h.get("时间", h.get("timestamp", 0)), 2)
            else:
                t = round(h.get("时间", h.get("timestamp", 0)), 2)
            if t > 0:
                issue_index.setdefault(t, []).append((sev, label, full_text))

        # 每个问题时间点找最近的截图
        issue_to_snap = {}  # issue_t -> (snap_dict, snap_hash)
        for issue_t, issues in sorted(issue_index.items()):
            best_snap, best_hash = None, ""
            best_dist = float("inf")
            for s in raw_snaps:
                st = round(s.get("time", 0), 1)
                dist = abs(st - issue_t)
                if dist < best_dist:
                    best_dist = dist
                    best_snap = s
                    best_hash = snap_hashes.get(best_snap["url"], "")
            if best_snap and best_dist <= 1.0:
                issue_to_snap[issue_t] = (best_snap, best_hash, issues)

        # 按图片 dhash 分组 → 相同图片内容的问题合并
        DHASH_THRESHOLD = 5   # 汉明距离≤5视为同一图片（更严格，避免把不同手势合并）
        hash_groups = {}  # hash_key -> [(snap_dict, issues_list), ...]

        def _find_hash_group(hash_val):
            """查找或创建哈希分组。相近的哈希归入同一组。"""
            for leader, members in hash_groups.items():
                if _hamming_distance(hash_val, leader) <= DHASH_THRESHOLD:
                    return leader
            return None

        for issue_t, (snap, snap_hash, issues) in issue_to_snap.items():
            if not snap_hash:
                # 无哈希（文件不存在等）：单独成组
                hash_groups.setdefault(f"snap_{issue_t}", []).append((snap, issues))
                continue
            leader = _find_hash_group(snap_hash)
            if leader:
                hash_groups[leader].append((snap, issues))
            else:
                hash_groups[snap_hash] = [(snap, issues)]

        # 每组取第一张截图，合并所有问题
        snaps_final = []
        for hkey, members in hash_groups.items():
            representative = members[0][0]
            all_issues = []
            for _, issues in members:
                all_issues.extend(issues)
            all_issues.sort(key=lambda x: x[0])  # 按严重度
            all_labels = [label for _, label, _ in all_issues]
            all_texts = [txt for _, _, txt in all_issues]

            if len(all_labels) >= 3:
                # 列出前3个具体问题而非笼统的"整体手型需调整"
                top_labels = list(dict.fromkeys(all_labels))[:3]  # 去重取前3
                representative["reason"] = "；".join(top_labels)
                representative["all_issues"] = all_texts
                representative["multi_issue"] = True
            elif len(all_labels) == 2:
                representative["reason"] = f"{all_labels[0]}; {all_labels[1]}"
                representative["all_issues"] = all_texts
            else:
                representative["reason"] = all_labels[0]
                representative["all_issues"] = all_texts
            # 单个问题但涉及多根手指 → 也标记为整体问题
            if representative.get("reason") == "整体手型需调整":
                representative["multi_issue"] = True
            snaps_final.append(representative)

        # 时间维度兜底去重：dhash 未能合并的同秒截图，按时间合并（≤1.0s）
        if len(snaps_final) > 1:
            snaps_final.sort(key=lambda s: s.get("time", 0))
            merged_snaps = []
            i = 0
            while i < len(snaps_final):
                leader = snaps_final[i]
                merged_reasons = [leader.get("reason", "")]
                merged_issues = leader.get("all_issues", [])[:]
                merged_time = leader.get("time", 0)
                j = i + 1
                while j < len(snaps_final) and (snaps_final[j].get("time", 0) - merged_time) <= 1.0:
                    merged_reasons.append(snaps_final[j].get("reason", ""))
                    merged_issues.extend(snaps_final[j].get("all_issues", []))
                    j += 1
                if j > i + 1:
                    leader["reason"] = "; ".join(dict.fromkeys(merged_reasons))
                    leader["all_issues"] = merged_issues
                    leader["multi_issue"] = len(set(merged_reasons)) >= 2
                merged_snaps.append(leader)
                i = j
            snaps_final = merged_snaps

        if not snaps_final:
            snaps_final = [s for s in raw_snaps if not s.get("reason")][:2]

        snaps_final.sort(key=lambda s: s.get("time", 0))
        # 存储完整问题上下文供 AI 查询
        all_issue_context = []
        for h in deduped:
            full_text = h.get("问题", "")
            all_issue_context.append({
                "时间": h.get("时间", h.get("timestamp", 0)),
                "哪只手": h.get("哪只手", ""),
                "问题": full_text,
                "short_label": self._short_label(full_text),
            })
        video_data["snapshots"] = snaps_final
        video_data["issue_context"] = all_issue_context

        # 回写截图 URL 到每个手型问题
        for h in deduped:
            issue_t = round(h.get("时间", h.get("timestamp", 0)), 2)
            if issue_t in issue_to_snap:
                h["截图"] = issue_to_snap[issue_t][0].get("url", "")
            elif not h.get("截图"):
                # 时间没精确命中：找最近截图
                best_url = ""
                best_dist = 99.0
                for snap_t, (snap, _, _) in issue_to_snap.items():
                    dist = abs(snap_t - issue_t)
                    if dist < best_dist and dist <= 1.5:
                        best_dist = dist
                        best_url = snap.get("url", "")
                h["截图"] = best_url

        return video_data

    @staticmethod
    def _get_chroma_candidates(issue_time: float, sorted_chords: list, top_k: int = 3) -> list:
        """返回 issue_time 附近的 chroma top-K 候选和弦（去重、按综合分排序）。

        严格区间包含优先：时间戳落在和弦 [start, end] 内的候选永远排在邻近候选之前。
        这解决了短和弦（如 fmaj7）被相邻高置信度长和弦（如 Am/C）淹没的问题。
        """
        if not sorted_chords:
            return []
        containing = []  # 区间包含 issue_time 的和弦（按置信度排）
        nearby = []      # 区间不包含但在附近的和弦（按邻近度+置信度排）
        for c in sorted_chords:
            t = c.get("time", 0)
            end_t = c.get("end_time", t + 2.0)
            conf = c.get("confidence", 0)
            if t <= issue_time <= end_t:
                containing.append((conf, c))
            else:
                dist = min(abs(issue_time - t), abs(issue_time - end_t))
                if dist < 3.0:
                    time_score = 1.0 - dist / 3.0
                    combined = conf * 0.5 + time_score * 0.5
                    nearby.append((combined, c))
        containing.sort(key=lambda x: -x[0])
        nearby.sort(key=lambda x: -x[0])
        all_ranked = containing + nearby
        seen = set()
        candidates = []
        for score, c in all_ranked:
            cid = c.get("chord_id", "")
            if cid and cid not in seen:
                seen.add(cid)
                candidates.append(dict(c, _candidate_score=round(score, 3)))
                if len(candidates) >= top_k:
                    break
        return candidates

    @staticmethod
    def _count_hand_mismatches(hand_landmarks: list, issue_time: float, chord_id: str) -> int:
        """统计手型和弦模板之间的手指矛盾数（用于惩罚重排）。

        Returns:
            矛盾的手指数量 (0-4)。无手型数据或模板时返回 0（不惩罚）。
        """
        if not hand_landmarks or not chord_id:
            return 0
        chord_id_short = chord_id.replace("chord-", "")
        left_frames = [f for f in hand_landmarks
                       if f.get("hand") == "Left" and abs(f.get("time", 0) - issue_time) < 0.5]
        if not left_frames:
            return 0
        closest = min(left_frames, key=lambda f: abs(f.get("time", 0) - issue_time))
        landmarks = closest.get("landmarks", [])
        if len(landmarks) < 21:
            return 0
        try:
            from chord_analyzer import ChordAnalyzer
            finger_positions = ChordAnalyzer._extract_finger_positions(landmarks)
            templates = ChordAnalyzer._load_chord_templates_with_finger_data()
            chord_data = templates.get(chord_id_short) or templates.get(chord_id)
            if not chord_data:
                return 0
            finger_map = chord_data.get("finger_map", {})
            if not finger_map:
                return 0
            expected_pressing = set(finger_map.keys())
            all_fingers = ["食指", "中指", "无名指", "小指"]
            mismatches = 0
            for finger in all_fingers:
                fp = finger_positions.get(finger)
                if not fp:
                    continue
                is_pressing = fp.get("pressing", False)
                should_press = finger in expected_pressing
                if is_pressing != should_press:
                    mismatches += 1
            return mismatches
        except Exception:
            return 0

    # 音名→pitch class 映射
    _NOTE_TO_PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
                   "E": 4, "Fb": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7,
                   "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}

    @classmethod
    def _chord_pcs(cls, chord_id: str) -> tuple:
        """解析 chord_id 返回 (root_pc, triad_pcs, extra_pcs)。

        triad_pcs: 基础三和弦的 pitch class 集合
        extra_pcs: 扩展音（7音等）的 pitch class 集合，无则为空
        """
        import re
        cid = chord_id.replace("chord-", "")
        m = re.match(r'^([A-G](?:#|b)?)(.*)$', cid)
        if not m:
            return None, set(), set()
        root_name, suffix = m.group(1), m.group(2)
        root = cls._NOTE_TO_PC.get(root_name)
        if root is None:
            return None, set(), set()
        # 三和弦
        if suffix in ("", "maj", "major", "M"):
            is_minor = False
        elif suffix in ("m", "min", "minor"):
            is_minor = True
        elif suffix in ("maj7", "M7", "Δ"):
            is_minor = False
        elif suffix in ("7", "dom7"):
            is_minor = False
        elif suffix in ("m7", "min7", "-7"):
            is_minor = True
        else:
            is_minor = "m" in suffix.lower()
        third = (root + 3) % 12 if is_minor else (root + 4) % 12
        fifth = (root + 7) % 12
        triad = {root, third, fifth}
        # 扩展音
        extra = set()
        if "maj7" in suffix or "M7" in suffix or "Δ" in suffix:
            extra.add((root + 11) % 12)
        elif "7" in suffix and "maj" not in suffix:
            extra.add((root + 10) % 12)
        elif "m7" in suffix or "min7" in suffix:
            extra.add((root + 10) % 12)
        return root, triad, extra

    @classmethod
    def _compute_note_bonus(cls, audio_result: dict, issue_time: float, chord_id: str) -> float:
        """基于 basic-pitch 音符检查和弦的「签名音」是否被实际弹奏。

        检查逻辑：
        1. 七和弦缺七音 → -0.12
        2. 候选和弦的三音/五音缺失（说明候选可能是错的）→ 梯度惩罚
        3. 三和弦出现七音 → -0.03 (给七和弦让路)
        """
        notes = audio_result.get("notes", []) if audio_result else []
        if not notes or not chord_id:
            return 0.0
        root, triad, extra = cls._chord_pcs(chord_id)
        if root is None:
            return 0.0
        window_notes = [n for n in notes
                        if abs(n.get("start_time", 0) - issue_time) < 0.8]
        if not window_notes:
            return 0.0
        played_pcs = set()
        for n in window_notes:
            pitch = n.get("pitch")
            if pitch is not None:
                played_pcs.add(int(pitch) % 12)
        extra_present = extra & played_pcs
        triad_missing = triad - played_pcs
        triad_missing_count = len(triad_missing)
        # 候选和弦的核心三音严重缺失 → 这个候选大概率是错的
        if triad_missing_count == 3:
            return -0.25
        if triad_missing_count == 2:
            return -0.18
        if extra and not extra_present:
            # 七和弦但七音未被弹到 → 扣分
            return -0.12
        if not extra:
            extra_pcs = cls._seventh_for_triad(chord_id)
            if extra_pcs and extra_pcs & played_pcs:
                return -0.03
        return 0.0

    @classmethod
    def _seventh_for_triad(cls, chord_id: str) -> set:
        """对于三和弦，返回其可能的大七/属七扩展音集合（如果该和弦确实没有指定扩展音）。"""
        root, triad, extra = cls._chord_pcs(chord_id)
        if root is None or extra:
            return set()
        # 大三和弦可能有 maj7(根+11) 或 dom7(根+10)
        # 小三和弦可能有 min7(根+10)
        third = (root + 4) % 12
        is_major = third in triad
        if is_major:
            return {(root + 11) % 12, (root + 10) % 12}
        else:
            return {(root + 10) % 12}

    @classmethod
    def _triad_of(cls, chord_id: str) -> Optional[str]:
        """将七和弦简化为对应三和弦 chord_id (含 chord- 前缀)。"""
        import re
        root, triad, extra = cls._chord_pcs(chord_id)
        if root is None or not extra:
            return None
        cid = chord_id.replace("chord-", "")
        m = re.match(r'^([A-G](?:#|b)?)(.*)$', cid)
        if not m:
            return None
        root_name = m.group(1)
        third = (root + 3) % 12
        is_minor = third in triad
        return f"chord-{root_name}m" if is_minor else f"chord-{root_name}"

    def _match_references(self, deduped, detected_chords, technique_hints, video_data,
                           audio_result, vl_chords=None):
        """匹配正确手型参考图。四级回退链（Phase 2e）。

        1. 和弦直接匹配 (detected_chords → chord tag)
        2. 技巧匹配 (technique → technique tag)
        3. 手别匹配 (hand → hand tag)
        4. dual_channel_match() 兜底

        每个问题最多 1 张参考图，时间排序去重。
        """
        reference_images = []
        # 每个 group 独立匹配最佳参考图（允许不同时间段的同一和弦复用参考图）
        sorted_chords = sorted(detected_chords or [], key=lambda c: c.get("time", 0))

        def _find_chord_at(issue_time: float):
            """chroma 提名 + VL + 音符 四重评分：返回最匹配的和弦候选。

            使用原始 chroma 检测结果（含多候选），而非融合后的单一结果。
            VL 和弦提供高权重视觉基准（+0.15 奖励）。
            """
            # 优先用融合/合并后的和弦（段边界干净，避免原始 chroma 的 flickering 噪声）
            # 原始 chroma 每帧独立匹配，产生大量短段，会污染 top-K 候选列表
            chroma_chords = audio_result.get("detected_chords", []) if audio_result else []
            source_chords = sorted_chords if sorted_chords else chroma_chords
            if not source_chords:
                return None
            candidates = AnalysisService._get_chroma_candidates(issue_time, source_chords, top_k=4)
            if not candidates:
                return None
            # 扩展候选池：确保每个七和弦的三和弦版本也在候选列表中
            existing_ids = {c.get("chord_id", "") for c in candidates}
            expansions = []
            for c in list(candidates):
                triad_cid = AnalysisService._triad_of(c.get("chord_id", ""))
                if triad_cid and triad_cid not in existing_ids:
                    existing_ids.add(triad_cid)
                    expansions.append({
                        "chord_id": triad_cid,
                        "chord_name": triad_cid.replace("chord-", ""),
                        "_candidate_score": round(c["_candidate_score"] - 0.02, 3),
                        "_expanded_from": c.get("chord_id", ""),
                    })
            candidates.extend(expansions)
            if expansions:
                logger.info(f"[ref-match] 候选扩展: +{[e['chord_id'] for e in expansions]}")

            # Find nearest VL verification for bonus
            vl_chord_id = None
            vl_chord_triad = None
            vl_chords_list = vl_chords or []
            if vl_chords_list:
                best_vl_diff = float("inf")
                for vc in vl_chords_list:
                    diff = abs(vc.get("time", 0) - issue_time)
                    if diff < 2.0 and diff < best_vl_diff:
                        best_vl_diff = diff
                        # 新格式用 vl_chosen (VL 验证选择)，老格式用 chord_id (VL 独立检测)
                        vl_chord_id = vc.get("vl_chosen") or vc.get("chord_id", "")
                if vl_chord_id:
                    vl_chord_triad = AnalysisService._triad_of(vl_chord_id)

            # Find nearest FUSED chord for consistency bonus
            # The fused result already did VL+chroma fusion, so it's more reliable
            # than raw chroma candidates. Give its chord a bonus to prevent divergence.
            # 关键：只取区间包含 issue_time 的和弦，避免邻近高置信度和弦覆盖当前和弦
            fusion_chord_id = None
            for fc in sorted_chords:
                fc_start = fc.get("time", 0)
                fc_end = fc.get("end_time", fc_start + 2.0)
                if fc_start <= issue_time <= fc_end:
                    fusion_chord_id = fc.get("chord_id", "")
                    break

            hand_landmarks = video_data.get("hand_landmarks", [])
            for candidate in candidates:
                cid = candidate.get("chord_id", "")
                # MediaPipe 手指矛盾
                mismatches = AnalysisService._count_hand_mismatches(hand_landmarks, issue_time, cid)
                hand_penalty = mismatches * 0.08
                # basic-pitch 音符检查
                note_bonus = AnalysisService._compute_note_bonus(audio_result, issue_time, cid)
                # VL 奖励
                vl_bonus = 0.0
                if vl_chord_id and cid:
                    if cid == vl_chord_id:
                        vl_bonus = 0.15
                    elif vl_chord_triad and cid == vl_chord_triad:
                        vl_bonus = 0.10
                    elif cid.replace("chord-", "") == vl_chord_id.replace("chord-", ""):
                        vl_bonus = 0.15
                # 融合一致性奖励：融合结果已经综合了 VL+chroma，比 raw chroma 更可靠
                fusion_bonus = 0.0
                if fusion_chord_id and cid and fusion_chord_id.replace("chord-", "") == cid.replace("chord-", ""):
                    fusion_bonus = 0.12
                candidate["_mismatches"] = mismatches
                candidate["_note_bonus"] = round(note_bonus, 3)
                candidate["_vl_bonus"] = round(vl_bonus, 3)
                candidate["_fusion_bonus"] = round(fusion_bonus, 3)
                # 区间包含加分：候选区间包含 issue_time 的和弦获得 0.22 加分
                # 防止高置信度邻近和弦（如 G conf=0.61 邻接）覆盖包含 chords（如 D conf=0.52 包含 4.4s）
                c_start = candidate.get("time", 0)
                c_end = candidate.get("end_time", c_start + 2.0)
                containment_bonus = 0.22 if (c_start <= issue_time <= c_end) else 0.0
                candidate["_containment_bonus"] = round(containment_bonus, 3)
                candidate["_adjusted_score"] = round(
                    candidate["_candidate_score"] - hand_penalty + note_bonus + vl_bonus + fusion_bonus + containment_bonus, 3)
            candidates.sort(key=lambda c: c["_adjusted_score"], reverse=True)
            best = candidates[0]
            cid_log = ", ".join(f"{c['chord_id']}(adj={c['_adjusted_score']:.3f},hand={c['_mismatches']},"
                               f"note={c['_note_bonus']},vl={c.get('_vl_bonus',0):.2f},fuse={c.get('_fusion_bonus',0):.2f},contain={c.get('_containment_bonus',0):.2f})"
                               for c in candidates[:4])
            logger.info(f"[ref-match] 四重评分 @{issue_time:.1f}s → {cid_log} → 选用: {best['chord_id']}")
            return best

        def _problem_category(issue_text):
            """将问题归类为 wrist_thumb/finger，用于分组匹配对应视角的参考图"""
            if "手腕" in issue_text or "拇指" in issue_text:
                return "wrist_thumb"
            return "finger"

        # 按 (时间, 和弦, 问题类型) 分组
        # 手腕/拇指/手指各自匹配各自的参考图（俯视图 vs 正面图）
        # 先拆分复合问题：若一条描述同时包含手腕/拇指和手指问题，拆成两条独立条目
        FINGER_KEYWORDS = ["食指", "中指", "无名指", "小指"]
        split_deduped = []
        for h in (deduped or []):
            problem_text = h.get("问题", "")
            has_wrist_thumb = "手腕" in problem_text or "拇指" in problem_text
            has_finger = any(f in problem_text for f in FINGER_KEYWORDS)
            if has_wrist_thumb and has_finger:
                segments = [s.strip() for s in problem_text.split("；") if s.strip()]
                wt_segs = [s for s in segments if "手腕" in s or "拇指" in s]
                finger_segs = [s for s in segments if any(f in s for f in FINGER_KEYWORDS)]
                # 不属于以上两类的片段（如"试着自然一些"）归入两者
                other_segs = [s for s in segments if s not in wt_segs and s not in finger_segs]
                if wt_segs:
                    wt_entry = dict(h)
                    wt_entry["问题"] = "；".join(wt_segs + other_segs)
                    wt_entry["_split_cat"] = "wrist_thumb"
                    split_deduped.append(wt_entry)
                if finger_segs:
                    finger_entry = dict(h)
                    finger_entry["问题"] = "；".join(finger_segs + other_segs)
                    finger_entry["_split_cat"] = "finger"
                    split_deduped.append(finger_entry)
                if not wt_segs and not finger_segs:
                    split_deduped.append(h)
            else:
                split_deduped.append(h)
        grouped_issues = []  # [(issue_time, chord_id, ch_name, issue_hand, category, group_issues)]
        for h in split_deduped:
            issue_time = h.get("时间", h.get("timestamp", 0))
            ch = _find_chord_at(issue_time)
            chord_id = ch.get("chord_id", "") if ch else ""
            ch_name = ch.get("chord_name", "") if ch else ""
            if chord_id and not chord_id.startswith("chord-"):
                chord_id = f"chord-{chord_id}"
            # 规范化大小写：chord-fmaj7 → chord-Fmaj7
            if chord_id.startswith("chord-"):
                base = chord_id[6:]  # 去掉 "chord-" 前缀
                normalized = base[0].upper() + base[1:] if base else base
                chord_id = f"chord-{normalized}"
            chord_id_for_ref = chord_id
            issue_hand = h.get("哪只手", "")
            cat = h.get("_split_cat") or _problem_category(h.get("问题", ""))

            # 找同组（时间差≤1.0s 且 和弦相同 且 问题类型相同）
            found_group = False
            for g in grouped_issues:
                g_time, g_chord, g_name, g_hand, g_cat, g_issues = g
                if abs(issue_time - g_time) <= 1.0 and chord_id == g_chord and chord_id and cat == g_cat:
                    g_issues.append(h)
                    found_group = True
                    break
            if not found_group:
                grouped_issues.append((issue_time, chord_id, ch_name, issue_hand, cat, [h]))

        # 二次合并：同和弦同类型的组 → 减少参考图张数
        merged_groups = []
        skip_idx = set()
        for i, g1 in enumerate(grouped_issues):
            if i in skip_idx:
                continue
            t1, ch1, name1, hand1, cat1, issues1 = g1
            if cat1 not in ("finger", "wrist_thumb"):
                merged_groups.append(g1)
                continue
            # 查找同和弦 ±2s 内的同类型组，合并
            merged_issues = list(issues1)
            merged_time = t1
            for j, g2 in enumerate(grouped_issues):
                if j <= i or j in skip_idx:
                    continue
                t2, ch2, name2, hand2, cat2, issues2 = g2
                if cat2 == cat1 and ch1 == ch2 and ch1 and abs(t1 - t2) <= 2.0:
                    merged_issues.extend(issues2)
                    merged_time = min(merged_time, t2)
                    skip_idx.add(j)
            merged_groups.append((merged_time, ch1, name1, hand1, cat1, merged_issues))
        grouped_issues = merged_groups

        # 保持顺序：wrist_thumb 在 finger 前面
        grouped_issues.sort(key=lambda g: (g[4] != "wrist_thumb", g[0]))

        for issue_time, chord_id, ch_name, issue_hand, cat, group_issues in grouped_issues:
            # 取组内第一个问题的文本作为代表（用于问题类型匹配）
            issue_text = group_issues[0].get("问题", "")

            chord_id_for_ref = chord_id

            # 诊断日志
            chord_list = [(c.get('chord_id','?'), c.get('time',0), c.get('end_time', c.get('time',0)+2))
                          for c in (sorted_chords or [])[:8]]
            logger.info(f"[ref-match] group@t={issue_time:.1f}s cat={cat} x{len(group_issues)} issues, "
                        f"chord_id={chord_id} ch_name={ch_name} "
                        f"chord_list={chord_list}")

            # 手腕/拇指问题 → 优先用俯视图（展示拇指位置和手腕弧度最清楚）
            is_wrist_thumb = cat == "wrist_thumb"
            result = None
            if is_wrist_thumb:
                all_refs = self.reference_db.list_all()
                top_down_refs = [r for r in all_refs if "俯视图" in str(r.get("tags", []))]
                # 优先匹配特定和弦的俯视图（chord标签少=特定），万能参考图（标签多）仅作兜底
                specific_match = None
                generic_match = None
                for r in top_down_refs:
                    r_tags = r.get("tags", [])
                    if chord_id_for_ref and isinstance(r_tags, list) and chord_id_for_ref in r_tags:
                        n_chord_tags = sum(1 for t in r_tags if str(t).startswith("chord-"))
                        if n_chord_tags < 6:
                            specific_match = r
                            break  # 找到特定和弦俯视图，立即使用
                        elif not generic_match:
                            generic_match = r  # 记下万能参考图作为备用
                if specific_match:
                    result = specific_match
                elif generic_match:
                    result = generic_match
                elif top_down_refs:
                    result = top_down_refs[0]
                if result:
                    thumb_score = 35  # 拇指/手腕通道基础分
                    res_tags = result.get("tags", [])
                    has_chord = chord_id_for_ref and isinstance(res_tags, list) and chord_id_for_ref in res_tags
                    if has_chord:
                        n_chord_tags = sum(1 for t in res_tags if str(t).startswith("chord-"))
                        thumb_score += 15 if n_chord_tags >= 6 else 80
                    if ch_name and ch_name in str(result.get("tags", [])):
                        thumb_score += 35
                    if result.get("hand") == issue_hand:
                        thumb_score += 30
                    if thumb_score <= 35:
                        logger.info(f"[ref-match] 拇指/手腕俯视图低分(score={thumb_score})，回退到 dual_channel_match")
                        result = None
                    else:
                        logger.info(f"[ref-match] 拇指/手腕问题 → 俯视图 ref_id={result.get('id','')} score={thumb_score}")
                        result["_score"] = thumb_score
                        result["_channel"] = "thumb_wrist_topdown"

            if not result:
                # 手指问题→正视图优先，手腕/拇指→俯视图优先
                preferred_view_tag = "正视图" if cat == "finger" else "俯视图"
                # 使用组内所有问题做 dual_channel_match（更完整的问题上下文）
                candidates = self.reference_db.dual_channel_match(
                    group_issues,
                    preferred_hand=issue_hand or "左手",
                    chord_id=chord_id_for_ref,
                    chord_name=ch_name,
                    top_k=10,
                    preferred_view=preferred_view_tag,
                )
                chord_matched = []
                chord_unmatched = []
                for c in candidates:
                    ref_tags = str(c.get("tags", []))
                    view_bonus = 1.0 if preferred_view_tag in ref_tags else 0.0
                    c["_view_bonus"] = view_bonus
                    c["_is_preferred_view"] = preferred_view_tag in ref_tags
                    if chord_id_for_ref and chord_id_for_ref in ref_tags:
                        chord_matched.append(c)
                    else:
                        chord_unmatched.append(c)
                # 手腕/拇指：俯视图优先（即使和弦不完全匹配也比正视图更合适）
                if is_wrist_thumb:
                    chord_top = [c for c in chord_matched if c.get("_is_preferred_view")]
                    any_top = [c for c in chord_unmatched if c.get("_is_preferred_view")]
                    if chord_top:
                        result = chord_top[0]
                    elif any_top:
                        result = any_top[0]
                        logger.info(f"[ref-match] 手腕/拇指：无{chord_id}俯视图，使用通用俯视图 ref_id={result.get('id','')}")
                    elif chord_matched:
                        result = chord_matched[0]
                    elif candidates:
                        result = candidates[0]
                else:
                    # 手指：和弦匹配优先，同和弦内视角匹配者优先
                    chord_matched.sort(key=lambda c: c.get("_view_bonus", 0), reverse=True)
                    chord_unmatched.sort(key=lambda c: c.get("_view_bonus", 0), reverse=True)
                    if chord_matched:
                        result = chord_matched[0]
                    elif chord_unmatched and chord_id_for_ref:
                        # No chord-matched reference exists. Check if the best unmatched
                        # candidate has a DIFFERENT chord tag — showing F reference
                        # for C chord is worse than showing nothing.
                        best_unmatched = chord_unmatched[0]
                        um_tags = str(best_unmatched.get("tags", []))
                        other_chords = [t for t in best_unmatched.get("tags", [])
                                        if str(t).startswith("chord-") and t != chord_id_for_ref]
                        if other_chords:
                            logger.warning(f"[ref-match] 无{chord_id_for_ref}参考图，最佳候选是{other_chords[0]}，"
                                          f"跳过以免误导（显示F参考图给C和弦）")
                            result = None
                        else:
                            result = best_unmatched
                    elif chord_unmatched:
                        result = chord_unmatched[0]
                    if not result and candidates:
                        result = candidates[0]

            if result:
                rid = result.get("id", "")
                # matched_ids.add(rid)  # 允许不同时间组复用
                # 为整个组构建一条参考图记录（合并组内所有问题类型）
                all_problem_types = []
                all_labels = []
                for h in group_issues:
                    problem_text = h.get("问题", "")
                    problem_type = ""
                    # 身体部位优先（手腕/拇指需要专属参考图）
                    if "手腕" in problem_text:
                        problem_type = "wrist:collapsed"
                    elif "拇指" in problem_text:
                        problem_type = "thumb:too-high"
                    elif "小指" in problem_text and ("翘" in problem_text or "飞" in problem_text):
                        problem_type = "finger:pinky-flying"
                    # 手指问题类型
                    elif "太直" in problem_text or "太竖直" in problem_text:
                        problem_type = "finger:too-straight"
                    elif "太趴" in problem_text or "塌" in problem_text or "没立起来" in problem_text:
                        problem_type = "finger:too-flat"
                    elif "太弯" in problem_text or "紧张" in problem_text:
                        problem_type = "finger:too-curved"
                    else:
                        problem_type = "finger:general"
                    all_problem_types.append(problem_type)
                    all_labels.append(problem_type.split(":")[-1] if ":" in problem_type else problem_type)

                # 去重并合并问题类型
                unique_types = list(dict.fromkeys(all_problem_types))
                # 2+ 同类型手指问题 → 使用"整体"标签（数原始问题数，不去重）
                raw_straight = sum(1 for t in all_problem_types if t == "finger:too-straight")
                raw_curved = sum(1 for t in all_problem_types if t == "finger:too-curved")
                raw_flat = sum(1 for t in all_problem_types if t == "finger:too-flat")
                if raw_straight >= 2:
                    unique_types = [t for t in unique_types if t != "finger:too-straight"] + ["finger:overall-straight"]
                if raw_curved >= 2:
                    unique_types = [t for t in unique_types if t != "finger:too-curved"] + ["finger:overall-curved"]
                if raw_flat >= 2:
                    unique_types = [t for t in unique_types if t != "finger:too-flat"] + ["finger:overall-flat"]
                # 重排序: 整体标签优先，wrist/thumb 其次
                def _label_priority(t):
                    if "overall" in t:
                        return 0
                    if t.startswith("wrist") or t.startswith("thumb"):
                        return 1
                    return 2
                unique_types.sort(key=_label_priority)
                channel_label = "; ".join(unique_types)
                if ch_name:
                    channel_label += f" ({ch_name})"

                # 匹配分数和原因
                match_score = result.get("_score", 0)
                match_channel = result.get("_channel", "dual_channel")
                match_reason_parts = []
                if ch_name:
                    match_reason_parts.append(f"匹配和弦{ch_name}")
                if unique_types:
                    type_names = {"finger:too-straight": "手指过直", "finger:too-flat": "手指塌陷",
                                  "finger:too-curved": "手指过弯", "wrist:collapsed": "手腕塌陷",
                                  "thumb:too-high": "拇指位置", "finger:pinky-flying": "小指翘起",
                                  "finger:general": "手指问题",
                                  "finger:overall-straight": "手型整体过于竖直",
                                  "finger:overall-curved": "手型整体过于弯曲",
                                  "finger:overall-flat": "手型整体塌陷"}
                    display_types = [type_names.get(t, t) for t in unique_types]
                    match_reason_parts.append("; ".join(display_types))
                match_reason = ", ".join(match_reason_parts) if match_reason_parts else match_channel

                # 组内所有问题的时间点
                group_times = [h.get("时间", h.get("timestamp", 0)) for h in group_issues]

                # 判断匹配到的图是否为俯视图（供前端显示正确的视角标签）
                result_tags = result.get("tags", [])
                is_topdown = "俯视图" in str(result_tags)
                is_front = "正视图" in str(result_tags)

                reference_images.append({
                    "id": result["id"],
                    "description": result["description"],
                    "body_part": result["body_part"],
                    "hand": result["hand"],
                    "technique": result["technique"],
                    "tags": result["tags"],
                    "image_url": self.reference_db.image_url(result),
                    "channel": channel_label,
                    "issue_time": issue_time,
                    "group_times": group_times,
                    "group_issue_count": len(group_issues),
                    "match_score": match_score,
                    "match_reason": match_reason,
                    "chord_name": ch_name,
                    "chord_matched": bool(chord_id and isinstance(result.get("tags", []), list) and chord_id in result.get("tags", [])),
                    "is_topdown": is_topdown,
                    "is_front": is_front,
                })

        # 兜底
        if len(reference_images) < 1:
            # 尝试从 deduped 中提取和弦信息
            fallback_chord_id = ""
            fallback_chord_name = ""
            if sorted_chords:
                fb_times = [h.get("时间", h.get("timestamp", 0)) for h in (deduped or [])]
                if fb_times:
                    fb_t = sum(fb_times) / len(fb_times)
                    fb_ch = _find_chord_at(fb_t)
                    if fb_ch:
                        fallback_chord_id = fb_ch.get("chord_id", "")
                        fallback_chord_name = fb_ch.get("chord_name", "")
                        if fallback_chord_id and not fallback_chord_id.startswith("chord-"):
                            fallback_chord_id = f"chord-{fallback_chord_id}"
            fallback = self.reference_db.dual_channel_match(
                deduped, preferred_hand="左手", top_k=2,
                chord_id=fallback_chord_id, chord_name=fallback_chord_name,
            )
            for ref in fallback[:2]:
                fb_tags = ref.get("tags", [])
                reference_images.append({
                        "id": ref["id"], "description": ref["description"],
                        "body_part": ref["body_part"], "hand": ref["hand"],
                        "technique": ref["technique"], "tags": ref["tags"],
                        "image_url": self.reference_db.image_url(ref),
                        "channel": "fallback", "issue_time": 0,
                        "group_times": [], "group_issue_count": 0,
                        "match_score": ref.get("_score", 0),
                        "match_reason": "兜底匹配",
                        "chord_name": fallback_chord_name,
                        "chord_matched": bool(fallback_chord_id and isinstance(fb_tags, list) and fallback_chord_id in fb_tags),
                        "is_topdown": "俯视图" in str(fb_tags),
                        "is_front": "正视图" in str(fb_tags),
                    })

        logger.info(f"参考图匹配完成: {len(reference_images)} 张（{len(deduped)}个问题）")
        reference_images.sort(key=lambda r: r.get("issue_time", 0))
        return reference_images[:max(len(deduped) + 1, 2)]

    def _run_vision_analysis(self, all_snaps, instrument, reference_images, deduped, video_data):
        """执行 Vision AI 分析"""
        vision_result = {"vision_issues": [], "vision_available": False}
        UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "snapshots")

        snap_paths = []
        for s in all_snaps:
            fname = os.path.basename(s["url"])
            fpath = os.path.join(UPLOAD_DIR, fname)
            if os.path.exists(fpath):
                snap_paths.append(fpath)
        if not snap_paths:
            return vision_result

        # 构建 RAG 上下文
        freeplay_ctx = list(video_data.get("scored_frames", []))
        try:
            if self.knowledge_db:
                search_terms = [h.get("问题", "") for h in deduped[:3] if h.get("问题", "")]
                if search_terms:
                    entries = self.knowledge_db.search(" ".join(search_terms), top_k=3)
                    if entries:
                        freeplay_ctx.append({"body": self.knowledge_db.format_for_prompt(entries), "source": "rag"})
        except Exception as e:
            logger.warning(f"RAG 失败: {e}")

        ref_paths = [os.path.join(self.reference_db.images_dir, r["image_path"])
                     for r in reference_images if "image_path" in r]
        ref_paths = [p for p in ref_paths if os.path.exists(p)]

        return self.vision_analyzer.analyze_frames(
            snap_paths, instrument,
            reference_paths=ref_paths if ref_paths else None,
            mediapipe_context=freeplay_ctx if freeplay_ctx else None,
        )

    def _merge_vision_results(self, video_data, deduped, vision_result):
        """合并 Vision AI 结果到已有手型问题。

        策略（防止非确定性）：
        - 只用 high 置信度结果
        - 如果 MediaPipe 已检测到相似问题，不重复添加
        - Vision AI 最多补充 2 条新问题
        """
        conf_map = {"high": 0.9, "medium": 0.6, "low": 0.3}
        new_issues = []
        vision_hand = vision_result.get("handedness", "")
        existing_texts = [h.get("问题", "")[:20] for h in (deduped or [])]
        for vi in vision_result.get("vision_issues", []):
            # 只接受 high 置信度，降低非确定性
            conf = vi.get("confidence", "medium")
            if conf != "high":
                logger.debug(f"Vision AI issue skipped (confidence={conf}): {vi.get('description', '')[:40]}")
                continue
            per_issue_hand = vi.get("handedness", vision_hand)
            desc = vi.get("description", "")
            # 与已有 MediaPipe 问题去重：提取手指名 + 问题类型做语义匹配
            is_dup = False
            _finger_kw = ["食指", "中指", "无名指", "小指", "拇指"]
            _problem_kw = ["太直", "太竖直", "竖直", "太弯", "太趴", "没立起来", "立起来",
                          "紧张", "手腕", "塌", "太低", "太紧", "太松", "捏太紧", "位置",
                          "塌陷", "内扣", "扣", "平直", "太平", "弯曲", "过度弯曲",
                          "蜷", "僵硬", "过度"]
            def _extract_fingers(text):
                """返回文本中所有提到的手指集合。"""
                found = set()
                for fk in _finger_kw:
                    if fk in text:
                        found.add(fk)
                return found
            def _extract_problem(text):
                for pk in _problem_kw:
                    if pk in text:
                        return pk
                return ""
            vi_fingers = _extract_fingers(desc)
            vi_problem = _extract_problem(desc)
            for h in (deduped or []):
                et = h.get("问题", "")
                mp_fingers = _extract_fingers(et)
                mp_problem = _extract_problem(et)
                common_fingers = vi_fingers & mp_fingers
                # 有共同手指 + 同类问题 → 去重
                if common_fingers:
                    if vi_problem and mp_problem and vi_problem == mp_problem:
                        is_dup = True
                        break
                    # 不同关键词但同类（太直≈太竖直≈竖直）
                    straight_kw = {"太直", "太竖直", "竖直"}
                    if vi_problem in straight_kw and mp_problem in straight_kw:
                        is_dup = True
                        break
                    bent_kw = {"太弯", "太趴", "没立起来", "立起来", "塌", "塌陷", "弯曲", "过度弯曲", "蜷", "过度"}
                    if vi_problem in bent_kw and mp_problem in bent_kw:
                        is_dup = True
                        break
                    thumb_kw = {"太低", "太紧", "太松", "捏太紧", "位置"}
                    if vi_problem in thumb_kw and mp_problem in thumb_kw:
                        is_dup = True
                        break
                    # Vision AI 没提取到问题类型 → 同手指即认为是重复
                    if not vi_problem:
                        is_dup = True
                        break
                # 手腕/身体部位问题去重（无手指名，用问题关键词匹配）
                body_parts = {"手腕", "手肘", "坐姿"}
                if not vi_fingers and vi_problem in body_parts:
                    if mp_problem == vi_problem or (vi_problem == "手腕" and mp_problem in ("手腕", "太直", "平直", "内扣")):
                        is_dup = True
                        break
                # 旧兜底：字符集交集（仅当两者都没提取到手指/问题时使用）
                if not vi_fingers and not vi_problem and not mp_fingers and not mp_problem:
                    common = len(set(desc[:20]) & set(et))
                    if common >= max(len(desc[:20]), len(et)) * 0.5:
                        is_dup = True
                        break
            if is_dup:
                logger.debug(f"Vision AI issue dedup: {desc[:60]}")
                continue
            logger.debug(f"Vision AI issue NOT deduped (added as new): {desc[:60]}")
            # 为 Vision AI 新问题匹配最近的截图
            vi_snap_url = ""
            vi_ts = vi.get("timestamp", 0)
            vi_snaps = video_data.get("snapshots", [])
            if vi_snaps:
                best_snap = min(vi_snaps, key=lambda s: abs(s.get("time", 0) - vi_ts))
                if abs(best_snap.get("time", 0) - vi_ts) <= 0.8:
                    vi_snap_url = best_snap.get("url", "")
            new_issues.append({
                "哪只手": per_issue_hand if per_issue_hand else "未知",
                "问题": desc,
                "严重程度": vi.get("severity", "moderate"),
                "建议": vi.get("suggestion", ""),
                "时间": vi_ts,
                "来源": "Vision AI",
                "置信度": conf_map.get(conf, 0.9),
                "截图": vi_snap_url,
                "_ai_severity": "high",  # Vision AI only passes high confidence
            })

        if new_issues:
            # 最多补充 2 条，减少非确定性影响
            new_issues = new_issues[:2]
            all_deduped = dedup_by_time(list(deduped) + new_issues)
            video_data["hand_issues"] = all_deduped
            video_data["vision_analyzed"] = True
            logger.info(f"Vision AI 新增 {len(new_issues)} 条问题 (仅 high 置信度)")

        # 右手技巧误报过滤 + 手标签修正
        if video_data.get("vision_analyzed"):
            vision_handedness = vision_result.get("handedness", "")
            is_right_tech = (vision_handedness == "右手")
            if is_right_tech:
                left_kw = ["手指较平", "手指没立", "指尖没有垂直", "按弦", "趴下去", "塌指",
                           "食指按", "中指按", "无名指按", "小指按", "太弯了", "太直了", "没立起来"]
                all_deduped = video_data["hand_issues"]
                filtered = [h for h in all_deduped
                            if not ("左手" in h.get("哪只手", "") and any(kw in h.get("问题", "") for kw in left_kw))]
                if len(filtered) < len(all_deduped):
                    logger.info(f"过滤了 {len(all_deduped) - len(filtered)} 条右手技巧误报")
                    video_data["hand_issues"] = filtered

            if vision_handedness in ("左手", "右手"):
                for h in video_data["hand_issues"]:
                    who = h.get("哪只手", "")
                    mp_conf = h.get("置信度", 1.0)
                    disagrees = (
                        (vision_handedness == "右手" and "左手" in who) or
                        (vision_handedness == "左手" and "右手" in who)
                    )
                    if disagrees and mp_conf < 0.7:
                        corrected = who.replace("左手", "右手") if "左手" in who else who.replace("右手", "左手")
                        h["哪只手"] = corrected

        return video_data

    def _determine_hand_status(self, video_data):
        actual_issues = video_data.get("hand_issues", [])
        analysis_parsed = {}
        try:
            analysis_parsed = json.loads(video_data.get("frame_analysis_data", "{}"))
        except Exception as e:
            logger.warning(f"frame_analysis_data JSON解析失败: {e}")
        hands_detected = analysis_parsed.get("hands_found", False)
        # Fallback: check hand_landmarks and snapshots as secondary indicators
        if not hands_detected:
            if video_data.get("hand_landmarks") or video_data.get("snapshots"):
                hands_detected = True
        has_real_issues = any(
            "正常" not in h.get("问题", "") and "基本正常" not in h.get("问题", "")
            for h in actual_issues
        ) if actual_issues else False

        if not actual_issues and not hands_detected:
            video_data["hand_status"] = "nHand"
        elif has_real_issues:
            video_data["hand_status"] = "issues"
        else:
            video_data["hand_status"] = "ok"

        logger.info(
            f"手型状态判定: status={video_data['hand_status']} "
            f"(issues_count={len(actual_issues)}, hands_found={analysis_parsed.get('hands_found', '?')}, "
            f"has_real_issues={has_real_issues}, landmarks={bool(video_data.get('hand_landmarks'))}, "
            f"snapshots={bool(video_data.get('snapshots'))})"
        )
        return video_data

    @staticmethod
    def _extract_ai_audio_observations(report_md: str, existing_errors: list) -> list:
        """从 AI 报告中提取音频相关观察，补充到前端检测详情列表。

        仅提取 AI 音频章节中的 > 引用行中描述具体问题的观察，
        排除练习建议（如「试试节拍器」「建议练习」等）。
        """
        import re

        # 匹配 AI 音频章节中的 > 引用行（仅第一层引用，非列表/建议）
        audio_kw = r'(节奏|节拍|拍子|速度|时值|越弹越快|忽快忽慢|抢拍|拖拍|不稳|不均匀|着急|往前赶|偏快|偏慢)'
        patterns = [
            # AI 写的 > 引用行，如 "> 节奏上有点着急，像赶着去上课一样"
            rf'>\s*{audio_kw}[^>]*?[。\n]',
        ]

        # 排除练习建议/指令的关键词
        advice_kw = ['试试', '建议', '练习', '可以尝试', '节拍器', '跟着', '每拍',
                     '打开手机', '调到', 'Tempo', 'BPM', 'bpm', '下载']

        observations = []
        seen_texts = set()

        for e in existing_errors:
            detail = e.get('detail', '')
            if detail:
                seen_texts.add(detail[:20])

        for pattern in patterns:
            for match in re.finditer(pattern, report_md):
                text = match.group(0).strip()
                text = re.sub(r'^>\s*', '', text).strip()
                if len(text) < 6 or len(text) > 80:
                    continue
                if any(skip in text for skip in ['✅', '良好', '未检测', '正常', '准确']):
                    continue
                if any(adv in text for adv in advice_kw):
                    continue
                key = text[:20]
                if key in seen_texts:
                    continue
                seen_texts.add(key)
                observations.append({
                    'type': 'ai_observation',
                    'detail': f'💡 {text}',
                    'time': 0,
                    'severity': 'low',
                })

        if observations:
            logger.info(f"从 AI 报告中提取了 {len(observations)} 条音频观察")
        return observations[:3]

    @staticmethod
    def _validate_fret_positions(report_md: str, audio_result: dict) -> str:
        """校验报告中的品位描述是否在音频数据中真实出现过。

        从 audio_result.notes 中提取所有实际出现的 guitar_position，
        对照报告中的「X弦Y品」模式，若出现数据中不存在的品位组合则替换并记录 warning。
        """
        import re

        notes = audio_result.get("notes", [])
        actual_positions = set()
        for n in notes:
            pos = n.get("guitar_position", "")
            if pos and len(pos) >= 3:
                actual_positions.add(pos)

        # Also collect from errors which may reference positions
        for e in audio_result.get("errors", []):
            detail = e.get("detail", "")
            for m in re.finditer(r'[一二三四五六]弦[零一二三四五六七八九十]+品|空品', detail):
                actual_positions.add(m.group(0))

        if not actual_positions:
            return report_md

        pattern = re.compile(r'([一二三四五六]弦)([零一二三四五六七八九十]+品|空品)')
        replaced_count = 0

        def _validate_match(m):
            nonlocal replaced_count
            full = m.group(0)
            if full in actual_positions:
                return full
            replaced_count += 1
            logger.warning(f"品位幻觉: '{full}' 不在实际数据中 (已知: {sorted(actual_positions)[:10]})")
            return "[品位待核实]"

        report_md = pattern.sub(_validate_match, report_md)
        if replaced_count > 0:
            logger.warning(f"品位校验: 共修正 {replaced_count} 处幻觉品位")
        return report_md

    @staticmethod
    def _calc_deterministic_technique(hand_issues):
        """确定性手型分：基于严重程度加权扣分 + 问题数量惩罚。

        优先级: _ai_severity > 诊断优先级 > ⚠️/💡前缀 > 文本模式匹配
        地板: 不低于 70（少量轻微问题不应导致分数崩溃）
        """
        if not hand_issues:
            return 100
        deductions = 0
        for h in hand_issues:
            prio = h.get("诊断优先级")
            text = h.get("问题", "")
            ai_sev = h.get("_ai_severity", "")
            has_warn = text.startswith("⚠️")
            has_tip = text.startswith("💡")

            if ai_sev == "high":
                deductions += 10
            elif ai_sev == "medium":
                deductions += 5
            elif ai_sev == "low":
                deductions += 2
            elif prio is not None and prio <= 2:
                deductions += 8 if prio == 0 else 4
            elif has_warn:
                deductions += 4
            elif has_tip:
                deductions += 2
            else:
                deductions += 3
        # 问题数量惩罚（降低权重，避免多问题时分数崩塌）
        count = len(hand_issues)
        if count >= 4:
            deductions += (count - 3) * 2
        return max(50, 100 - deductions)

    @staticmethod
    def _calc_deterministic_audio(audio_result, field="pitch"):
        """确定性音频分：从 audio_stats 基线出发，逐错误扣分。

        - 基线: audio_stats.pitch_accuracy / rhythm_accuracy（无 stats 时默认 100）
        - 逐错误扣: high→-10, medium→-5, low→-2
        - 保证有错误时不会比无错误时更高（修复 paradox）
        """
        errors = audio_result.get("errors", [])
        stats = audio_result.get("stats", {})
        if field == "pitch":
            relevant = [e for e in errors if e.get("type") in
                        ("dead_note", "wrong_note", "extra_note", "missing_note")]
            baseline = stats.get("pitch_accuracy", 100)
        else:
            relevant = [e for e in errors if e.get("type") in
                        ("rhythm_fault", "transition_gap", "overlap", "pause")]
            baseline = stats.get("rhythm_accuracy", 100)
        baseline = max(40, min(100, int(baseline))) if baseline is not None else 100
        deductions = 0
        for e in relevant:
            sev = e.get("severity", "low")
            if sev == "high":
                deductions += 10
            elif sev == "medium":
                deductions += 5
            else:
                deductions += 2
        return max(0, baseline - deductions)

    def _postprocess_report(self, report, audio_result, video_data, detected_chords, reference_images, audio_diagnosis=None):
        """后处理：评分归一化 + 锚定 + 章节强制一致性"""
        score = report.get("score", {})
        hand_status = video_data.get("hand_status", "nHand")
        report_md = report.get("report_markdown") or report.get("summary", "")

        # 品位幻觉校验：对照实际数据检查报告中的品位描述
        report_md = self._validate_fret_positions(report_md, audio_result)

        # 归一化 0-10 → 0-100
        for sk in ['overall', 'pitch', 'rhythm', 'technique']:
            val = score.get(sk)
            if val is not None and isinstance(val, (int, float)) and 0 < val < 10:
                score[sk] = val * 10

        # 评分锚定：音频信号轻量地板（仅 rhythm，pitch 在自由演奏模式下无法精确判断）
        audio_stats = audio_result.get("stats", {})
        audio_pitch = audio_stats.get("pitch_accuracy")
        audio_rhythm = audio_stats.get("rhythm_accuracy")
        audio_overall = audio_stats.get("overall_score")

        # rhythm 音频地板（35%）：托底防止 AI 严重低估节奏
        logger.info(f"评分地板检查: score={score}, audio_rhythm={audio_rhythm}")
        if audio_rhythm is not None and "rhythm" in score:
            if score["rhythm"] < audio_rhythm:
                old = score["rhythm"]
                score["rhythm"] = round(audio_rhythm * 0.35 + score["rhythm"] * 0.65)
                logger.info(f"评分地板 rhythm: {old} → {score['rhythm']} (audio={audio_rhythm})")

        # 硬约束：rhythm 最终分数不得低于音频信号 15 分以上
        if audio_rhythm is not None and "rhythm" in score and score["rhythm"] is not None:
            floor_val = max(40, audio_rhythm - 15)
            if score["rhythm"] < floor_val:
                old = score["rhythm"]
                score["rhythm"] = floor_val
                logger.info(f"评分硬地板 rhythm: {old} → {score['rhythm']} (min={floor_val})")

        # 确定性评分：基于实际错误数量和严重程度计算基准分
        hand_issues = video_data.get("hand_issues", [])
        det_tech = self._calc_deterministic_technique(hand_issues)
        det_pitch = self._calc_deterministic_audio(audio_result, "pitch")
        det_rhythm_audio = self._calc_deterministic_audio(audio_result, "rhythm")
        logger.info(f"确定性计算: det_tech={det_tech} det_pitch={det_pitch} det_rhythm={det_rhythm_audio} "
                    f"hand_issues_count={len(hand_issues)} "
                    f"severities={[h.get('_ai_severity','?') for h in hand_issues[:10]]} "
                    f"ai_score={score}")

        # technique: 确定性分 ±5 锚定 AI 输出
        if "technique" in score and score["technique"] is not None:
            ai_tech = score["technique"]
            lo, hi = max(0, det_tech - 5), min(100, det_tech + 5)
            if ai_tech < lo:
                logger.info(f"评分锚定 technique: AI={ai_tech} → {lo} (det={det_tech}, range=[{lo},{hi}])")
                score["technique"] = lo
            elif ai_tech > hi:
                logger.info(f"评分锚定 technique: AI={ai_tech} → {hi} (det={det_tech}, range=[{lo},{hi}])")
                score["technique"] = hi

        # pitch: 确定性分 ±5 锚定
        if "pitch" in score and score["pitch"] is not None:
            ai_pitch = score["pitch"]
            lo, hi = max(0, det_pitch - 5), min(100, det_pitch + 5)
            if ai_pitch < lo:
                logger.info(f"评分锚定 pitch: AI={ai_pitch} → {lo} (det={det_pitch})")
                score["pitch"] = lo
            elif ai_pitch > hi:
                logger.info(f"评分锚定 pitch: AI={ai_pitch} → {hi} (det={det_pitch})")
                score["pitch"] = hi

        # rhythm: 保留现有音频地板 + 新增确定性锚定
        if "rhythm" in score and score["rhythm"] is not None:
            ai_rhythm = score["rhythm"]
            lo, hi = max(0, det_rhythm_audio - 5), min(100, det_rhythm_audio + 5)
            if ai_rhythm < lo:
                logger.info(f"评分锚定 rhythm: AI={ai_rhythm} → {lo} (det={det_rhythm_audio})")
                score["rhythm"] = lo
            elif ai_rhythm > hi:
                logger.info(f"评分锚定 rhythm: AI={ai_rhythm} → {hi} (det={det_rhythm_audio})")
                score["rhythm"] = hi

        # 总分 = 加权分量（透明可验证）
        # 自由演奏模式：音频信号(pitch/rhythm)比手型(technique)更可靠
        components = [score.get(k) for k in ['pitch', 'rhythm', 'technique']]
        if all(v is not None for v in components):
            score["overall"] = round(score["pitch"] * 0.25 + score["rhythm"] * 0.45 + score["technique"] * 0.30)
            logger.info(f"总分计算: {score['pitch']}×0.25 + {score['rhythm']}×0.45 + {score['technique']}×0.30 = {score['overall']}")

        # 音频状态判定
        audio_errs = audio_result.get("errors", [])
        audio_has_real = any(
            e.get("type") not in ("audio_extraction_failed", "no_notes_detected")
            for e in audio_errs
        )
        if not audio_errs:
            audio_status = "ok"
        elif audio_has_real:
            audio_status = "issues"
        else:
            audio_status = "nSound"

        # 提取 AI 在报告中写到的音频观察（在 force_audio_section 替换之前）
        ai_audio_obs = self._extract_ai_audio_observations(report_md, audio_errs)

        # 章节强制一致性
        report_md = force_audio_section(report_md, audio_errs)
        if audio_status == "nSound":
            report_md = force_audio_section_nosound(report_md)

        # 检查 AI 是否在报告中发现了手型问题（不被 force_hand_section 误覆盖）
        tips = report.get("practice_tips", [])
        tips_text = " ".join(tips) if tips else ""
        ai_hand_keywords = ["立起来", "趴", "塌", "太直", "太弯", "太竖直", "紧张", "放松指根",
                           "指腹按弦", "手指较平", "PIP", "MCP", "手腕往里扣", "手腕太直", "拇指捏"]
        ai_found_hand_issues = any(kw in report_md or kw in tips_text for kw in ai_hand_keywords)

        if hand_status == "nHand":
            score["technique"] = None
            # 无手型数据时：pitch 33% + rhythm 67%（归一化后权重）
            if score.get("pitch") is not None and score.get("rhythm") is not None:
                score["overall"] = round(score["pitch"] * 0.33 + score["rhythm"] * 0.67)
            else:
                score["overall"] = None
            report_md = force_hand_section(report_md, "nHand")
        elif hand_status == "ok":
            technique_score = score.get("technique")
            if ai_found_hand_issues:
                # AI 看到了引擎没检测到的问题 → 保留 AI 的手型分析
                hand_status = "issues"
                report_md = force_hand_section(report_md, "ai_found")
                # 在 hand_issues 列表中追加 AI 的观察，让前端检测详情也有显示
                hand_issues = video_data.get("hand_issues", [])
                hand_issues.append({
                    "哪只手": "左手", "问题": "💡 AI 观察到手型细节可优化（详见下方报告手型章节）",
                    "时间": 0, "置信度": 0.5,
                })
                video_data["hand_issues"] = hand_issues
            else:
                report_md = force_hand_section(report_md, "ok", technique_score)

        # 清理残留占位符（AI 有时仍输出 [品位待核实] 等）
        from services.report_postprocessor import fix_placeholders
        report_md = fix_placeholders(report_md, detected_chords, audio_result.get("notes", []))

        # 输出端最后一道防线：拦截弦/品引用（替代分散在各处的源头过滤）
        from pipeline.sanitizer import sanitize_report
        report_md = sanitize_report(report_md)

        # 将 AI 音频观察合并到错误列表（供前端检测详情展示）
        display_audio_errs = list(audio_errs)
        if ai_audio_obs:
            # AI 观察放在 basic-pitch 错误之后
            display_audio_errs = list(audio_errs[:3]) + ai_audio_obs
            # 如果原本没有 basic-pitch 错误但 AI 观察到了问题，修正状态
            if not audio_errs and audio_status == "ok":
                audio_status = "issues"

        # 手部运动修正：利用手腕摆动幅度区分扫弦/柱式
        onset_clusters = self._refine_clusters_with_hand_motion(
            audio_result.get("onset_clusters", []),
            video_data.get("hand_landmarks", []),
            right_hand=video_data.get("right_hand_analysis", {})
        )

        return {
            "score": score,
            "audio_errors": display_audio_errs[:5],
            "audio_status": audio_status,
            "hand_issues": video_data.get("hand_issues", []),
            "hand_status": hand_status,
            "snapshots": video_data.get("snapshots", []),
            "issue_context": video_data.get("issue_context", []),
            "report_markdown": report_md,
            "practice_tips": report.get("practice_tips", []),
            "summary": report.get("summary", ""),
            "audio_stats": audio_result.get("stats", {}),
            "reference_images": [{k: v for k, v in r.items() if not k.startswith("_")} for r in reference_images],
            "detected_chords": detected_chords,
            # Phase 1 新增字段：直接传递 audio_result 中的数据
            "notes": audio_result.get("notes", []),
            "phantom_notes": audio_result.get("phantom_notes", []),
            "technique_segments": audio_result.get("technique_segments", []),
            "note_techniques": audio_result.get("note_techniques", []),
            "onset_clusters": onset_clusters,  # 经手部运动修正的匹配法分类结果
            "strumming_patterns": audio_result.get("strumming_patterns", []),  # 向后兼容
            "strumming_quality": audio_result.get("strumming_quality"),
            "playing_onset": audio_result.get("playing_onset"),
            "beat_info": audio_result.get("beat_info", {}),
            "detected_capo": audio_result.get("detected_capo", 0),
            "content_type": audio_result.get("content_type", "chord"),
            "recording_quality": audio_result.get("recording_quality", {}),
        }
