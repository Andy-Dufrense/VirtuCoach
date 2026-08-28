"""
和弦/技巧手型检查服务。

从 main.py 的 check_chord() 提取，原来 440 行单体函数。
"""

import os
import json
import re
import uuid
from typing import Optional

from pipeline import FilterContext, create_chord_check_pipeline, dedup_by_content, dedup_by_time
from db.reference_db import TECHNIQUE_OPTIONS
from logging_config import get_logger

logger = get_logger(__name__)

# 右手技巧 ID 集合
RIGHT_HAND_TECHNIQUES = {
    "basic-picking", "strumming", "chop", "pm-technique", "am-technique",
    "rasgueado", "rest-stroke", "right-hand-tapping", "right-hand-muting",
    "slap-harmonic", "artificial-harmonics", "arpeggio", "string-slap",
}


class HandCheckService:
    """和弦和技巧手型检查服务"""

    def __init__(self, chord_analyzer, video_processor, vision_analyzer,
                 deepseek_agent, reference_db, upload_dir):
        self.chord_analyzer = chord_analyzer
        self.video_processor = video_processor
        self.vision_analyzer = vision_analyzer
        self.deepseek_agent = deepseek_agent
        self.reference_db = reference_db
        self.upload_dir = upload_dir

    def check(self, video_path: str, chord: str = "", instrument: str = "guitar",
              mode: str = "chord", technique: str = "") -> dict:
        """执行和弦或技巧手型检查"""
        is_technique_mode = (mode == "technique" and technique)
        chord_data = {}
        technique_data = {}
        technique_name = ""

        # 1. 加载知识
        if is_technique_mode:
            technique_data = self.chord_analyzer.load_technique_knowledge(technique)
            technique_name = technique_data.get("name", technique)
            has_knowledge = technique_data.get("has_knowledge", False)
            string_results = []
            audio_result = type('obj', (object,), {'string_results': []})()
        else:
            chord_data = self.chord_analyzer.load_chord_knowledge(chord)
            if not chord_data:
                chord_data = self.chord_analyzer.make_generic_chord_data(chord)
            has_knowledge = chord_data.get("has_knowledge", False)
            technique_name = chord
            audio_result = self.chord_analyzer.analyze_per_string(video_path, chord_data)
            string_results = audio_result.string_results

        # 2. 构建过滤上下文
        is_right_hand_tech = (
            is_technique_mode and technique in TECHNIQUE_OPTIONS
            and TECHNIQUE_OPTIONS[technique] in RIGHT_HAND_TECHNIQUES
        )
        technique_allowed_kw = self._load_technique_allowed_keywords(
            technique_data if is_technique_mode else {}, is_technique_mode, is_right_hand_tech
        )
        technique_hints = (
            [TECHNIQUE_OPTIONS[technique]] if is_technique_mode and technique in TECHNIQUE_OPTIONS else None
        )

        filter_ctx = FilterContext(
            mode="technique" if is_technique_mode else "chord",
            technique_id=technique,
            is_right_hand_tech=is_right_hand_tech,
            technique_allowed_keywords=technique_allowed_kw,
            force_hand="right" if is_right_hand_tech else "left",
        )

        # 3. 手型检测
        hand_issues, hands_detected, user_image_url = self._detect_hand_issues(
            video_path, instrument, technique_hints, filter_ctx,
            chord=chord if not is_technique_mode else None,
            chord_data=chord_data if not is_technique_mode else None,
        )

        # 4. 匹配参考图
        reference_images, ref_image_paths = self._match_references(
            is_technique_mode, technique, chord, hand_issues
        )

        # 5. Vision AI 分析
        hand_issues = self._run_vision_if_needed(
            hand_issues, user_image_url, instrument,
            ref_image_paths, technique_data, chord_data, is_technique_mode,
            is_right_hand_tech, technique_name
        )

        # 6. 去重
        hand_issues = self._final_dedup(hand_issues)

        # 7. 未检测到手 → 直接返回
        if not hands_detected and not hand_issues:
            return {
                "chord_name": technique_name,
                "chord_id": technique if is_technique_mode else chord,
                "has_knowledge": has_knowledge,
                "hands_detected": False,
                "overall_score": None,
                "string_results": string_results,
                "issues": [{
                    "body_part": "手部", "severity": "severe",
                    "description": "未检测到手部画面，请确保手部在镜头内清晰可见，光线充足",
                    "suggestion": "调整摄像头角度，确保按弦手和琴颈完整出现在画面中",
                }],
                "summary": "未检测到手部画面，无法评估手型。请调整拍摄角度后重新录制。",
                "user_image_url": user_image_url,
                "reference_images": reference_images,
                "mode": mode, "audio_quality": "unknown", "quality_note": "",
            }

        # 8. 音频质量检查 + 交叉验证
        quality_note = self._cross_validate(string_results, hand_issues, audio_result, is_technique_mode, chord_data)

        # 9. AI 综合评估
        try:
            evaluation = self.deepseek_agent.evaluate_chord(
                chord_data=technique_data if is_technique_mode else chord_data,
                string_results=string_results,
                hand_issues=hand_issues,
                instrument=instrument,
                mode=mode,
            )
        except Exception as e:
            logger.warning(f"AI评估失败: {e}")
            evaluation = {
                "overall_score": 5,
                "issues": hand_issues[:3] if hand_issues else [],
                "summary": f"手型检测完成（AI评估暂不可用）。共检测到 {len(hand_issues)} 个问题。",
            }

        # 把规则引擎的置信度/出现次数/诊断ID 回填到 AI 重写的问题上
        # （AI 可能改写描述，按 body_part + 描述相似度匹配回填）
        eval_issues = evaluation.get("issues", [])
        if eval_issues and hand_issues:
            eval_issues = self._attach_issue_meta(eval_issues, hand_issues)
        evaluation["issues"] = eval_issues

        audio_quality = getattr(audio_result, "audio_quality", "unknown") if not is_technique_mode else "unknown"

        return {
            "chord_name": technique_name,
            "chord_id": technique if is_technique_mode else chord,
            "has_knowledge": has_knowledge,
            "hands_detected": hands_detected,
            "overall_score": evaluation.get("overall_score", 0),
            "string_results": string_results,
            "issues": evaluation.get("issues", []),
            "summary": evaluation.get("summary", ""),
            "practice_tips": evaluation.get("practice_tips", []),
            "user_image_url": user_image_url,
            "reference_images": reference_images,
            "mode": mode, "audio_quality": audio_quality, "quality_note": quality_note,
        }

    def _attach_issue_meta(self, ai_issues, hand_issues):
        """把规则引擎的置信度/出现次数/诊断ID 回填到 AI 重写的问题上"""
        for ai in ai_issues:
            meta = self._find_issue_meta(ai, hand_issues)
            if not meta:
                continue
            if "置信度" not in ai:
                ai["置信度"] = meta.get("置信度")
            if "出现次数" not in ai:
                ai["出现次数"] = meta.get("出现次数")
            if "诊断ID" not in ai:
                ai["诊断ID"] = meta.get("诊断ID")
        return ai_issues

    @staticmethod
    def _find_issue_meta(ai_issue, hand_issues):
        """按 body_part + 描述相似度（difflib，容忍 AI 改写插词）找对应的规则引擎问题"""
        from difflib import SequenceMatcher
        part = ai_issue.get("body_part", "")
        desc = HandCheckService._norm_desc(ai_issue.get("description", ""))
        if not part or not desc:
            return None
        best, best_ratio = None, 0.0
        for h in hand_issues:
            if h.get("body_part") != part:
                continue
            h_desc = HandCheckService._norm_desc(h.get("description", ""))
            if not h_desc:
                continue
            ratio = SequenceMatcher(None, desc, h_desc).ratio()
            if ratio > best_ratio:
                best_ratio, best = ratio, h
        return best if best_ratio >= 0.4 else None

    @staticmethod
    def _norm_desc(s):
        """归一化问题描述：去掉「（疑似）」和 ⚠️/💡 前缀"""
        s = s.replace("（疑似）", "")
        return re.sub(r"^[⚠️💡]+\s*", "", s).strip()

    def _load_technique_allowed_keywords(self, technique_data, is_technique_mode, is_right_hand_tech):
        """从技巧知识库加载豁免关键词。增加同义变体和反向关键词过滤。"""
        allowed = set()
        if not is_technique_mode:
            return allowed

        body = technique_data.get("body", "")
        has_knowledge = technique_data.get("has_knowledge", False)
        if not has_knowledge:
            return allowed

        # 基础关键词 + 同义变体
        base_keywords = [
            "手指平", "手指直", "手指近乎平行", "关节角度大", "手指弯曲蜷缩",
            "手腕角度偏大", "手指较平", "瞬时角度变化", "手指快速伸展",
            "食指伸直", "手指伸直", "接近伸直", "手指放平",
            "手指自然平放", "食指放平", "四指平放", "手指不必立",
            "手指趴", "食指趴", "手指没立",
        ]
        # 反向信号：原文中含这些词的段落不应提取豁免关键词
        negation_signals = ["不应平放", "不能放平", "不可放平", "错误", "不对"]

        def _is_negated(text: str) -> bool:
            return any(sig in text for sig in negation_signals)

        # 解析 markdown 表格中「正常表现」列
        for line in body.split("\n"):
            if "不应算问题" in line or "正常表现" in line:
                continue
            if _is_negated(line):
                continue
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 3:
                normal_desc = parts[-1]
                for kw in base_keywords:
                    if kw in normal_desc:
                        allowed.add(kw)

        # 解析「铁律」和「AI检测豁免规则」章节
        in_exemption = False
        for line in body.split("\n"):
            ls = line.strip()
            if "AI检测豁免规则" in ls or "AI 检测豁免规则" in ls or "铁律" in ls:
                in_exemption = True
                continue
            if in_exemption and ls.startswith("##"):
                in_exemption = False
                continue
            if in_exemption and ls and not _is_negated(ls):
                for kw in ["手指平放", "手指直", "手指伸直", "食指伸直",
                           "手指接近伸直", "手指近乎平行", "右手在指板",
                           "轻触", "手指较平", "手指没立", "手指自然平放",
                           "食指放平", "四指平放", "手指不必立"]:
                    if kw in ls:
                        allowed.add(kw)

        # 右手技巧通用豁免
        if is_right_hand_tech:
            allowed.update([
                "手指较平", "手指平", "手指放平", "食指伸直",
                "手指伸直", "接近伸直", "手指没立", "食指平",
                "手指趴", "食指趴", "手指自然平放", "食指放平",
                "四指平放", "手指不必立",
            ])

        if allowed:
            logger.info(f"loaded {len(allowed)} technique exemption keywords: {sorted(allowed)}")
        return allowed

    def _detect_hand_issues(self, video_path, instrument, technique_hints, filter_ctx,
                             chord=None, chord_data=None):
        """MediaPipe 手型检测 + 管道过滤"""
        try:
            # 构建和弦上下文，传给 video_processor 以启用 KB 驱动检测（横按豁免等）
            detected_chords = None
            if chord and chord_data:
                detected_chords = [{
                    "time": 0,
                    "end_time": 999,
                    "chord_id": chord,
                    "chord_name": chord_data.get("display_name", chord),
                    "confidence": 1.0,
                }]
            video_data = self.video_processor.process(
                video_path, [], instrument,
                technique_hints=technique_hints,
                force_hand=filter_ctx.force_hand,
                detected_chords=detected_chords,
            )
        except Exception as e:
            logger.warning(f"手型检测失败: {e}")
            import traceback
            traceback.print_exc()
            return [], False, ""

        raw_issues = video_data.get("hand_issues", [])

        # 统一过滤管道
        hand_filter = create_chord_check_pipeline()
        filtered = hand_filter.apply(raw_issues, filter_ctx)

        # 提取截图
        snapshots = video_data.get("snapshots", [])
        user_image_url = ""

        # 检测是否真的有手部
        hands_detected = False
        try:
            analysis_parsed = json.loads(video_data.get("frame_analysis_data", "{}"))
            hands_detected = analysis_parsed.get("hands_found", False)
        except:
            hands_detected = len(video_data.get("hand_landmarks", [])) > 0
        # Fallback: check snapshots as secondary indicator
        if not hands_detected:
            if video_data.get("snapshots"):
                hands_detected = True

        # 转换格式，保留 ⚠️/💡 前缀的严重程度信息，并回填置信度/出现次数做门控
        hand_issues = []
        for h in filtered:
            who = h.get("哪只手", h.get("handedness", ""))
            iss = h.get("问题", "")
            ts_val = h.get("时间", h.get("timestamp", 0))
            if iss and "正常" not in iss:
                # 根据前缀判断严重程度（与主流程 video_processor 阈一致）
                if "⚠️" in iss:
                    severity = "severe"
                elif "💡" in iss:
                    severity = "mild"
                else:
                    severity = "moderate"
                # 去掉 ⚠️/💡 前缀，避免「疑似⚠️」拼接
                desc = re.sub(r"^[⚠️💡]+\s*", "", iss).strip()
                conf = h.get("置信度", h.get("handedness_confidence", 0.0))
                times = h.get("所有检测时间点") or []
                occ = len(times) if times else int(h.get("出现次数", h.get("来源数", 1)) or 1)
                # 置信度门控：MediaPipe 手性判断置信度低 → 检测证据不足，降级并标「疑似」
                # （0.6 与 video_processor 内部"低于 0.6 不信任手性判断"的阈值一致）
                if isinstance(conf, (int, float)) and conf < 0.6:
                    if severity == "severe":
                        severity = "moderate"
                    elif severity == "moderate":
                        severity = "mild"
                    desc = "（疑似）" + desc
                hand_issues.append({
                    "body_part": who, "hand_side": who,
                    "description": desc, "severity": severity,
                    "timestamp": float(ts_val) if ts_val is not None else 0,
                    "置信度": conf, "出现次数": occ,
                    "诊断ID": h.get("诊断ID"),
                })

        # 单和弦手型检查只报明确问题（⚠️严重）。💡轻微角度提示（手腕太直/有点太直/偏竖直/太紧张）
        # 在正确和弦上也大量触发，且会被 pipeline 的 _summarize_problems 归纳成无前缀的
        # 「手指：xxx站得太直」中等严重度条目，同样是与「无错误」冲突的噪声。故只保留 severe。
        # 注意：浮空指（下方追加）与音频闷音是独立问题，不受此过滤影响。
        hand_issues = [h for h in hand_issues if h.get("severity") == "severe"]

        # 浮空指检测：KB 期望按弦的手指，若被 VL 信号判定为浮空（伸得太直、未按实），
        # 则「太竖直/太直/太弯」是错误诊断——浮空=没按弦，报「戳弦太竖直」会误导。
        # 改为：抑制该指的误报，并报「浮空未按弦」。
        if chord_data:
            _en2cn = {"index": "食指", "middle": "中指", "ring": "无名指", "pinky": "小指"}
            _vl = video_data.get("vl_signals", {}) or {}
            _floating_en = _vl.get("floating", []) or []
            if _floating_en:
                _expected = set()
                for _fn, _fi in (chord_data.get("fingers", {}) or {}).items():
                    if isinstance(_fi, dict) and _fi.get("string") is not None:
                        _expected.add(_fn)
                _floating_cn = [_en2cn.get(f, f) for f in _floating_en]
                _angle_kw = ("太竖直", "太直", "太弯")
                _kept = []
                for _hi in hand_issues:
                    _desc = _hi.get("description", "")
                    _drop = any(
                        _cn in _expected and _cn in _desc and any(k in _desc for k in _angle_kw)
                        for _cn in _floating_cn
                    )
                    if not _drop:
                        _kept.append(_hi)
                for _cn in _floating_cn:
                    if _cn in _expected:
                        _kept.append({
                            "body_part": "左手", "hand_side": "左手",
                            "description": f"{_cn}浮空未按弦（该和弦通常用{_cn}按弦）",
                            "severity": "mild",
                            "timestamp": 0,
                            "置信度": _vl.get("floating_conf", 0.0),
                            "出现次数": 1,
                        })
                hand_issues = _kept

        # 匹配截图 URL（带去重：视觉相似的截图不重复出现）
        import os as _os
        from services.analysis_service import _image_dhash, _hamming_distance

        UPLOAD_DIR = _os.path.join(_os.path.dirname(__file__), "..", "snapshots")
        snap_hashes = {}
        for s in snapshots:
            fname = _os.path.basename(s["url"])
            fpath = _os.path.join(UPLOAD_DIR, fname)
            if _os.path.exists(fpath):
                try:
                    snap_hashes[round(s.get("time", 0), 2)] = _image_dhash(fpath)
                except Exception:
                    snap_hashes[round(s.get("time", 0), 2)] = ""

        used_hashes = set()
        for h in hand_issues:
            t = h.get("timestamp", 0)
            best_url = ""
            best_hash = ""
            best_dist = 999
            for s in snapshots:
                st = round(s.get("time", 0), 1)
                dist = abs(st - t)
                if dist < best_dist and dist < 1.0:
                    best_dist = dist
                    best_url = s.get("url", "")
                    best_hash = snap_hashes.get(st, "")

            # 视觉去重：该hash已经给前面的issue用过，且距离≤15 → 仍分配URL（不同issue可以共享同一截图）
            if best_hash:
                dup = False
                for uh in used_hashes:
                    if best_hash == uh or _hamming_distance(best_hash, uh) <= 15:
                        dup = True
                        break
                if not dup:
                    used_hashes.add(best_hash)
                # 无论是否去重命中，都分配截图URL，确保每个issue都有视觉参考

            h["snapshot_url"] = best_url
            if best_url and not user_image_url:
                user_image_url = best_url

        # 没有匹配问题时，取中间帧截图
        if not user_image_url and snapshots:
            mid = len(snapshots) // 2
            user_image_url = snapshots[mid].get("url", "")

        return hand_issues, hands_detected, user_image_url

    def _match_references(self, is_technique_mode, technique, chord, hand_issues):
        """匹配正确手型参考图"""
        reference_images = []
        ref_image_paths = []

        if is_technique_mode:
            search_technique = technique
            if search_technique in TECHNIQUE_OPTIONS:
                search_technique = TECHNIQUE_OPTIONS[search_technique]
            if hand_issues:
                matched = self.reference_db.match_from_issues(hand_issues)
                if len(matched) < 2:
                    tech_refs = self.reference_db.search(technique=search_technique)
                    seen = {r["id"] for r in matched}
                    for r in tech_refs:
                        if r["id"] not in seen:
                            seen.add(r["id"])
                            matched.append(r)
                        if len(matched) >= 3:
                            break
            else:
                matched = self.reference_db.search(technique=search_technique)
        else:
            all_refs = self.reference_db.list_all()
            all_basic = [r for r in all_refs if r.get("technique", "") == "basic-fretting"]
            chord_tag = f"chord-{chord}"
            chord_specific = [r for r in all_basic if chord_tag in set(r.get("tags", []))]
            generic = [r for r in all_basic if r not in chord_specific]
            has_chord_specific = len(chord_specific) > 0
            if not has_chord_specific:
                logger.warning(f"和弦 {chord} 无专属参考图，使用通用手型图")
            # 和弦指定图片优先按创建时间排序
            chord_specific.sort(key=lambda r: r.get("created_at", ""), reverse=True)
            matched = list(chord_specific)
            seen = {r["id"] for r in matched}
            if hand_issues:
                for r in self.reference_db.match_from_issues(hand_issues):
                    # 和弦模式只接受 basic-fretting 参考图，排除横按等不相关技巧
                    if r.get("technique", "") != "basic-fretting":
                        continue
                    if r["id"] not in seen:
                        seen.add(r["id"])
                        matched.append(r)
                    if len(matched) >= 5:
                        break
            if len(matched) < 3:
                for r in generic:
                    if r["id"] not in seen:
                        seen.add(r["id"])
                        matched.append(r)
                    if len(matched) >= 3:
                        break
            if not matched:
                matched = self.reference_db.search(body_part="full_hand")

        for i, ref in enumerate(matched[:5]):
            ref_path = os.path.join(self.reference_db.images_dir, ref["image_path"])
            if os.path.exists(ref_path) and i < 2:
                ref_image_paths.append(ref_path)
            ref_tags = ref.get("tags", [])
            ref_chord_id = f"chord-{chord}" if chord else ""
            reference_images.append({
                "id": ref["id"], "description": ref["description"],
                "body_part": ref["body_part"], "hand": ref["hand"],
                "technique": ref["technique"], "tags": ref["tags"],
                "image_url": self.reference_db.image_url(ref),
                "chord_name": f"{chord}和弦" if chord else "",
                "chord_matched": bool(ref_chord_id and isinstance(ref_tags, list) and ref_chord_id in ref_tags),
                "is_topdown": "俯视图" in str(ref_tags),
                "is_front": "正视图" in str(ref_tags),
            })

        return reference_images, ref_image_paths

    def _run_vision_if_needed(self, hand_issues, user_image_url, instrument,
                              ref_image_paths, technique_data, chord_data,
                              is_technique_mode, is_right_hand_tech, technique_name):
        """执行 Vision AI 分析（如适用）"""
        if is_right_hand_tech:
            logger.info("vision skipped: right-hand technique")
            return hand_issues
        if not user_image_url or not self.vision_analyzer.available:
            return hand_issues

        # 从 user_image_url 提取文件路径（snapshot 在 snapshots/ 目录，非 uploads/）
        fname = os.path.basename(user_image_url)
        snap_dir = os.path.join(os.path.dirname(__file__), "..", "snapshots")
        fpath = os.path.join(snap_dir, fname)
        snap_paths = [fpath] if os.path.exists(fpath) else []

        if not snap_paths:
            return hand_issues

        # 构建知识上下文
        if is_technique_mode:
            chord_context = [{"technique_name": technique_data.get("name", technique_name),
                              "body": technique_data.get("body", "")[:1000]}]
        else:
            chord_context = [{"chord_name": chord_data.get("chord_name", ""),
                              "fingers": chord_data.get("fingers", {}),
                              "thumb_position": chord_data.get("thumb_position", {}),
                              "body": chord_data.get("body", "")[:1000]}]

        vision_result = self.vision_analyzer.analyze_frames(
            snap_paths, instrument,
            reference_paths=ref_image_paths if ref_image_paths else None,
            mediapipe_context=chord_context,
            check_target=technique_name,
        )

        if vision_result.get("vision_issues"):
            vision_hand = vision_result.get("handedness", "")
            for vi in vision_result["vision_issues"]:
                hand_issues.append({
                    "body_part": vi.get("body_part", "手部"),
                    "hand_side": vi.get("handedness", vision_hand),
                    "description": vi.get("description", ""),
                    "severity": vi.get("severity", "moderate"),
                    "suggestion": vi.get("suggestion", ""),
                })

        return hand_issues

    def _final_dedup(self, hand_issues):
        """最终去重：仅内容去重，保留所有不同时间点的问题（pipeline 已做时间合并）"""
        seen = set()
        deduped = []
        for hi in hand_issues:
            hand = hi.get("hand_side", hi.get("哪只手", ""))
            desc = hi.get("description", "")
            key = f"{hand}|{desc[:80]}"
            if key not in seen:
                seen.add(key)
                deduped.append(hi)
        return deduped

    def _cross_validate(self, string_results, hand_issues, audio_result, is_technique_mode, chord_data):
        """手型⇔音色交叉验证"""
        quality_note = ""
        audio_quality = getattr(audio_result, "audio_quality", "unknown") if not is_technique_mode else "unknown"

        if audio_quality == "poor":
            quality_note = "音频信号很弱（可能是录音音量过低或麦克风距离太远），逐弦音色分析已跳过，结果仅供参考手型。"
            for sr in string_results:
                sr["status_text"] = "信号不足以判断"
        elif audio_quality == "weak":
            quality_note = "音频信号偏弱，逐弦音色分析可能不够准确，建议靠近麦克风或提高录音音量后重新录制。"

        if not string_results or not hand_issues or is_technique_mode:
            return quality_note

        # 构建 手指→弦 映射
        finger_to_string = {}
        fingers = chord_data.get("fingers", {})
        for finger_name, finger_info in fingers.items():
            if isinstance(finger_info, dict):
                s = finger_info.get("string")
                if s is not None:
                    try:
                        finger_to_string[finger_name] = int(s)
                    except (ValueError, TypeError):
                        pass

        # 收集手型问题的身体部位
        hand_problem_parts = set()
        for hi in hand_issues:
            desc = hi.get("description", "") + hi.get("问题", "")
            for kw in ["食指", "中指", "无名指", "小指", "拇指"]:
                if kw in desc:
                    hand_problem_parts.add(kw)

        audio_issue_strings = [s for s in string_results if not s.get("ok", True) and s.get("has_signal", False)]
        audio_only = []
        for sr in audio_issue_strings:
            string_num = sr.get("string_num", 0)
            finger_for_string = None
            for fname, fstring in finger_to_string.items():
                if fstring == string_num:
                    finger_for_string = fname
                    break
            if finger_for_string and finger_for_string not in hand_problem_parts:
                audio_only.append(sr.get("string_name", f"弦{string_num}"))

        if audio_only and audio_quality != "poor":
            if quality_note:
                quality_note += " "
            quality_note += f"{'、'.join(audio_only)}的音色异常但未检测到对应手指的手型问题，可能是收音或拨弦力度导致的，建议靠近麦克风确认。"

        # 闷音弦提示：flatness>0.30 的弦 = 手指没按实（比「音色异常」更具体）
        muted = []
        for sr in string_results:
            if sr.get("has_signal") and (sr.get("flatness", 0) or 0) > 0.30:
                sn = sr.get("string_num", 0)
                fn = None
                for fname, fstring in finger_to_string.items():
                    if fstring == sn:
                        fn = fname
                        break
                name = sr.get("string_name", f"弦{sn}")
                muted.append(f"{name}（{fn}可能没按实）" if fn else name)
        if muted and audio_quality != "poor":
            if quality_note:
                quality_note += " "
            quality_note += f"{'、'.join(muted)}检测到闷音，很可能是对应手指没按实。"

        return quality_note
