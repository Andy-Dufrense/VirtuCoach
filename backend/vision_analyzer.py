"""
VirtuCoach - AI 视觉手型分析模块
支持多提供商：通义千问 VL（国内首选）/ Claude Vision / 自动降级 MediaPipe
"""

import os
import json
import base64
import traceback
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

import cv2
import numpy as np
import httpx

from logging_config import get_logger

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

logger = get_logger(__name__)


class VisionAnalyzer:
    """AI 视觉手型/姿势分析器，支持多提供商"""
    _kdb = None

    @classmethod
    def _get_kdb(cls):
        if cls._kdb is None:
            try:
                from db.knowledge_db import knowledge_db
                cls._kdb = knowledge_db
            except Exception:
                cls._kdb = False
        return cls._kdb if cls._kdb is not False else None

    @classmethod
    def _get_rag_context(cls, query: str, top_k: int = 3) -> str:
        """从知识库检索相关教学知识，返回格式化的 prompt 片段"""
        kdb = cls._get_kdb()
        if not kdb:
            return ""
        try:
            entries = kdb.search(query, top_k=top_k)
            if entries:
                return kdb.format_for_prompt(entries)
        except Exception as e:
            logger.warning(f"RAG 检索失败: {e}")
        return ""

    def __init__(self):
        self.max_frames = 5  # 智能帧选择后最多分析5帧（原2帧盲选）
        self.available = False
        self.provider = None  # "qwen" | "claude"
        self.api_enabled = True  # 启用视觉 API 调用

        # 优先级：通义千问 VL > Claude Vision
        dashscope_key = os.getenv("DASHSCOPE_API_KEY", "")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")

        if dashscope_key and dashscope_key != "your-dashscope-api-key-here":
            self._init_qwen(dashscope_key)
        elif anthropic_key and anthropic_key != "your-anthropic-api-key-here":
            self._init_claude(anthropic_key)

        if not self.available:
            logger.info("未配置视觉 API Key，将使用 MediaPipe 规则引擎")

    def _init_qwen(self, api_key: str):
        """初始化通义千问 VL（DashScope OpenAI 兼容接口）"""
        try:
            from openai import OpenAI
            self.qwen_timeout = httpx.Timeout(120.0, connect=30.0, read=120.0, write=30.0, pool=10.0)
            self.qwen_tab_timeout = httpx.Timeout(300.0, connect=30.0, read=300.0, write=60.0, pool=10.0)
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                timeout=self.qwen_timeout,
                max_retries=0,
            )
            self.model = os.getenv("VISION_MODEL", "qwen-vl-max")
            self.provider = "qwen"
            self.available = True
            self.api_enabled = True
            logger.info(f"通义千问 VL ({self.model}) 就绪 (timeout: connect=30s, read=120s, tab_read=300s)")
        except Exception as e:
            logger.warning(f"通义千问初始化失败: {e}")

    def _init_claude(self, api_key: str):
        """初始化 Claude Vision"""
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = os.getenv("VISION_MODEL", "claude-sonnet-4-6")
            self.provider = "claude"
            self.available = True
            logger.info(f"Claude Vision ({self.model}) 就绪")
        except Exception as e:
            logger.warning(f"Claude 初始化失败: {e}")

    def _compress_image_b64(self, image_path: str, max_width: int = 800, fmt: str = ".jpg") -> Tuple[str, str]:
        """压缩图片并返回 (base64_string, mime_type)。

        将图片缩放到 max_width 宽度。fmt='.png' 用于二值图（PNG 对黑白图压缩率极高），
        默认 '.jpg' 用于自然照片。Qwen-VL 按图片分辨率计算 token 消耗。
        """
        img = self._imread_safe(image_path)
        if img is not None:
            h, w = img.shape[:2]
            if w > max_width:
                scale = max_width / w
                img = cv2.resize(img, (max_width, int(h * scale)))
            if fmt == ".png":
                _, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                mime = "image/png"
            else:
                _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                mime = "image/jpeg"
            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            logger.info(f"[resize] {w}x{h} -> {img.shape[1]}x{img.shape[0]}, {fmt}={len(b64)//1024}KB")
            return b64, mime

        # cv2 读取失败时回退到原始文件
        with open(image_path, "rb") as f:
            raw = f.read()
        ext = os.path.splitext(image_path)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        return base64.b64encode(raw).decode("utf-8"), mime

    def analyze_frame(self, image_path: str, instrument: str = "guitar",
                       mediapipe_ctx: Optional[Dict] = None,
                       rag_context: str = "", check_target: str = "") -> Dict:
        if not self.available:
            return {}

        fname = os.path.basename(image_path)
        try:
            image_data, mime = self._compress_image_b64(image_path)
            prompt = self._build_frame_prompt(instrument, mediapipe_ctx, rag_context, check_target)

            logger.info(f"[call] analyzing with {self.provider}: {fname} ({len(image_data)//1024}KB)...")
            if self.provider == "qwen":
                text = self._call_qwen(prompt, image_data, mime)
            elif self.provider == "claude":
                text = self._call_claude(prompt, image_data, mime)
            else:
                return {}

            parsed = self._parse_vision_result(text)
            if parsed:
                logger.info(f"[OK] {fname} done: hands_detected={parsed.get('hands_detected')}, issues={len(parsed.get('issues', []))}")
            else:
                logger.warning(f"[WARN] {fname} unparseable: {text[:150] if text else '(empty)'}")
            return parsed

        except Exception as e:
            import traceback
            logger.error(f"frame analysis failed ({fname}): {e}")
            traceback.print_exc()
            return {}

    def _call_qwen(self, prompt: str, image_data: str, mime: str, extra_content: Optional[List] = None, max_tokens: int = 1024, timeout: Optional[httpx.Timeout] = None) -> str:
        """通义千问 VL — OpenAI 兼容格式，支持多图参考对比"""
        if extra_content:
            content = extra_content + [{"type": "text", "text": prompt}]
        else:
            content = [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_data}"}},
                {"type": "text", "text": prompt},
            ]
        req_timeout = timeout if timeout is not None else self.qwen_timeout
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=0.3,
                max_tokens=max_tokens,
                timeout=req_timeout,
            )
            return response.choices[0].message.content
        except httpx.ReadTimeout:
            logger.warning(f"[TIMEOUT] Qwen read timeout({req_timeout.read}s) - server processing too slow")
            raise
        except httpx.ConnectTimeout:
            logger.warning(f"[TIMEOUT] Qwen connect timeout({req_timeout.connect}s) - cannot reach DashScope")
            raise
        except httpx.TimeoutException as e:
            logger.warning(f"[TIMEOUT] Qwen timeout({req_timeout.read}s): {type(e).__name__}")
            raise
        except httpx.RemoteProtocolError as e:
            logger.error(f"[ERROR] Qwen protocol error: {e}")
            raise
        except Exception as e:
            logger.error(f"[ERROR] Qwen API call failed: {e}")
            raise

    def _call_claude(self, prompt: str, image_data: str, mime: str) -> str:
        """Claude Vision"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system="你是一位资深吉他教师，精通手型、指法、姿势分析。只返回 JSON，不要其他文字。",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": mime, "data": image_data}},
                    {"type": "text", "text": prompt},
                ]
            }],
            temperature=0.3,
        )
        return response.content[0].text

    def analyze_frame_with_references(
        self, image_path: str, reference_paths: List[str], instrument: str = "guitar",
        rag_context: str = "", check_target: str = "", mediapipe_ctx: Optional[Dict] = None,
    ) -> Dict:
        """对比参考图和用户截图，利用正确手型样本辅助判断。

        Qwen-VL 支持单条消息内多张图片，我们把参考图放前面作为「正确范例」，
        用户截图放后面，让模型做有参照的对比分析。
        """
        if not self.available:
            return {}
        if not reference_paths:
            return self.analyze_frame(image_path, instrument, rag_context=rag_context, check_target=check_target, mediapipe_ctx=mediapipe_ctx)

        try:
            frame_b64, mime = self._compress_image_b64(image_path)

            # 构建多图内容：参考图在前，用户截图在后
            content = []
            for ref_path in reference_paths[:2]:  # 最多 2 张参考图
                if not os.path.exists(ref_path):
                    continue
                ref_b64, ref_mime = self._compress_image_b64(ref_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{ref_mime};base64,{ref_b64}"},
                })
                content.append({
                    "type": "text",
                    "text": "↑ 这是正确手型的参考图",
                })

            # 用户实际演奏截图
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{frame_b64}"},
            })
            content.append({
                "type": "text",
                "text": "↑ 这是学员的实际演奏画面，请对比参考图判断手型是否正确",
            })

            prompt = self._build_comparison_prompt(instrument, rag_context, check_target, mediapipe_ctx)

            if self.provider == "qwen":
                text = self._call_qwen(prompt, frame_b64, mime, extra_content=content)
            else:
                # Claude 不支持多图 content 数组，回退到单图分析
                logger.warning("Claude does not support multi-image, refs skipped")
                text = self._call_claude(prompt, frame_b64, mime)

            return self._parse_vision_result(text)

        except Exception as e:
            logger.warning(f"参考对比分析失败: {e}，回退到单图模式")
            return self.analyze_frame(image_path, instrument, rag_context=rag_context, check_target=check_target)

    def analyze_frames(
        self, snapshot_paths: List[str], instrument: str = "guitar",
        reference_paths: Optional[List[str]] = None,
        mediapipe_context: Optional[List[Dict]] = None,
        check_target: str = "",
    ) -> Dict:
        """分析多帧截图。

        Args:
            snapshot_paths: 截图文件路径列表
            instrument: 乐器类型
            reference_paths: 参考图路径
            mediapipe_context: MediaPipe scored_frames 或 chord/technique 知识上下文
                [{timestamp, total_score, top_issues, chord_name, technique_name, body}]
        """
        if not self.available:
            logger.warning("analyze_frames skip: vision AI not available")
            return {"vision_issues": [], "vision_available": False}
        if not snapshot_paths:
            logger.warning("analyze_frames skip: no snapshots")
            return {"vision_issues": [], "vision_available": False}
        if not self.api_enabled:
            logger.info("[PAUSED] API disabled, using MediaPipe rule engine")
            return {"vision_issues": [], "vision_available": False}

        # 从 mediapipe_context 中提取知识库内容（优先直接注入技巧 body）
        rag_context = ""
        if mediapipe_context:
            direct_body = ""
            search_terms = []
            for ctx in mediapipe_context:
                if isinstance(ctx, dict):
                    cn = ctx.get("chord_name", "")
                    if cn:
                        search_terms.append(cn + " 和弦 指法")
                    tn = ctx.get("technique_name", "")
                    if tn:
                        search_terms.append(tn)
                    body = ctx.get("body", "")
                    if body:
                        # 提取 AI 检测关键词：如果 body 含「铁律」或「AI检测」段落，直接注入
                        if "铁律" in body or "AI检测" in body or "AI 检测" in body:
                            import re as _re
                            key_parts = []
                            for line in body.split("\n"):
                                if any(kw in line for kw in ["铁律", "AI检测", "AI 检测", "区分", "判断原则", "🚨", "⚠️"]):
                                    key_parts.append(line.strip())
                            if key_parts:
                                direct_body = "\n".join(key_parts[:15])
                        # 和弦知识库：提取「常见错误」部分直接注入，比RAG搜索更精准
                        elif "常见错误" in body:
                            import re as _re
                            chord_parts = []
                            in_errors = False
                            for line in body.split("\n"):
                                if "常见错误" in line or "错误" in line and "###" in line:
                                    in_errors = True
                                if in_errors:
                                    chord_parts.append(line.strip())
                                    if len(chord_parts) > 40:  # 限制长度
                                        break
                            if chord_parts:
                                direct_body = "## 和弦常见错误（必须逐条检查！）\n" + "\n".join(chord_parts)
                        search_terms.append(body[:200])
            # 直接注入技巧/和弦知识的关键规则（比 RAG 搜索更精准）
            if direct_body:
                rag_context = f"{direct_body}\n"
                logger.info(f"[RAG] direct knowledge injection ({len(direct_body)} chars)")
            elif search_terms:
                query = " ".join(search_terms)
                rag_context = self._get_rag_context(query)
                if rag_context:
                    logger.info(f"[RAG] knowledge base injected ({len(rag_context)} chars)")
                # Also extract AI detection rules from knowledge entries for Vision prompts
                kdb = self._get_kdb()
                if kdb:
                    try:
                        entries = kdb.search(query, top_k=3)
                        detection_rules = []
                        for entry in entries:
                            rules = kdb.extract_detection_rules(entry)
                            if rules:
                                detection_rules.append(rules)
                        if detection_rules:
                            rules_text = "\n\n".join(detection_rules[:2])
                            rag_context = (rag_context or "") + f"\n\n## AI检测规则（必须遵守）\n{rules_text}"
                            logger.info(f"[RAG] AI detection rules injected ({len(rules_text)} chars)")
                    except Exception:
                        pass

        # 智能帧选择：优先选 MediaPipe 高分帧
        if mediapipe_context:
            # 构建 timestamp → context 映射
            ts_to_ctx = {}
            for ctx in mediapipe_context:
                ts_to_ctx[round(ctx.get("timestamp", 0), 1)] = ctx

            # 为每张截图找对应的 MediaPipe 上下文
            scored_snaps = []
            for path in snapshot_paths:
                ts = self._extract_timestamp(path)
                ctx = ts_to_ctx.get(round(ts, 1))
                score = ctx.get("total_score", 0) if ctx else 0
                scored_snaps.append((path, score, ctx))
            scored_snaps.sort(key=lambda x: x[1], reverse=True)
            frames_to_analyze = [s[0] for s in scored_snaps[:self.max_frames] if s[1] > 0]
            if not frames_to_analyze:
                frames_to_analyze = [s[0] for s in scored_snaps[:self.max_frames]]
            logger.info(f"智能帧选择: {len(scored_snaps)}张中选{len(frames_to_analyze)}张 "
                  f"(top scores: {[s[1] for s in scored_snaps[:self.max_frames]]})")
        else:
            frames_to_analyze = snapshot_paths[:self.max_frames]

        logger.info(f"[START] analyzing {len(frames_to_analyze)} frames (provider={self.provider}, refs={len(reference_paths or [])})")
        all_issues = []
        hands_found_count = 0
        handedness_votes = []

        for path in frames_to_analyze:
            # 获取该帧的上下文：优先按时间戳匹配，兜底使用第一条（和弦/技巧知识无时间戳）
            frame_context = None
            if mediapipe_context:
                ts = self._extract_timestamp(path)
                for ctx in mediapipe_context:
                    if abs(ctx.get("timestamp", 0) - ts) < 0.5:
                        frame_context = ctx
                        break
                # 无时间戳匹配时使用第一条上下文（和弦/技巧知识上下文没有timestamp字段）
                if not frame_context:
                    frame_context = mediapipe_context[0]

            if reference_paths:
                result = self.analyze_frame_with_references(path, reference_paths, instrument, rag_context=rag_context, check_target=check_target, mediapipe_ctx=frame_context)
            else:
                result = self.analyze_frame(path, instrument, mediapipe_ctx=frame_context, rag_context=rag_context, check_target=check_target)
            if result:
                if result.get("hands_detected"):
                    hands_found_count += 1
                frame_hand = result.get("handedness", "")
                if frame_hand:
                    handedness_votes.append(frame_hand)
                for issue in result.get("issues", []):
                    issue["frame"] = os.path.basename(path)
                    if frame_hand:
                        issue["handedness"] = frame_hand
                    all_issues.append(issue)

        deduped = self._deduplicate_issues(all_issues)

        import re
        for issue in deduped:
            match = re.search(r'snap_[a-f0-9]+_(\d+\.?\d*)s', issue.get("frame", ""))
            if match:
                issue["timestamp"] = float(match.group(1))
            else:
                issue["timestamp"] = 0

        majority_hand = ""
        if handedness_votes:
            from collections import Counter
            majority_hand = Counter(handedness_votes).most_common(1)[0][0]

        result = {
            "vision_issues": deduped,
            "vision_available": True,
            "frames_analyzed": len(frames_to_analyze),
            "hands_detected_frames": hands_found_count,
            "provider": self.provider,
            "handedness": majority_hand,
        }
        logger.info(f"[DONE] {len(frames_to_analyze)} frames, {hands_found_count} with hands, {len(deduped)} issues, handedness={majority_hand}")
        return result

    # ==================== 图片手型分析（不受 api_enabled 控制） ====================

    def analyze_hand_posture_image(
        self, image_path: str, instrument: str = "guitar",
        reference_paths: Optional[List[str]] = None, check_area: str = "",
    ) -> Dict:
        """图片手型分析专用入口。用户主动触发，始终调用视觉 AI（不受 api_enabled 控制）。"""
        if not self.available:
            return {
                "success": False, "error": "视觉AI未配置",
                "issues": [], "overall_hand_score": 0,
            }

        # 从 check_area 构建 RAG 检索
        rag_context = ""
        if check_area:
            rag_context = self._get_rag_context(check_area + " 手型 按弦")

        prompt = self._build_hand_posture_prompt(instrument, check_area, rag_context) if check_area else self._build_frame_prompt(instrument, rag_context=rag_context, check_target=check_area or "")

        if reference_paths:
            result = self.analyze_frame_with_references(image_path, reference_paths, instrument, rag_context=rag_context, check_target=check_area or "")
        else:
            result = self.analyze_frame(image_path, instrument, rag_context=rag_context, check_target=check_area or "")

        return {
            "success": True,
            "hands_detected": result.get("hands_detected", False),
            "handedness": result.get("handedness", ""),
            "overall_hand_score": result.get("overall_hand_score", 0),
            "issues": result.get("issues", []),
            "provider": self.provider,
        }

    def _build_hand_posture_prompt(self, instrument: str, check_area: str, rag_context: str = "") -> str:
        """针对性 prompt：用户指定了要检查的内容（如 C和弦）"""
        instrument_text = "吉他" if instrument == "guitar" else "钢琴"
        return f"""分析这张{instrument_text}演奏画面中的手型。用户特别想检查：{check_area}。

{rag_context}
请重点检查用户指定的这个姿势/技巧，同时如果发现了其他明显的手型问题也一并报告。

返回严格 JSON 格式（不要 markdown 代码块）：
{{{{
  "hands_detected": true/false,
  "handedness": "左手/右手/双手",
  "overall_hand_score": 0-10,
  "issues": [
    {{{{
      "body_part": "具体手指或部位（如：无名指、手腕）",
      "issue_type": "问题类型（collapsed_finger/flat_finger/wrist_too_low/wrist_too_high/thumb_too_high/finger_too_curled/finger_straight）",
      "severity": "mild/moderate/severe",
      "description": "中文描述你观察到的具体问题",
      "suggestion": "中文改正建议"
    }}}}
  ]
}}}}

重要：
- 优先检查用户指定的 {check_area} 相关的手型
- 如果画面中没有清晰可见的手，hands_detected 设为 false，issues 为空数组
- 只报告确实看到的问题
- issues 最多报3个最重要的问题"""

    # ==================== 基础 prompt 构建 ====================

    def _build_frame_prompt(self, instrument: str, mediapipe_ctx: Optional[Dict] = None, rag_context: str = "", check_target: str = "") -> str:
        instrument_text = "吉他" if instrument == "guitar" else "钢琴"

        # 检查目标（和弦名或技巧名）
        target_hint = f"\n\n**学员当前在练习: {check_target}**" if check_target else ""

        # 构建上下文注入（支持和弦知识库和MediaPipe两种来源）
        context_block = ""
        if mediapipe_ctx:
            # 和弦/技巧知识库上下文（含 fingers, thumb_position, body）
            fingers = mediapipe_ctx.get("fingers", {})
            if fingers:
                finger_lines = []
                for fname, fpos in fingers.items():
                    finger_lines.append(f"  - {fname}: {fpos}")
                thumb = mediapipe_ctx.get("thumb_position", {})
                thumb_line = f"  拇指位置: {thumb}" if thumb else ""
                body_text = mediapipe_ctx.get("body", "")[:500]
                context_block = f"""
## 标准手型参考（必须对比检查！）

**{mediapipe_ctx.get('chord_name', '')}** 的正确指法:
{chr(10).join(finger_lines) if finger_lines else ''}
{thumb_line}

{body_text if body_text else ''}

**你的任务**: 逐指对比学员手型与上述标准指法。
- 如果学员某手指的按弦位置与标准指法一致 → 正常，不报问题
- 如果学员某手指明显偏离标准位置 → 必须报告该手指的具体问题
- 用大白话描述差异（如「无名指没有按在标准位置」「食指趴下去了」）
- **禁止使用任何角度数字、PIP/MCP关节名称、度数等术语**

"""
            else:
                # MediaPipe scored_frames 上下文（含 top_issues）
                top_issues = mediapipe_ctx.get("top_issues", [])
                if top_issues:
                    issue_descs = [f"  - {iss['issue']}" for iss in top_issues[:5]]
                    context_block = f"""
## MediaPipe 初步检测结果（请重点验证）

本帧经 MediaPipe 手部追踪初步检测到以下问题：

{chr(10).join(issue_descs)}

**你的任务**: 用视觉能力验证以上问题是否确实存在，并用大白话描述：
- 手指是太弯了还是太直了？用「手指趴下去了」「手指太弯了」「手指没立起来」这种话说
- 指尖接触位置是「指尖立着」还是「指腹压着」？
- 手腕是「自然弯着」还是「往里扣」或者「往外翻」？
- 如果 MediaPipe 判断错误（如某个手指实际正常），请纠正并说明
- **禁止使用任何角度数字、PIP/MCP关节名称、度数等术语**

"""
                else:
                    # 和弦/技巧知识库上下文，但fingers未解析成功——直接用body文本
                    body_text = mediapipe_ctx.get("body", "")[:800]
                    chord_name = mediapipe_ctx.get("chord_name", "")
                    technique_name = mediapipe_ctx.get("technique_name", "")
                    label = chord_name or technique_name or "当前"
                    if body_text:
                        context_block = f"""
## 标准手型参考（必须逐指对比检查！）

以下是 **{label}** 的标准指法知识，请用它来判断学员手型：

{body_text}

**你的任务**: 对比学员手型与上述参考标准。
- 学员手指位置与参考一致 → 不报问题
- 学员手指位置明显偏离参考 → 必须报告具体问题
- 用大白话描述差异（如「无名指没有立起来」「食指趴下去了碰到了邻弦」）
- **禁止使用任何角度数字、PIP/MCP关节名称、度数等术语**

"""

        return f"""分析这张{instrument_text}演奏画面中的手型和姿势。{target_hint}
{rag_context}
{context_block}

## 🚨 左右手判断铁律（必须遵守）
- **靠近琴头（有调音旋钮的那一端）的手 = 左手**
- **靠近音孔/琴桥（琴体那一端）的手 = 右手**
- 如果画面中能看到琴头和调音旋钮，靠近琴头的手100%是左手
- 如果画面中能看到音孔，靠近音孔的手100%是右手

## 🚨 特殊技巧识别
- **人工泛音**：右手食指轻触琴弦泛音点（通常在12品以上），右手无名指拨弦。右手会出现在指板上方靠近音孔的位置，这是正确的人工泛音手型，**禁止将其误判为左手按弦姿势错误**
- **点弦**：手指垂直敲击品丝，不同于普通按弦
- **推弦**：手指按住弦后推动改变音高
- 如果手型符合以上特殊技巧特征，说明这是正常技巧动作，不要报告为手型错误
- 如果你不确定画面上是什么技巧，根据手的**位置**来判断左右手：琴头侧=左手，音孔侧=右手

请检查以下项目并返回 JSON：

1. 手指弯曲度：按弦手指是否自然弯曲成弧形？有无塌指或过度弯曲？
2. 指尖站立：是否用指尖垂直按弦？还是用指腹？
3. 手腕位置：手腕是否自然放松？有无塌腕或过高？
4. 拇指位置：左手拇指是否放在琴颈后方合适位置？有无过高或过低？

## 关键原则：只报告有问题的部位
- 如果某个手指姿势正常 → 不要列出它！不要在 issues 中描述正常的手指
- 如果手型整体规范 → issues 应该是空数组 []，overall_hand_score 给 8-10
- 不要把"检查每个手指"理解成"描述每个手指"
- issues 中最多只报 2-3 个确实有问题的部位，宁少勿多
- 不确定时 → 不报，confidence 设为 low 的问题不要报

返回严格 JSON 格式（不要 markdown 代码块）：
{{{{
  "hands_detected": true/false,
  "handedness": "左手/右手/双手",
  "overall_hand_score": 0-10,
  "issues": [
    {{{{
      "body_part": "具体手指或部位（如：右手无名指、左手手腕）",
      "issue_type": "问题类型（collapsed_finger/flat_finger/wrist_too_low/wrist_too_high/thumb_too_high/finger_too_curled/finger_straight）",
      "severity": "mild/moderate/severe",
      "description": "用大白话描述问题，如'食指趴下去了没立起来'、'手腕往里扣了'，禁止角度数字和关节术语",
      "suggestion": "中文改正建议",
      "confidence": "high/medium/low"
    }}}}
  ]
}}}}

重要：
- 如果画面中没有清晰可见的手，hands_detected 设为 false，issues 为空数组
- 只报告确实看到的问题，不要猜测
- 描述用大白话（如"食指趴下去了""中指太弯了"），禁止角度数字和关节术语
- 每个 issue 必须包含 confidence 字段
- issues 最多报3个最重要的问题"""

    def _build_comparison_prompt(self, instrument: str, rag_context: str = "", check_target: str = "", mediapipe_ctx: Optional[Dict] = None) -> str:
        """有参考图对比时的分析 prompt"""
        instrument_text = "吉他" if instrument == "guitar" else "钢琴"
        target_hint = f"\n\n**学员当前在练习: {check_target}。参考图展示的就是 {check_target} 的标准手型。请对比学员的手型是否和 {check_target} 的参考图一致。**" if check_target else ""

        # 注入和弦/技巧知识（body文本优先）
        knowledge_block = ""
        if mediapipe_ctx:
            body_text = mediapipe_ctx.get("body", "")[:800]
            chord_name = mediapipe_ctx.get("chord_name", "")
            technique_name = mediapipe_ctx.get("technique_name", "")
            label = chord_name or technique_name or ""
            if body_text and label:
                knowledge_block = f"""
## {label} 标准指法知识（参考图+知识库双重对照）

{body_text}

**请结合参考图 AND 上述知识库标准，双重对照判断学员手型。**
"""
        return f"""你前面看到了正确手型的参考图（展示的是标准技巧手型），最后一张是学员的实际演奏画面。{target_hint}

{rag_context}
{knowledge_block}

## 🚨 左右手判断铁律（必须遵守）
- **靠近琴头（有调音旋钮的那一端）的手 = 左手**
- **靠近音孔/琴桥（琴体那一端）的手 = 右手**
- 人工泛音时右手会出现在指板上方靠近音孔（12品附近），这是正确技巧动作，**禁止将其误判为左手按弦错误**

请对比参考图和学员画面，判断学员的手型是否规范。

## 对比标准（用大白话判断）
- 学员手指弯曲程度和参考图差不多 → 正常，不报问题
- 学员手指明显比参考图更弯或更直（肉眼能看出来）→ 需注意
- 学员手指和参考图差别很大（完全不同的姿势）→ 问题
- 手腕位置明显偏离参考图 → 需报告
{"- 如果上方有知识库标准，知识库描述的指法细节优先于参考图的外观判断" if knowledge_block else ""}

## 判断原则
- 如果学员的手型与参考图一致或接近 → 姿势正确，不要报告问题
- **描述问题时禁止使用角度数字、度数、PIP/MCP等术语。用大白话：手指立起来了吗？手腕自然吗？**
- 如果学员的手型与参考图有明显差异 → 才报告具体问题
- 某些技巧（泛音/点弦/PM等）会导致手型变化，如果学员可能在使用这些技巧，请在描述中注明
- **最多只报 2 个最重要的问题，宁可少报不要多报。正常的手指不要列出来！**

返回严格 JSON 格式（不要 markdown 代码块）：
{{{{
  "hands_detected": true/false,
  "handedness": "左手/右手/双手",
  "overall_hand_score": 0-10,
  "issues": [
    {{{{
      "body_part": "具体手指或部位",
      "issue_type": "问题类型",
      "severity": "mild/moderate/severe",
      "description": "对比参考图后发现的差异描述，包含估算角度差",
      "suggestion": "改正建议",
      "confidence": "high/medium/low"
    }}}}
  ]
}}}}

重要：
- 不要因为参考图和学员图有差异就一律报问题——要考虑技巧动作的可能性
- 只报告确定的、与正确姿势明显不符的问题
- 最多报 2 个最重要的问题"""

    @staticmethod
    def _fix_json_text(text: str) -> str:
        """修复 LLM 输出 JSON 的常见语法错误：末尾多余逗号。"""
        import re
        # 去掉 ```json / ``` 包裹
        text = re.sub(r'^```(?:json)?\s*\n?', '', text.strip())
        text = re.sub(r'\n?```\s*$', '', text)
        # 去掉对象/数组末尾的多余逗号: ,] 或 ,}
        text = re.sub(r',\s*(\]|\})', r'\1', text)
        # 去掉 ... 占位符（模型有时偷懒）
        text = re.sub(r'"\.\.\."', '""', text)
        return text

    def _parse_vision_result(self, text: str) -> Dict:
        if not text:
            return {}
        import re
        # Step 1: 直接解析（修正常见错误后）
        cleaned = self._fix_json_text(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            pass
        except Exception:
            pass
        # Step 2: 代码块内提取 + 修正
        for pat in [r'```json\s*\n?([\s\S]*?)\n?```', r'```\s*\n?([\s\S]*?)\n?```']:
            match = re.search(pat, text)
            if match:
                inner = self._fix_json_text(match.group(1))
                try:
                    return json.loads(inner)
                except json.JSONDecodeError:
                    pass
        # Step 3: 平衡括号提取 + 修正
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start >= 0:
                    candidate = self._fix_json_text(text[start:i+1])
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        start = -1
                        continue
        # Step 4: 简单正则兜底 + 修正
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            candidate = self._fix_json_text(match.group(0))
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON解析失败 @pos={e.pos}: {e.msg}")
                if e.pos < len(candidate):
                    ctx = candidate[max(0,e.pos-40):e.pos+40]
                    logger.warning(f"错误上下文: ...{ctx}...")
        logger.warning(f"无法解析返回内容 (前200字符): {text[:200]}")
        return {}

    # ==================== 谱面预处理管道 ====================

    @staticmethod
    def _imread_safe(path: str):
        """安全读取图片，兼容 Windows 中文路径。cv2.imread 在中文路径下可能返回 None。"""
        img = cv2.imread(path)
        if img is not None:
            return img
        # 回退方案：用 Python 读取字节，再用 cv2.imdecode 解码
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    def _deduplicate_issues(self, issues: List[Dict]) -> List[Dict]:
        seen = set()
        result = []
        for issue in issues:
            # Include handedness in dedup key: same issue on different hands should not be collapsed
            hand = issue.get("handedness", "")
            key = f"{hand}|{issue.get('body_part', '')}|{issue.get('issue_type', '')}"
            if key not in seen:
                seen.add(key)
                result.append(issue)
        return result

    @staticmethod
    def _extract_timestamp(path: str) -> float:
        """从截图文件名中提取时间戳。如 snap_abc123_12.5s.jpg → 12.5"""
        import re
        match = re.search(r'_(\d+\.?\d*)s\.', os.path.basename(path))
        if match:
            return float(match.group(1))
        return 0.0


vision_analyzer = VisionAnalyzer()
