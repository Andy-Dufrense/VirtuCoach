"""
VirtuCoach - RAG 知识库模块
从 knowledge/ 目录加载教学知识条目，提供语义搜索和 prompt 注入。

主方案: ChromaDB + bge-large-zh embedding
降级方案: 关键词匹配（依赖缺失时自动切换）
"""

import os
import re
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# 必须在 import sentence_transformers 之前设置，否则 huggingface_hub 联网检查会触发 504 重试
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME",
                      os.path.join(os.path.dirname(__file__), "..", "models", "st_cache"))

from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class KnowledgeEntry:
    """知识库条目"""
    id: str
    type: str          # technique | problem | exercise
    title: str
    severity: str = ""
    difficulty: str = ""
    related_techniques: List[str] = field(default_factory=list)
    related_problems: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    body: str = ""     # markdown body
    filepath: str = ""


class KnowledgeDB:
    """吉他教学知识库。启动时加载 knowledge/ 目录下所有 .md 文件，
    用 embedding 向量化后存储在 ChromaDB 中，支持语义搜索。
    """

    def __init__(self, knowledge_dir: str = None):
        if knowledge_dir is None:
            knowledge_dir = os.path.join(os.path.dirname(__file__), "..", "knowledge")
        self.knowledge_dir = os.path.abspath(knowledge_dir)
        self.entries: List[KnowledgeEntry] = []
        self._chroma_available = False
        self._collection = None
        self._embed_fn = None

        # 尝试初始化 ChromaDB
        try:
            import chromadb
            from chromadb.config import Settings
            self._chroma_client = chromadb.Client(Settings(
                anonymized_telemetry=False,
                is_persistent=False,
            ))
            self._collection = self._chroma_client.get_or_create_collection("guitar_knowledge")
            self._chroma_available = True
            logger.info("ChromaDB 就绪")
        except ImportError:
            logger.warning("ChromaDB 未安装，使用关键词匹配模式")
        except Exception as e:
            logger.warning(f"ChromaDB 初始化失败: {e}，使用关键词匹配模式")

        # 尝试加载 embedding 模型
        if self._chroma_available:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("正在加载 bge-small-zh embedding 模型（离线）...")
                self._embed_model = SentenceTransformer(
                    "BAAI/bge-small-zh-v1.5",
                    local_files_only=True,
                )
                self._embed_fn = lambda texts: self._embed_model.encode(
                    texts, normalize_embeddings=True
                ).tolist()
                logger.info("bge-small-zh embedding 就绪")
            except ImportError:
                logger.warning("sentence-transformers 未安装，ChromaDB 无法使用 embedding")
                self._chroma_available = False
            except Exception as e:
                logger.warning(f"embedding 模型加载失败: {e}，使用关键词匹配模式")
                self._chroma_available = False

        # 加载知识条目
        self._load_all()

    def _load_all(self):
        """加载 knowledge/ 下所有 .md 文件"""
        if not os.path.isdir(self.knowledge_dir):
            logger.warning(f"知识库目录不存在: {self.knowledge_dir}")
            return

        md_files = []
        for root, dirs, files in os.walk(self.knowledge_dir):
            # 排除 templates 目录
            if "templates" in root.split(os.sep):
                continue
            for f in files:
                if f.endswith(".md"):
                    md_files.append(os.path.join(root, f))

        for fpath in sorted(md_files):
            try:
                entry = self._parse_entry(fpath)
                if entry:
                    self.entries.append(entry)
            except Exception as e:
                logger.warning(f"解析失败: {fpath} - {e}")

        logger.info(f"加载 {len(self.entries)} 个知识条目")

        # 向量化并存入 ChromaDB
        if self._chroma_available and self._embed_fn and self.entries:
            try:
                texts = [self._entry_to_text(e) for e in self.entries]
                ids = [e.id for e in self.entries]
                embeddings = self._embed_fn(texts)
                metadatas = [
                    {"type": e.type, "severity": e.severity, "difficulty": e.difficulty,
                     "tags": ",".join(e.tags), "title": e.title}
                    for e in self.entries
                ]
                # 清空并重新填充
                try:
                    self._chroma_client.delete_collection("guitar_knowledge")
                    self._collection = self._chroma_client.get_or_create_collection("guitar_knowledge")
                except Exception:
                    pass
                self._collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    ids=ids,
                    metadatas=metadatas,
                )
                logger.info(f"{len(self.entries)} 条目已向量化")
            except Exception as e:
                logger.warning(f"向量化失败: {e}")
                self._chroma_available = False

    @staticmethod
    def _parse_entry(filepath: str) -> Optional[KnowledgeEntry]:
        """解析 YAML frontmatter + markdown body"""
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        frontmatter = {}
        body = content
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                fm_text = parts[1]
                body = parts[2]
                for line in fm_text.strip().split("\n"):
                    line = line.strip()
                    if ":" in line:
                        key, val = line.split(":", 1)
                        key = key.strip()
                        val = val.strip()
                        # 解析列表 [a, b, c]
                        if val.startswith("[") and val.endswith("]"):
                            val = [v.strip().strip('"').strip("'") for v in val[1:-1].split(",") if v.strip()]
                        frontmatter[key] = val

        # 提取标题（第一个 # 行）
        title = frontmatter.get("id", os.path.splitext(os.path.basename(filepath))[0])
        for line in body.split("\n"):
            if line.startswith("# "):
                title = line[2:].strip()
                break

        return KnowledgeEntry(
            id=frontmatter.get("id", os.path.splitext(os.path.basename(filepath))[0]),
            type=frontmatter.get("type", "technique"),
            title=title,
            severity=frontmatter.get("severity", ""),
            difficulty=frontmatter.get("difficulty", ""),
            related_techniques=frontmatter.get("related_techniques", []),
            related_problems=frontmatter.get("related_problems", []),
            tags=frontmatter.get("tags", []),
            body=body.strip(),
            filepath=filepath,
        )

    @staticmethod
    def _entry_to_text(entry: KnowledgeEntry) -> str:
        """将条目转为可供 embedding 的文本"""
        parts = [
            f"标题: {entry.title}",
            f"类型: {entry.type}",
        ]
        if entry.tags:
            parts.append(f"标签: {', '.join(entry.tags)}")
        parts.append(entry.body[:800])
        return "\n".join(parts)

    def search(self, query: str, top_k: int = 5) -> List[KnowledgeEntry]:
        """语义搜索（ChromaDB）或关键词匹配（降级）"""
        if self._chroma_available and self._embed_fn:
            try:
                q_embedding = self._embed_fn([query])
                results = self._collection.query(
                    query_embeddings=q_embedding,
                    n_results=min(top_k, len(self.entries)),
                )
                ids = results.get("ids", [[]])[0]
                id_to_entry = {e.id: e for e in self.entries}
                return [id_to_entry[eid] for eid in ids if eid in id_to_entry]
            except Exception as e:
                logger.warning(f"语义搜索失败: {e}，降级到关键词匹配")

        return self._keyword_search(query, top_k)

    # 中文停用字：高频干扰字，不参与关键词匹配
    _STOP_CHARS = set('的是我了不在有这要什么为个可以就也会对中后上和没到说让把但而很能去如与或及所它从当用两向自因好还样被更里吗呢嘛哦哈啊吧呀嗯')
    # 吉他领域高频通用字：出现在几乎所有条目中，无区分度
    _NOISE_CHARS = set('弦手指弹奏音练习用级高按方每检学')

    def _keyword_search(self, query: str, top_k: int = 5) -> List[KnowledgeEntry]:
        """关键词匹配降级方案。中文按单字+双字组合分词，英文按单词。
        停用字和高频噪音字会被过滤，英文和双字组合权重更高。"""
        chars = re.findall(r'[一-鿿]', query)
        # 单字：过滤停用字和噪音字
        single_chars = set(c for c in chars if c not in self._STOP_CHARS and c not in self._NOISE_CHARS)
        # 双字组合：高精度，不过滤
        bigrams = set()
        for i in range(len(chars) - 1):
            bg = chars[i] + chars[i + 1]
            # 双字组合只要不全是停用字就保留
            if not all(c in self._STOP_CHARS for c in bg):
                bigrams.add(bg)
        eng_words = set(re.findall(r'[a-zA-Z]+', query.lower()))
        tokens = single_chars | bigrams | eng_words
        if not tokens:
            return self.entries[:top_k]

        scored = []
        for entry in self.entries:
            text = (entry.title + " " + " ".join(entry.tags) + " " + entry.body[:500]).lower()
            # 加权命中：英文词权重3，双字组合权重2，单字权重1
            weighted_hits = 0
            for t in tokens:
                if t in text:
                    if t in eng_words:
                        weighted_hits += 3  # PM, tapping 等英文缩写区分度极高
                    elif len(t) == 2 and t in bigrams:
                        weighted_hits += 2  # 双字组合如"轮扫""闷音"高精度
                    else:
                        weighted_hits += 1
            if weighted_hits > 0:
                scored.append((weighted_hits, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def retrieve_by_problem(self, problem_type: str, top_k: int = 3) -> List[KnowledgeEntry]:
        """按问题类型检索：先精确匹配 id/tags，再语义搜索相关条目。

        Args:
            problem_type: 如 'flat_finger', 'wrist_too_low', 'pinky_flying'
            top_k: 返回条数
        """
        results = []

        # 第一轮：精确匹配 problem id
        for entry in self.entries:
            if entry.id == problem_type or problem_type in entry.tags:
                results.append(entry)

        # 第二轮：匹配 related_problems
        for entry in self.entries:
            if entry not in results and problem_type in entry.related_problems:
                results.append(entry)

        # 第三轮：语义搜索补充
        if len(results) < top_k:
            semantic = self.search(problem_type.replace("-", " ").replace("_", " "), top_k)
            for e in semantic:
                if e not in results:
                    results.append(e)

        # 展开关联条目
        added = set(e.id for e in results)
        for entry in list(results):
            for rel_id in entry.related_techniques:
                if rel_id not in added:
                    rel = self._get_by_id(rel_id)
                    if rel:
                        results.append(rel)
                        added.add(rel_id)

        return results[:top_k]

    def _get_by_id(self, entry_id: str) -> Optional[KnowledgeEntry]:
        for e in self.entries:
            if e.id == entry_id:
                return e
        return None

    # Section priority for AI prompts: critical detection rules must never be truncated
    _SECTION_PRIORITY = [
        ("AI检测", 400), ("AI 检测", 400), ("铁律", 400),
        ("区分", 300), ("判断原则", 300), ("常见错误", 300),
        ("错误", 200), ("练习方法", 300), ("练习曲", 200),
        ("标准手型", 200), ("正确姿势", 200), ("指法", 200),
    ]

    @classmethod
    def _extract_sections(cls, body: str, max_total: int = 1500) -> str:
        """Section-aware body extraction. High-priority sections (AI检测/铁律/常见错误)
        are extracted in full; remaining space is filled with the beginning of body.
        """
        lines = body.split("\n")
        extracted_parts = []
        used_chars = 0

        # Collect all priority marker names for section-boundary detection
        _all_markers = {m[0] for m in cls._SECTION_PRIORITY}

        for marker, budget in cls._SECTION_PRIORITY:
            if used_chars >= max_total:
                break
            in_section = False
            section_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("#") and marker in stripped:
                    in_section = True
                    section_lines.append(line)
                elif in_section:
                    # Check against ALL priority markers, not just current one
                    if stripped.startswith("#") and any(m in stripped for m in _all_markers):
                        break
                    section_lines.append(line)
            if section_lines:
                section_text = "\n".join(section_lines)
                section_text = section_text[:budget]
                extracted_parts.append(section_text)
                used_chars += len(section_text)

        # Fill remaining budget with body beginning (skip already-extracted sections)
        if used_chars < max_total:
            # Simple approach: take the first N chars of body not in extracted sections
            body_remain = body
            for part in extracted_parts:
                body_remain = body_remain.replace(part, "")
            remaining = body_remain.strip()[:max_total - used_chars]
            if remaining:
                extracted_parts.append(remaining)

        return "\n\n".join(extracted_parts)

    @classmethod
    def extract_detection_rules(cls, entry: KnowledgeEntry) -> str:
        """Extract only the AI-facing detection rules from a knowledge entry.
        Used by Vision Analyzer to inject authoritative detection criteria.
        """
        body = entry.body
        rules = []
        for line in body.split("\n"):
            stripped = line.strip()
            if any(kw in stripped for kw in ["铁律", "AI检测", "AI 检测", "区分", "判断原则",
                                               "🚨", "⚠️", "禁止", "必须"]):
                rules.append(stripped)
        if rules:
            return f"## {entry.title} 检测规则\n" + "\n".join(rules[:15])
        return ""

    def format_for_prompt(self, entries: List[KnowledgeEntry]) -> str:
        """将知识条目格式化为 LLM prompt 片段。
        使用 section-aware 提取，优先保留 AI检测/铁律/常见错误 等关键段落。

        重要：指令已改为「参考框架」而非「绝对权威」，避免 AI 机械复读。
        """
        if not entries:
            return ""

        blocks = [
            "## 教学参考资料\n",
            "以下是吉他教学的专业知识。你需要理解这些内容后，用自己的教学语言向学员解释。",
            "不要逐字复读——根据学员的水平，选择合适的深度和表达方式重新组织语言。\n",
        ]
        for i, entry in enumerate(entries, 1):
            blocks.append(f"### {i}. {entry.title}")
            blocks.append(f"**类型**: {entry.type} | **难度**: {entry.difficulty}")
            if entry.tags:
                blocks.append(f"**标签**: {', '.join(entry.tags)}")
            body_smart = self._extract_sections(entry.body)
            if len(body_smart) > 800:
                body_smart = body_smart[:800] + "\n\n..."
            blocks.append(body_smart)
            blocks.append("")

        return "\n".join(blocks)

    def build_problem_kw_map(self) -> dict:
        """从知识库自动生成 关键词→问题/技巧类型 映射表。

        替代 deepseek_agent.py 中手写的 60+ 条目 problem_kw_map。
        每个知识条目的 frontmatter 包含 tags，body 中包含常见中文描述。
        """
        kw_map = {}

        # 预定义关键词映射（中文常见描述 → 知识条目 ID）
        # 这些是用户和检测系统常用的中文表达
        _BASE_MAP = {
            # 手型问题
            "放平": "flat-fingers", "塌指": "flat-fingers", "偏平": "flat-fingers",
            "指尖低于": "flat-fingers", "指尖偏平": "flat-fingers",
            "指尖不够垂直": "flat-fingers", "手型塌陷": "flat-fingers",
            "塌下去": "flat-fingers", "指尖塌": "flat-fingers",
            "趴下去": "flat-fingers", "没立起来": "flat-fingers",
            "太竖直": "too-vertical-fingers", "竖直": "too-vertical-fingers",
            "太垂直": "too-vertical-fingers", "垂直按弦": "too-vertical-fingers",
            "站得太直": "too-vertical-fingers", "过度垂直": "too-vertical-fingers",
            "太直了": "too-vertical-fingers", "站得太垂直": "too-vertical-fingers",
            "太弯了": "too-curved-fingers", "太弯": "too-curved-fingers",
            "蜷缩": "too-curved-fingers", "太紧张": "too-curved-fingers",
            "手腕": "collapsed-wrist", "往里扣": "collapsed-wrist",
            "拇指": "thumb-too-high", "拇指位置": "thumb-too-high",
            "捏太紧": "thumb-too-high",
            "小指": "pinky-flying", "翘起": "pinky-flying",
            "力度": "excessive-pressure", "太用力": "excessive-pressure",
            "弯曲过度": "excessive-pressure",
            "拇指过度伸直": "thumb-no-support",
            "手掌": "palm-perpendicular",
            # 技巧
            "击弦": "hammer-on", "勾弦": "pull-off",
            "击勾弦": "trill", "滑音": "slide",
            "无首滑音": "slide-in", "无尾滑音": "slide-out",
            "横按": "barre-technique", "推弦": "bend",
            "揉弦": "vibrato", "点弦": "left-hand-tapping",
            "泛音": "natural-harmonics", "基础按弦": "basic-fretting",
            "扫弦": "strumming", "下扫": "strumming", "上扫": "strumming",
            "拨弦": "basic-picking", "分解": "arpeggio",
            "琶音": "arpeggio", "依次拨弦": "arpeggio",
            "打板": "pm-technique", "敲击面板": "pm-technique", "腕骨": "pm-technique",
            "切音": "chop", "大鱼际压弦": "chop",
            "闷音": "right-hand-muting", "掌根": "right-hand-muting",
            "靠弦": "rest-stroke",
            "轮扫": "rasgueado", "四指连弹": "rasgueado",
            "人工泛音": "artificial-harmonics", "拍泛音": "slap-harmonic",
            "右手点弦": "right-hand-tapping",
            "AM": "am-technique", "am技巧": "am-technique", "PM": "pm-technique",
            # 心理/体验
            "手指疼": "finger-pain", "指尖疼": "finger-pain", "手疼": "finger-pain",
            "按弦疼": "finger-pain", "按弦痛": "finger-pain", "手指痛": "finger-pain",
            "指套": "finger-pain", "护指": "finger-pain", "硅胶套": "finger-pain",
            "茧子": "finger-pain", "长茧": "finger-pain", "老茧": "finger-pain",
            "起泡": "finger-pain", "水泡": "finger-pain", "手指起泡": "finger-pain",
            "手痛": "finger-pain", "指尖痛": "finger-pain", "指尖起泡": "finger-pain",
            # 右手疼痛/紧张
            "右手疼": "right-hand-tension", "右手痛": "right-hand-tension",
            "右手酸": "right-hand-tension", "前臂酸": "right-hand-tension",
            "手腕酸": "right-hand-tension", "拨弦手酸": "right-hand-tension",
            "弹琴手酸": "right-hand-tension", "右手僵硬": "right-hand-tension",
            "手腕僵硬": "right-hand-tension", "手腕太僵": "right-hand-tension",
            "右手紧张": "right-hand-tension", "右手太紧": "right-hand-tension",
        }

        # 从知识库 entries 扩展：每个 entry 的 title 和 tags 作为额外关键词
        for entry in self.entries:
            entry_id = entry.id
            # title 中的中文部分作为关键词
            title_words = re.findall(r'[一-鿿]{2,}', entry.title)
            for w in title_words:
                if w not in kw_map and w not in _BASE_MAP:
                    kw_map[w] = entry_id
            # tags 作为关键词
            for tag in entry.tags:
                if tag not in kw_map and tag not in _BASE_MAP:
                    kw_map[tag] = entry_id

        # 基础映射优先级更高
        kw_map.update(_BASE_MAP)
        return kw_map

    def search_by_audio_errors(self, audio_errors: list, top_k: int = 3) -> List[KnowledgeEntry]:
        """根据音频错误类型搜索相关教学知识"""
        audio_kw_map = {
            "rhythm_fault": "节奏 练习 节拍器",
            "transition_gap": "换和弦 和弦转换 换把位 流畅度",
            "wrong_note": "音准 按弦 品位",
            "extra_note": "多余拨弦 制音",
            "missed_note": "漏弹 完整性",
            "overlap": "音符重叠 消音 闷音",
            "pause": "停顿 流畅度",
            "bend_issue": "推弦 音高",
            "slide_issue": "滑音",
        }
        search_terms = set()
        for e in audio_errors[:5]:
            etype = e.get("type", "")
            if etype in audio_kw_map:
                search_terms.add(audio_kw_map[etype])
        if search_terms:
            return self.search(" ".join(search_terms), top_k)
        return []

    def build_rag_for_question(self, question: str, context: dict, top_k: int = 3) -> str:
        """统一的 RAG 检索入口，供 chat router 和 deepseek_agent 共用。

        根据用户问题 + 分析上下文构造搜索查询，返回格式化的 prompt 片段。
        """
        import re
        search_terms = [question]

        # 英文缩写扩展（如 PM → PM技巧）
        eng_abbr = re.findall(r'[a-zA-Z]{1,4}', question.strip())
        if eng_abbr:
            search_terms.append(f"{eng_abbr[0]}技巧")

        # 和弦名
        chord_name = context.get("chord_name", "")
        if chord_name:
            search_terms.append(f"{chord_name} 和弦 指法")

        # 技巧名
        technique = context.get("technique", "")
        if technique:
            search_terms.append(technique)

        # 手型问题文本
        hand_issues = context.get("hand_issues", [])
        if hand_issues:
            search_terms.extend([h.get("问题", "") for h in hand_issues[:2]])

        query = " ".join(search_terms)
        entries = self.search(query, top_k=top_k)
        if entries:
            return self.format_for_prompt(entries)
        return ""

    # ==================== KB 结构化查询（Phase 2a） ====================

    def get_chord_angle_standards(self, chord_id: str) -> Dict:
        """获取指定和弦的逐指理想角度标准。

        返回：{"食指": {"pip_ideal": 135, "mcp_ideal": 70}, ...}
        使用内存缓存。
        """
        if not hasattr(self, '_chord_angle_cache'):
            self._chord_angle_cache = {}

        if chord_id in self._chord_angle_cache:
            return self._chord_angle_cache[chord_id]

        try:
            from chord_analyzer import ChordAnalyzer
            templates = ChordAnalyzer._load_chord_templates_with_finger_data()
            if chord_id in templates:
                angles = templates[chord_id].get("finger_angles", {})
                self._chord_angle_cache[chord_id] = angles
                return angles
        except Exception as e:
            logger.warning(f"加载和弦角度标准失败({chord_id}): {e}")

        self._chord_angle_cache[chord_id] = {}
        return {}

    def get_problem_thresholds(self) -> Dict:
        """解析 KB problem 文件的检测规则，返回结构化阈值。

        返回：{"flat-fingers": {"triggers": [...], "exclusions": [...], ...}, ...}
        使用内存缓存。
        """
        if hasattr(self, '_problem_thresholds_cache'):
            return self._problem_thresholds_cache

        rules = {}
        problems_dir = os.path.join(self.knowledge_dir, "problems")
        if not os.path.isdir(problems_dir):
            self._problem_thresholds_cache = rules
            return rules

        import re
        for fname in sorted(os.listdir(problems_dir)):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(problems_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                pid = fname[:-3]

                triggers = []
                for match in re.finditer(
                    r'PIP\s*[>＜<≥≤]\s*(\d+)\s*[°度]', content
                ):
                    op = ">" if ">" in match.group() or "＞" in match.group() else "<"
                    if "＜" in match.group() or "≤" in match.group():
                        op = "<"
                    triggers.append({"type": "pip", "op": op, "value": int(match.group(1))})

                for match in re.finditer(
                    r'MCP\s*[>＜<≥≤]\s*(\d+)\s*[°度]', content
                ):
                    op = ">" if ">" in match.group() or "＞" in match.group() else "<"
                    if "＜" in match.group() or "≤" in match.group():
                        op = "<"
                    triggers.append({"type": "mcp", "op": op, "value": int(match.group(1))})

                min_fingers = 1
                fm = re.search(r'≥\s*(\d+)\s*[根只].*?手指', content)
                if fm:
                    min_fingers = int(fm.group(1))

                exclusions = []
                if ("横按" in content and ("除外" in content or "豁免" in content
                                            or "不应算问题" in content)):
                    exclusions.append("barre_chord")
                if "推弦" in content or "滑音" in content:
                    if "除外" in content:
                        exclusions.append("bend_or_slide")

                sustained = 0
                sm = re.search(r'持续\s*[>＞]\s*(\d+)\s*秒', content)
                if sm:
                    sustained = int(sm.group(1))

                rules[pid] = {
                    "triggers": triggers,
                    "exclusions": exclusions,
                    "min_fingers": min_fingers,
                    "require_sustained_seconds": sustained,
                }
            except Exception:
                continue

        self._problem_thresholds_cache = rules
        logger.info(f"解析 {len(rules)} 个 problem 检测规则: "
                     + ", ".join(f"{k}({len(v['triggers'])} trig)" for k, v in rules.items()))
        return rules


# 全局单例
knowledge_db = KnowledgeDB()
