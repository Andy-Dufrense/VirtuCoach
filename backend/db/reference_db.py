"""
VirtuCoach - 正确手型参考图数据库
使用 SQLite 存储元数据 + 手部关键点向量，图片文件存储在磁盘。
支持基于 MediaPipe 21点关键点向量的余弦相似度搜索（RAG 知识库）。
"""

import os
import re
import uuid
import sqlite3
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

from logging_config import get_logger

logger = get_logger(__name__)

# 数据库和图片存储路径
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference_hands")
DB_PATH = os.path.join(DB_DIR, "references.db")
IMAGES_DIR = os.path.join(DB_DIR, "images")

# 技巧选项（中文显示名 → 数据库值）
TECHNIQUE_OPTIONS = {
    # 左手技巧
    "基本按弦": "basic-fretting",
    "横按": "barre-technique",
    "击弦": "hammer-on",
    "勾弦": "pull-off",
    "推弦": "bend",
    "揉弦": "vibrato",
    "滑音": "slide",
    "无首滑音": "slide-in",
    "无尾滑音": "slide-out",
    "自然泛音": "natural-harmonics",
    "左手点弦": "left-hand-tapping",
    "击勾弦": "trill",
    # 右手技巧
    "基础拨弦": "basic-picking",
    "扫弦": "strumming",
    "切音": "chop",
    "PM技巧": "pm-technique",
    "AM技巧": "am-technique",
    "轮扫": "rasgueado",
    "靠弦": "rest-stroke",
    "右手点弦": "right-hand-tapping",
    "右手闷音": "right-hand-muting",
    "拍泛音": "slap-harmonic",
    "人工泛音": "artificial-harmonics",
    "琶音": "arpeggio",
    "拍弦": "string-slap",
}

# 身体部位与关键词的映射，用于从问题文本中提取
BODY_PART_KEYWORDS = {
    "wrist": ["手腕"],
    "thumb": ["拇指", "大拇指"],
    "index_finger": ["食指"],
    "middle_finger": ["中指"],
    "ring_finger": ["无名指"],
    "pinky": ["小指"],
    "full_hand": ["手型", "手指", "手部", "双手"],
}


class ReferenceDB:
    """SQLite 手型参考图数据库"""

    def __init__(self, db_path: str = DB_PATH, images_dir: str = IMAGES_DIR):
        self.db_path = db_path
        self.images_dir = images_dir
        os.makedirs(self.images_dir, exist_ok=True)
        self._init_db()
        self._migrate_add_chord_kb_ids()

    # ========== 数据库初始化 ==========

    def _get_conn(self):
        """获取数据库连接。
        sqlite3 的连接不是线程安全的，每次操作创建新连接最简单安全。
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 让查询结果可以用 dict 方式访问
        conn.execute("PRAGMA journal_mode=WAL")  # WAL 模式提高并发
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        """首次运行时自动创建表 + 自动迁移添加向量列"""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS hand_references (
                id TEXT PRIMARY KEY,
                instrument TEXT NOT NULL DEFAULT 'guitar',
                hand TEXT NOT NULL DEFAULT '双手',
                technique TEXT NOT NULL DEFAULT 'basic_fretting',
                body_part TEXT NOT NULL DEFAULT 'full_hand',
                view_direction TEXT DEFAULT '前',
                description TEXT NOT NULL,
                tags TEXT DEFAULT '[]',
                image_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                landmark_vector TEXT DEFAULT NULL
            )
        """)
        # 自动迁移：旧表可能没有 landmark_vector 列
        try:
            conn.execute("SELECT landmark_vector FROM hand_references LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute("ALTER TABLE hand_references ADD COLUMN landmark_vector TEXT DEFAULT NULL")
            logger.info("已添加 landmark_vector 列（迁移）")
        conn.commit()
        conn.close()

    def _migrate_add_chord_kb_ids(self):
        """为所有和弦特定参考图的 tags 添加知识库格式的 chord ID（如 chord-Am）。

        从已有 tags 中提取和弦名（如 "Am和弦"、"C大三和弦"），
        映射为 KB 格式 ID（如 "chord-Am"、"chord-C"）并加入 tags。
        幂等操作：已有对应 ID 的条目会被跳过。
        """
        # 和弦符号 → KB 格式 ID 的映射
        CHORD_SYMBOL_TO_KB = {
            "Am7": "chord-Am7", "Am": "chord-Am",
            "A7": "chord-A7", "A": "chord-A",
            "Bm7": "chord-Bm7", "Bm": "chord-Bm", "B7": "chord-B7", "B": "chord-B",
            "Cmaj7": "chord-Cmaj7", "C7": "chord-C7", "C": "chord-C",
            "Dm7": "chord-Dm7", "Dm": "chord-Dm",
            "D7": "chord-D7", "D": "chord-D",
            "D/F#": "chord-D-over-Fsharp", "D挂F#": "chord-D-over-Fsharp",
            "Em": "chord-Em", "E7": "chord-E7", "E": "chord-E",
            "Fmaj7": "chord-fmaj7", "fmaj7": "chord-fmaj7", "F": "chord-F",
            "G7": "chord-G7", "G": "chord-G",
        }
        conn = self._get_conn()
        rows = conn.execute("SELECT id, tags FROM hand_references").fetchall()
        updated = 0
        for row_id, tags_json in rows:
            try:
                tags = json.loads(tags_json) if isinstance(tags_json, str) else (tags_json or [])
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(tags, list) or not tags:
                continue

            # 从已有 tags 中匹配和弦符号
            found_kb_id = None
            for tag in tags:
                # 去掉"和弦"后缀得到和弦符号，如 "Am和弦" → "Am"，"C大三和弦" → "C大三"
                chord_symbol = tag.replace("和弦", "").replace("大三", "").replace("小三", "")
                # 尝试匹配各种形式
                for symbol in [tag, chord_symbol]:
                    if symbol in CHORD_SYMBOL_TO_KB:
                        found_kb_id = CHORD_SYMBOL_TO_KB[symbol]
                        break
                if found_kb_id:
                    break
                # 如果 tag 本身就是和弦符号
                if tag in CHORD_SYMBOL_TO_KB:
                    found_kb_id = CHORD_SYMBOL_TO_KB[tag]
                    break

            if found_kb_id and found_kb_id not in tags:
                tags.append(found_kb_id)
                conn.execute("UPDATE hand_references SET tags=? WHERE id=?", (json.dumps(tags, ensure_ascii=False), row_id))
                updated += 1

        conn.commit()
        conn.close()
        if updated > 0:
            logger.info(f"和弦 KB ID 迁移完成: {updated} 条参考图已添加 chord-* 标签")

    # ========== 增删查 ==========

    def add(self, metadata: dict, image_data: bytes, file_ext: str = ".jpg",
            landmark_vector: Optional[List[float]] = None) -> str:
        """添加一条参考图记录。

        参数:
            metadata: 包含 instrument, hand, technique, body_part, view_direction, description, tags(list)
            image_data: 图片的二进制数据
            file_ext: 图片扩展名 (.jpg / .png)
            landmark_vector: 可选，手部21点关键点向量(63维)，用于向量相似度搜索
        返回:
            新记录的 id
        """
        ref_id = uuid.uuid4().hex[:12]
        filename = f"{ref_id}{file_ext}"
        image_path = os.path.join(self.images_dir, filename)

        # 图片存磁盘
        with open(image_path, "wb") as f:
            f.write(image_data)

        vec_json = json.dumps(landmark_vector) if landmark_vector else None

        # 元数据存数据库
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO hand_references (id, instrument, hand, technique, body_part,
               view_direction, description, tags, image_path, created_at, landmark_vector)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ref_id,
                metadata.get("instrument", "guitar"),
                metadata.get("hand", "双手"),
                metadata.get("technique", "basic_fretting"),
                metadata.get("body_part", "full_hand"),
                metadata.get("view_direction", "前"),
                metadata.get("description", ""),
                json.dumps(metadata.get("tags", []), ensure_ascii=False),
                filename,
                datetime.now().isoformat(),
                vec_json,
            ),
        )
        conn.commit()
        conn.close()
        logger.info(f"已添加参考图: {ref_id} ({metadata.get('description', '')[:30]}...)")
        return ref_id

    def delete(self, ref_id: str) -> bool:
        """删除一条参考图（数据库记录 + 磁盘图片）"""
        conn = self._get_conn()
        row = conn.execute("SELECT image_path FROM hand_references WHERE id=?", (ref_id,)).fetchone()
        if not row:
            conn.close()
            return False

        # 删除磁盘文件
        img_path = os.path.join(self.images_dir, row["image_path"])
        if os.path.exists(img_path):
            os.remove(img_path)

        conn.execute("DELETE FROM hand_references WHERE id=?", (ref_id,))
        conn.commit()
        conn.close()
        logger.info(f"已删除参考图: {ref_id}")
        return True

    def get_by_id(self, ref_id: str) -> Optional[dict]:
        """按 ID 查询单条"""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM hand_references WHERE id=?", (ref_id,)).fetchone()
        conn.close()
        if row:
            return self._row_to_dict(row)
        return None

    def list_all(self) -> List[dict]:
        """列出全部参考图"""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM hand_references ORDER BY created_at DESC").fetchall()
        conn.close()
        return [self._row_to_dict(r) for r in rows]

    def search(
        self,
        keywords: Optional[List[str]] = None,
        hand: Optional[str] = None,
        technique: Optional[str] = None,
        body_part: Optional[str] = None,
        instrument: Optional[str] = None,
    ) -> List[dict]:
        """多条件搜索参考图。

        关键词匹配逻辑: 在 description 和 tags 中做子串匹配（LIKE %keyword%）
        hand/technique/body_part/instrument 做精确匹配
        """
        conn = self._get_conn()
        conditions = []
        params = []

        if instrument:
            conditions.append("instrument=?")
            params.append(instrument)
        if hand:
            conditions.append("hand=?")
            params.append(hand)
        if technique:
            conditions.append("technique=?")
            params.append(technique)
        if body_part:
            conditions.append("body_part=?")
            params.append(body_part)
        if keywords:
            kw_clauses = []
            for kw in keywords:
                kw_clauses.append("(description LIKE ? OR tags LIKE ?)")
                like_val = f"%{kw}%"
                params.extend([like_val, like_val])
            conditions.append(f"({' OR '.join(kw_clauses)})")

        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        sql = f"SELECT * FROM hand_references {where} ORDER BY created_at DESC LIMIT 20"
        # sqlite3.Row 不支持 dict 访问？已经设置了 row_factory
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [self._row_to_dict(r) for r in rows]

    def match_from_issues(self, issues: List[dict]) -> List[dict]:
        """从手型问题列表中提取身体部位，精确匹配最相关的参考图。

        优先按 body_part 列做精确匹配，确保每个检测到的问题部位都有对应参考图。
        如果 body_part 匹配不足，再用关键词搜索补充。

        参数:
            issues: [{"哪只手": "左手", "问题": "手指较平...", "body_part": "左手"}, ...]
        返回:
            匹配到的参考图列表（去重，最多5张），每项含 _match_source 说明匹配来源
        """
        if not issues:
            return []

        # Step 1: 从问题文本中提取身体部位 → 映射到 body_part 枚举值
        detected_body_parts = []  # 保持顺序，优先匹配靠前的问题

        for h in issues:
            text = h.get("问题", "") + h.get("哪只手", "") + h.get("body_part", "")
            for bp, kws in BODY_PART_KEYWORDS.items():
                for kw in kws:
                    if kw in text and bp not in detected_body_parts:
                        detected_body_parts.append(bp)

        if not detected_body_parts:
            detected_body_parts.append("full_hand")

        seen_ids = set()
        results = []

        # Step 2: 主通道 — 按 body_part 精确匹配
        # 为每个检测到的身体部位至少取1张参考图
        for bp in detected_body_parts:
            bp_results = self.search(body_part=bp)
            for r in bp_results:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    r["_match_source"] = f"body_part:{bp}"
                    results.append(r)
            if len(results) >= 5:
                break

        # Step 3: 补充通道 — 关键词搜索（body_part 精确匹配不足时）
        if len(results) < 3:
            all_kw = set()
            for bp in detected_body_parts:
                all_kw.update(BODY_PART_KEYWORDS.get(bp, []))
            if all_kw:
                kw_results = self.search(keywords=list(all_kw))
                for r in kw_results:
                    if r["id"] not in seen_ids:
                        seen_ids.add(r["id"])
                        r["_match_source"] = "keyword"
                        results.append(r)
                        if len(results) >= 5:
                            break

        # Step 4: 兜底 — 全手参考图
        if not results:
            results = self.search(body_part="full_hand")
            for r in results:
                r["_match_source"] = "fallback"

        # 去重：优先保留不同 body_part 的参考图
        # 对于 body_part=full_hand（最常见的类别），允许多个同技术的参考图
        deduped = []
        seen_bp = set()
        seen_keys = set()
        for r in results:
            bp = r.get("body_part", "")
            tech = r.get("technique", "")
            if bp not in seen_bp:
                seen_bp.add(bp)
                seen_keys.add((bp, tech))
                deduped.append(r)
            elif bp == "full_hand":
                # full_hand 类别允许多个参考图（每个可能展示不同和弦）
                deduped.append(r)
            elif (bp, tech) not in seen_keys:
                seen_keys.add((bp, tech))
                deduped.append(r)
            if len(deduped) >= 5:
                break

        return deduped

    # ========== 工具方法 ==========

    def _row_to_dict(self, row) -> dict:
        """将 sqlite3.Row 转为普通 dict"""
        d = dict(row)
        try:
            tags = json.loads(d["tags"])
            # 规范化：tags 必须是 list。旧数据可能存成了单个字符串
            if isinstance(tags, str):
                tags = [tags]
            elif not isinstance(tags, list):
                tags = []
            d["tags"] = tags
        except (json.JSONDecodeError, TypeError):
            d["tags"] = []
        return d

    def vector_search(self, query_vector: List[float], top_k: int = 5,
                      technique: Optional[str] = None) -> List[dict]:
        """基于手部关键点向量的余弦相似度搜索。

        参数:
            query_vector: 查询向量 (63维: 21关键点 × 3坐标, 归一化后)
            top_k: 返回 Top-K 结果
            technique: 可选，限定技巧类型
        返回:
            相似度最高的参考图列表，每项包含 _similarity 字段 (0-1)
        """
        conn = self._get_conn()
        if technique:
            rows = conn.execute(
                "SELECT * FROM hand_references WHERE landmark_vector IS NOT NULL AND technique=?",
                (technique,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM hand_references WHERE landmark_vector IS NOT NULL"
            ).fetchall()
        conn.close()

        if not rows:
            return []

        query = np.array(query_vector, dtype=np.float64)
        # 归一化查询向量
        q_norm = np.linalg.norm(query)
        if q_norm > 1e-10:
            query = query / q_norm

        scored = []
        for row in rows:
            try:
                vec = np.array(json.loads(row["landmark_vector"]), dtype=np.float64)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if len(vec) != len(query_vector):
                continue
            # 归一化参考向量
            v_norm = np.linalg.norm(vec)
            if v_norm < 1e-10:
                continue
            vec = vec / v_norm
            similarity = float(np.dot(query, vec))
            d = self._row_to_dict(row)
            d["_similarity"] = round(similarity, 4)
            scored.append(d)

        scored.sort(key=lambda x: x["_similarity"], reverse=True)
        return scored[:top_k]

    def semantic_search(self, problem_type: str, top_k: int = 5) -> List[dict]:
        """基于文本语义的问题类型匹配。

        使用 KnowledgeDB 语义搜索 → 将知识条目映射回参考图（按 technique/body_part/tags 匹配）。

        Args:
            problem_type: 如 'flat_finger', 'wrist_too_low'
            top_k: 返回条数
        Returns:
            匹配到的参考图列表
        """
        try:
            from db.knowledge_db import knowledge_db as kdb
            entries = kdb.retrieve_by_problem(problem_type, top_k=3)
        except ImportError:
            entries = []
        except Exception as e:
            logger.warning(f"语义搜索失败: {e}")
            entries = []

        if not entries:
            return []  # 无匹配知识条目时返回空，不随意推荐所有参考图

        # 将知识条目映射回参考图
        results = []
        seen_ids = set()
        for entry in entries:
            # 用条目 tags 匹配参考图的 body_part
            for tag in entry.tags:
                if tag in BODY_PART_KEYWORDS:
                    bp = [k for k, v in BODY_PART_KEYWORDS.items() if tag in v]
                    if bp:
                        matched = self.search(body_part=bp[0])
                        for r in matched:
                            if r["id"] not in seen_ids:
                                seen_ids.add(r["id"])
                                r["_match_source"] = f"semantic:{entry.id}"
                                results.append(r)
            # 用问题类型匹配 technique
            matched = self.search(keywords=entry.tags)
            for r in matched:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    r["_match_source"] = f"semantic_tag:{entry.id}"
                    results.append(r)
            if len(results) >= top_k:
                break

        return results[:top_k]

    def dual_channel_match(self, issues: List[dict],
                           landmark_query: Optional[List[float]] = None,
                           top_k: int = 5, preferred_technique: str = "",
                           chord_name: str = "", chord_id: str = "",
                           preferred_hand: str = "", preferred_view: str = "") -> List[dict]:
        """多通道检索 + 相关性评分。

        所有通道并行运行，每个候选参考图按匹配维度打分：
        - 和弦ID精确匹配（chord-{X} in tags）: +60 （最高优先级，保证和弦匹配的参考图必定胜出）
        - 问题类型精确匹配（如 flat-fingers → 含按弦/手指关键词的参考图）: +40
        - 和弦名匹配（检测到 Am → 参考图 tags 含 "Am和弦"）: +35
        - 手别匹配（左手问题 → 左手参考图）: +30
        - 技巧匹配（preferred_technique 精确命中）: +25
        - 身体部位关键词匹配: +10
        - basic-fretting 兜底: +5

        最终按分数降序排列，去重后返回 top_k。
        """

        seen_ids = set()
        scored = {}  # id -> {ref_dict, score}

        def add_candidate(ref: dict, base_score: int, channel: str):
            rid = ref["id"]
            bonus = 0
            # 错误示例参考图直接排除
            if "error-demo" in str(ref.get("tags", [])):
                return
            # 手别匹配 / 不匹配惩罚
            if preferred_hand:
                ref_hand = ref.get("hand", "")
                if ref_hand == preferred_hand:
                    bonus += 30
                elif ref_hand and ref_hand not in ("", "未知"):
                    # 手别明确且不匹配：强制惩罚，确保不会误匹配
                    bonus -= 100
            # 和弦 ID 精确匹配（知识库格式：chord-Am、chord-C 等）：最高优先级
            # 但标记了所有和弦的"万能参考图"（≥6个chord标签）降权：它是兜底，不是首选
            if chord_id:
                tags = ref.get("tags", [])
                if isinstance(tags, list) and chord_id in tags:
                    n_chord_tags = sum(1 for t in tags if t.startswith("chord-"))
                    if n_chord_tags >= 6:
                        bonus += 15  # 万能参考图：低优先兜底
                    else:
                        bonus += 80  # 特定和弦参考图：高优先
            # 和弦名匹配（中文名：Am和弦、C和弦等）
            # 使用词边界避免 "Am" 误匹配 "Am7和弦"
            if chord_name:
                tags = ref.get("tags", [])
                if isinstance(tags, list) and any(re.search(rf'\b{re.escape(chord_name)}\b', t) for t in tags):
                    bonus += 35
            # 视角方向加分（手指问题→正视图，手腕问题→俯视图）
            if preferred_view and ref.get("view_direction", "") == preferred_view:
                bonus += 25
            # 和弦场景下基础按弦/横按参考图大幅加权
            if chord_name and ref.get("technique", "") in ("basic-fretting", "barre-technique"):
                bonus += 50
            total = base_score + bonus
            if rid not in scored or total > scored[rid][1]:
                ref_copy = dict(ref)
                ref_copy["_channel"] = channel
                ref_copy["_score"] = total
                scored[rid] = (ref_copy, total)
                seen_ids.add(rid)

        # ── 通道 1: 问题类型 → 精确问题匹配（最高优先级）──
        problem_map = {
            # 塌指/指腹按弦
            "放平": "flat-fingers", "塌指": "flat-fingers", "扁平": "flat-fingers",
            "PIP": "flat-fingers", "没立起来": "flat-fingers", "塌下去": "flat-fingers",
            "指尖塌": "flat-fingers", "趴下去": "flat-fingers",
            # 手腕
            "手腕": "collapsed-wrist", "塌腕": "collapsed-wrist", "往里扣": "collapsed-wrist",
            # 拇指
            "拇指": "thumb-too-high", "捏太紧": "thumb-too-high",
            "拇指位置": "thumb-too-high",
            # 小指
            "小指": "pinky-flying", "翘起": "pinky-flying",
            # 手指太竖直/太直
            "太竖直": "too-vertical-fingers", "太直了": "too-vertical-fingers",
            "站得太直": "too-vertical-fingers", "站得太垂直": "too-vertical-fingers",
            # 手指太弯/太紧张
            "太弯了": "too-curved-fingers", "太紧张": "too-curved-fingers",
            "太弯": "too-curved-fingers", "蜷缩": "too-curved-fingers",
            # 力度
            "力度": "excessive-pressure", "太用力": "excessive-pressure",
            # 杂音
            "杂音": "string-buzzing", "打品": "string-buzzing",
            # 扫弦/节奏
            "扫弦": "strumming-rhythm", "节奏不稳": "strumming-rhythm",
            # 拨弦
            "拨弦": "picking-uneven", "不均": "picking-uneven",
            # 紧张/僵硬
            "紧张": "right-hand-tension", "僵硬": "right-hand-tension",
        }
        # 问题类型 → 最接近的技巧参考图兜底（当semantic_search无结果时）
        # 问题类型 → (技巧, 最佳视角)
        # 手指弧度/角度问题 → 正视图（看指尖和关节）
        # 拇指/手腕问题 → 俯视图（看琴颈后方和手腕弧度）
        problem_technique_fallback = {
            "flat-fingers": ("basic-fretting", "前"),
            "collapsed-wrist": ("basic-fretting", "俯"),
            "thumb-too-high": ("basic-fretting", "俯"),
            "pinky-flying": ("basic-fretting", "前"),
            "too-vertical-fingers": ("basic-fretting", "前"),
            "too-curved-fingers": ("basic-fretting", "前"),
            "excessive-pressure": ("basic-fretting", "前"),
            "string-buzzing": ("basic-fretting", "前"),
            "strumming-rhythm": ("strumming", "前"),
            "picking-uneven": ("strumming", "前"),
            "right-hand-tension": ("strumming", "前"),
        }
        # 左手专属问题类型（右手问题不应匹配这些）
        LEFT_HAND_ONLY_PROBLEMS = {
            "flat-fingers", "collapsed-wrist", "thumb-too-high", "pinky-flying",
            "too-vertical-fingers", "too-curved-fingers", "excessive-pressure",
            "string-buzzing",
        }
        detected_problems = set()
        for h in (issues or []):
            text = h.get("问题", "") + h.get("description", "")
            for kw, ptype in problem_map.items():
                if kw in text:
                    # 右手问题不匹配左手专属问题类型
                    if preferred_hand == "右手" and ptype in LEFT_HAND_ONLY_PROBLEMS:
                        continue
                    detected_problems.add(ptype)

        for ptype in detected_problems:
            sem_results = self.semantic_search(ptype, top_k=3)
            if sem_results:
                for r in sem_results:
                    add_candidate(r, 40, f"problem:{ptype}")
            else:
                # 语义搜索无结果时，用问题类型对应的技巧+视角兜底
                fb = problem_technique_fallback.get(ptype)
                if fb:
                    fb_tech, fb_view = fb if isinstance(fb, tuple) else (fb, None)
                    fb_refs = self.search(technique=fb_tech)
                    # 优先匹配最佳视角的参考图
                    if fb_view:
                        fb_refs_view = [r for r in fb_refs if r.get("view_direction") == fb_view]
                        if fb_refs_view:
                            fb_refs = fb_refs_view
                    for r in fb_refs[:2]:
                        add_candidate(r, 35, f"problem_fallback:{ptype}->{fb_tech}")

        # ── 通道 2: 和弦关键词匹配（chord_name + chord_id）──
        chord_kw = []
        if chord_name:
            chord_kw.append(chord_name)
        if chord_id and chord_id not in chord_kw:
            chord_kw.append(chord_id)
        # 同时加入去除 "chord-" 前缀的简短形式，提升匹配覆盖面
        if chord_id and chord_id.startswith("chord-"):
            short = chord_id.replace("chord-", "")
            if short not in chord_kw:
                chord_kw.append(short)
        if chord_kw:
            chord_refs = self.search(keywords=chord_kw)
            # 和弦 ID 精确过滤：避免 "chord-Am" 子串匹配到 "chord-Am7"
            if chord_id:
                chord_refs = [
                    r for r in chord_refs
                    if any(chord_id == t for t in (r.get("tags") or []))
                ]
            for r in chord_refs:
                add_candidate(r, 35, f"chord:{chord_kw[0]}")
            # 和弦 ID 精确匹配通道：+60 高优先级
            if chord_id:
                chord_id_refs = self.search(keywords=[chord_id])
                # 同样过滤：chord_id 必须精确匹配 tag
                chord_id_refs = [
                    r for r in chord_id_refs
                    if any(chord_id == t for t in (r.get("tags") or []))
                ]
                for r in chord_id_refs:
                    if r["id"] not in seen_ids:
                        add_candidate(r, 60, f"chord_id:{chord_id}")

        # ── 通道 3: 技巧精确匹配 ──
        if preferred_technique:
            tech_refs = self.search(technique=preferred_technique)
            for r in tech_refs:
                add_candidate(r, 25, f"technique:{preferred_technique}")

        # ── 通道 4: 身体部位关键词匹配 ──
        if issues:
            all_kw = set()
            for h in issues:
                text = h.get("问题", "") + h.get("哪只手", "")
                for bp, kws in BODY_PART_KEYWORDS.items():
                    for kw in kws:
                        if kw in text:
                            all_kw.add(kw)
            if all_kw:
                kw_refs = self.search(keywords=list(all_kw))
                for r in kw_refs:
                    # 右手问题优先匹配右手参考图，左手同样
                    if preferred_hand and r.get("hand", "") != preferred_hand:
                        continue
                    add_candidate(r, 10, "body_keyword")

        # ── 通道 5: 兜底（按手别选择对应技巧）──
        fallback_tech = "basic-fretting" if preferred_hand != "右手" else "strumming"
        basic_refs = self.search(technique=fallback_tech)
        for r in basic_refs[:3]:
            add_candidate(r, 5, "basic_fallback")

        # ── 通道 6: 向量相似度（可选）──
        if landmark_query:
            vec_results = self.vector_search(landmark_query, top_k=3)
            for r in vec_results:
                add_candidate(r, 15, "vector")

        # ── 按分数排序，返回 top_k ──
        # 右手问题没有和弦加分，MIN_SCORE 不能太高
        MIN_SCORE = 45
        sorted_refs = sorted(scored.values(), key=lambda x: x[1], reverse=True)
        # 排除明确标注为错误示例的参考图（通过 description 内容判断，而非 AI 备注）
        NEGATIVE_DESC_PREFIX = ["错误示范", "问题手型示例", "不要这样做"]
        result = []
        for ref, score in sorted_refs:
            if score < MIN_SCORE:
                continue
            desc = ref.get("description", "")
            # 只过滤 description 主体内容（不含 AI 备注部分）
            main_desc = desc.split("[AI")[0] if "[AI" in desc else desc
            if any(kw in main_desc for kw in NEGATIVE_DESC_PREFIX):
                continue
            result.append(ref)
            if len(result) >= top_k:
                break
        logger.info(
            f"dual_channel: technique={preferred_technique or 'none'}, "
            f"chord={chord_name or 'none'}, hand={preferred_hand or 'none'}, "
            f"problems={list(detected_problems)[:3]}, "
            f"results={[(r['id'][:6], r.get('_score', 0)) for r in result]}"
        )
        return result

    def update_landmark_vector(self, ref_id: str, vector: List[float]) -> bool:
        """更新已有参考图的关键点向量"""
        vec_json = json.dumps(vector)
        conn = self._get_conn()
        conn.execute(
            "UPDATE hand_references SET landmark_vector=? WHERE id=?",
            (vec_json, ref_id),
        )
        conn.commit()
        affected = conn.total_changes
        conn.close()
        return affected > 0

    def get_unvectorized(self) -> List[dict]:
        """获取所有还没有关键点向量的参考图"""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM hand_references WHERE landmark_vector IS NULL"
        ).fetchall()
        conn.close()
        return [self._row_to_dict(r) for r in rows]

    def image_url(self, ref: dict) -> str:
        """根据记录生成图片访问 URL"""
        return f"/api/reference-images/{ref['image_path']}"

    # ==================== Phase 2e: 直接匹配 ====================

    def match_by_chord(self, chord_id: str, hand: str = None,
                        exclude_ids: set = None) -> Optional[dict]:
        """按和弦 ID 精确匹配参考图。使用 exact tag membership 而非子串匹配。

        Args:
            chord_id: 和弦 ID (如 "B7", "Am", "C")
            hand: 可选手别过滤 ("左手" / "右手")
            exclude_ids: 排除已使用的参考图 ID 集合

        Returns:
            匹配的参考图 dict，若无精确匹配返回 None
        """
        conn = self._get_conn()
        query = "SELECT * FROM hand_references WHERE tags LIKE ?"
        # 搜索 chord-{id} 标签（如 chord-C, chord-Am），比裸子串匹配精确得多
        chord_tag = f"chord-{chord_id}"
        params = [f"%{chord_tag}%"]
        rows = conn.execute(query, params).fetchall()
        conn.close()

        all_results = [self._row_to_dict(r) for r in rows]
        # 精确 tag 过滤：只保留 tags 列表中有 chord-C 这个 exact tag 的条目
        # （SQLite LIKE 可能匹配到 chord-Cmaj7 等前缀，需要 Python 二次过滤）
        results = [
            r for r in all_results
            if chord_tag in set(r.get("tags", []))
        ]

        if not results:
            return None

        # 过滤错误示例（error-demo 标记的参考图不应作为正确手型返回）
        results = [r for r in results if "error-demo" not in str(r.get("tags", []))]

        # 排除已使用的
        if exclude_ids:
            results = [r for r in results if r.get("id") not in exclude_ids]

        # 手别过滤：必须匹配指定手，否则视为无结果
        if hand:
            filtered = [r for r in results if r.get("hand") == hand]
            if filtered:
                results = filtered
            else:
                return None  # 无匹配手别，回退到下一级

        # 优先 basic-fretting（和弦参考图），其次按创建时间倒序
        results.sort(key=lambda r: (
            0 if r.get("technique") == "basic-fretting" else 1,
            r.get("created_at", ""),
        ), reverse=False)
        # basic-fretting 优先，然后按创建时间旧→新（旧图先建先选）
        return results[0] if results else None

    def match_by_technique(self, technique: str, hand: str = None,
                            exclude_ids: set = None) -> Optional[dict]:
        """按技巧名精确匹配参考图。

        Args:
            technique: 技巧 ID (如 "barre-technique", "hammer-on")
            hand: 可选手别过滤
            exclude_ids: 排除已使用的参考图 ID 集合

        Returns:
            匹配的参考图 dict，若无精确匹配返回 None
        """
        conn = self._get_conn()
        query = "SELECT * FROM hand_references WHERE technique = ?"
        rows = conn.execute(query, (technique,)).fetchall()
        conn.close()

        results = [self._row_to_dict(r) for r in rows]
        if not results:
            return None

        if exclude_ids:
            results = [r for r in results if r.get("id") not in exclude_ids]

        if hand:
            filtered = [r for r in results if r.get("hand") == hand]
            if filtered:
                results = filtered
            else:
                return None  # 无匹配手别，回退到下一级

        return results[0] if results else None

    def match_reference_image(self, chord_id: str = None, technique: str = None,
                               hand: str = None, issue_text: str = "",
                               fallback_match_fn=None,
                               exclude_ids: set = None) -> Optional[dict]:
        """四级回退链参考图匹配。

        1. 和弦直接匹配 (chord_id → tag)
        2. 技巧匹配 (technique → technique tag)
        3. 手别匹配 (hand → hand tag)
        4. dual_channel_match() 兜底

        Returns:
            最佳参考图 dict，或 None
        """
        exclude_ids = exclude_ids or set()

        # Tier 1: 和弦直接匹配
        if chord_id:
            result = self.match_by_chord(chord_id, hand=hand, exclude_ids=exclude_ids)
            if result:
                logger.info(f"参考图匹配(T1和弦): {chord_id} → {result['id']}")
                return result

        # Tier 2: 技巧匹配
        if technique:
            result = self.match_by_technique(technique, hand=hand, exclude_ids=exclude_ids)
            if result:
                logger.info(f"参考图匹配(T2技巧): {technique} → {result['id']}")
                return result

        # Tier 3: 手别匹配
        if hand:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM hand_references WHERE hand = ? LIMIT 3",
                (hand,),
            ).fetchall()
            conn.close()
            results = [self._row_to_dict(r) for r in rows]
            if exclude_ids:
                results = [r for r in results if r.get("id") not in exclude_ids]
            if results:
                logger.info(f"参考图匹配(T3手别): {hand} → {results[0]['id']}")
                return results[0]

        # Tier 4: 兜底
        if fallback_match_fn and issue_text:
            results = fallback_match_fn(issue_text)
            if results:
                # 排除已使用的
                if exclude_ids:
                    results = [r for r in results if (r.get("id") if isinstance(r, dict) else r) not in exclude_ids]
                if results:
                    first = results[0]
                    logger.info(f"参考图匹配(T4兜底): → {first['id'] if isinstance(first, dict) else first}")
                    return first if isinstance(first, dict) else first

        return None


# 模块级别的单例
_db_instance = None


def get_db() -> ReferenceDB:
    """获取全局 ReferenceDB 实例（单例模式）"""
    global _db_instance
    if _db_instance is None:
        _db_instance = ReferenceDB()
        existing = len(_db_instance.list_all())
        logger.info(f"数据库就绪，已有 {existing} 条参考图")
    return _db_instance
