"""
统一的手型问题过滤管道。

替代原来分散在 run_analysis() 和 check_chord() 中的 8 层过滤逻辑。
过滤阶段可配置、可单独测试，每个阶段记录过滤原因。
"""

from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field
import json

from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class FilterContext:
    """过滤上下文：携带分析模式、技巧信息、Vision AI 结果等"""
    mode: str = "freeplay"  # "freeplay" | "chord" | "technique"
    technique_id: str = ""
    is_right_hand_tech: bool = False
    vision_handedness: str = ""  # Vision AI 判定的左右手（"左手"/"右手"/""）
    technique_allowed_keywords: set = field(default_factory=set)
    force_hand: str = ""  # 强制指定手别（和弦检查用）
    first_note_time: float = 0.0
    audio_quality: str = "unknown"
    technique_segments: list = field(default_factory=list)  # [{start_time, end_time, technique_id, confidence}]
    user_level: str = "beginner"  # "beginner" | "intermediate" | "advanced"
    is_barre_chord: bool = False  # 是否正在演奏大横按（横按时拇指力度阈值需放宽）
    is_bend_vibrato: bool = False  # 是否正在演奏推弦/揉弦（角度和力度阈值需放宽）


class HandIssueFilter:
    """手型问题过滤管道。按顺序执行多个过滤阶段。"""

    def __init__(self):
        self._stages: List[tuple] = []  # [(name, filter_fn)]
        self._filter_log: List[dict] = []

    def add_stage(self, name: str, fn: Callable):
        """添加过滤阶段。fn 接收 (issues, context) → filtered_issues"""
        self._stages.append((name, fn))
        return self

    def apply(self, issues: List[dict], context: FilterContext) -> List[dict]:
        """按顺序应用所有过滤阶段，返回过滤后的问题列表"""
        self._filter_log = []
        result = list(issues)
        before_count = len(result)

        for stage_name, stage_fn in self._stages:
            result = stage_fn(result, context)
            after_count = len(result)
            removed = before_count - after_count
            if removed > 0:
                self._filter_log.append({
                    "stage": stage_name,
                    "before": before_count,
                    "after": after_count,
                    "removed": removed,
                })
                logger.info(f"{stage_name}: {before_count} → {after_count} (-{removed})")
            before_count = after_count

        return result

    def get_log(self) -> List[dict]:
        return self._filter_log


# ====== 预定义过滤阶段 ======

def filter_early_frames(issues: List[dict], ctx: FilterContext) -> List[dict]:
    """过滤 t < 0.3s 的帧（用户尚未开始弹奏）"""
    return [
        h for h in issues
        if float(h.get("时间", h.get("timestamp", 0))) >= 0.3
    ]


def filter_pre_music(issues: List[dict], ctx: FilterContext) -> List[dict]:
    """过滤音乐开始前的问题。第一个音符前0.5秒保留（手型准备期），更早的视为调整期噪声。"""
    if ctx.first_note_time <= 0:
        return issues
    cutoff = max(0.5, ctx.first_note_time - 0.5)
    return [
        h for h in issues
        if float(h.get("时间", h.get("timestamp", 0))) >= cutoff
    ]


def filter_normal_issues(issues: List[dict], ctx: FilterContext) -> List[dict]:
    """过滤「正常」和「基本正常」的伪问题"""
    return [
        h for h in issues
        if "正常" not in h.get("问题", "") and "基本正常" not in h.get("问题", "")
        and "正常" not in h.get("description", "") and "基本正常" not in h.get("description", "")
    ]


def filter_technique_allowed(issues: List[dict], ctx: FilterContext) -> List[dict]:
    """过滤属于技巧正常表现的问题（从知识库豁免规则加载）。

    改进：仅当描述文本与豁免关键词有显著匹配时才过滤——
    需要 ≥2 个关键词命中，或者单个长关键词覆盖 >30% 的文本长度。
    避免单个短关键词（如"手指平"）误杀包含"手指平放是不对的"这类反向描述。
    """
    if not ctx.technique_allowed_keywords:
        return issues
    result = []
    for h in issues:
        iss = h.get("问题", "") + h.get("description", "")
        if not iss:
            result.append(h)
            continue

        # 检查反向关键词：如果问题描述中包含否定信号，不豁免
        if any(neg in iss for neg in ["不应", "不能", "不要", "错误", "不对", "不该"]):
            result.append(h)
            continue

        matched = [kw for kw in ctx.technique_allowed_keywords if kw in iss]
        if not matched:
            result.append(h)
            continue

        # 计算关键词覆盖率
        matched_chars = sum(len(kw) for kw in matched)
        coverage = matched_chars / max(len(iss), 1)

        # 豁免条件：≥2个关键词命中，或长关键词覆盖 >30%
        should_filter = len(matched) >= 2 or (len(matched) == 1 and coverage > 0.30)
        if should_filter:
            logger.info(f"technique_allowed filtered: {iss[:60]} (matched={len(matched)}, coverage={coverage:.0%})")
            continue
        else:
            logger.info(f"technique_allowed keep (insufficient match): {iss[:60]} (matched={len(matched)}, coverage={coverage:.0%})")
        result.append(h)
    return result


def filter_harmonic_segments(issues: List[dict], ctx: FilterContext) -> List[dict]:
    """过滤落在已确认泛音时间窗内的手型问题。

    泛音（自然泛音/人工泛音）的手型与正常按弦完全不同——
    轻触品丝 vs 用力按弦，手指近乎平行 vs 拱形站立。
    如果音频已确认某时段是泛音，该时段内仅过滤泛音特有的手型偏差，
    通用手型问题（手腕、拇指、紧张等）无论什么技巧都应该报告。
    """
    harmonics_segs = [
        s for s in ctx.technique_segments
        if s.get("technique_id", "") in ("natural-harmonics", "artificial-harmonics")
        and s.get("confidence", 0) >= 0.5
    ]
    if not harmonics_segs:
        return issues

    # 这些是泛音特有的手型特征，在泛音段可以豁免
    HARMONICS_SAFE_KEYWORDS = [
        "轻触", "指尖轻触", "手指平行", "手指放平", "指腹轻触",
        "轻放在弦上", "不要按实", "触弦", "泛音点",
    ]
    # 这些是通用手型问题，无论在什么技巧段都必须报告
    ALWAYS_REPORT_KEYWORDS = [
        "手腕", "拇指", "紧张", "僵硬", "用力", "捏", "扣",
        "塌", "飞", "翘", "挤", "缩", "姿势", "虎口",
    ]

    TOLERANCE = 0.3
    result = []
    for h in issues:
        t = float(h.get("时间", h.get("timestamp", 0)))
        in_harmonics = any(
            s["start_time"] - TOLERANCE <= t <= s["end_time"] + TOLERANCE
            for s in harmonics_segs
        )
        if not in_harmonics:
            result.append(h)
            continue

        iss = h.get("问题", "") + h.get("description", "")
        # 通用手型问题不因泛音而豁免
        if any(kw in iss for kw in ALWAYS_REPORT_KEYWORDS):
            result.append(h)
            continue
        # 只有在泛音安全关键词匹配时才豁免
        if any(kw in iss for kw in HARMONICS_SAFE_KEYWORDS):
            logger.info(f"harmonics filter (safe): {iss[:50]} at t={t:.1f}s")
            continue
        # 没有匹配到泛音安全关键词 → 保留问题
        result.append(h)
    return result


def filter_right_hand_false_positives(issues: List[dict], ctx: FilterContext) -> List[dict]:
    """右手技巧模式：过滤左手按弦类误报"""
    if not ctx.is_right_hand_tech:
        return issues

    LEFT_FRETTING_KW = [
        "手指较平", "手指没立", "指尖没有垂直", "按弦", "趴下去", "塌指", "PIP",
        "食指按", "中指按", "无名指按", "小指按", "食指放平", "食指趴",
        "没立起来", "指尖没有", "手指趴", "食指平", "放平", "指腹按弦",
        "关节角度", "第一个关节放平", "手指放平", "手指贴", "指腹按", "不是指尖",
        "太弯了", "太直了",
    ]

    result = []
    for h in issues:
        iss = h.get("问题", "") + h.get("description", "")
        who = h.get("哪只手", h.get("handedness", ""))
        # 只过滤标记为左手的问题，右手的问题保留
        if "右手" in who or "右手" in iss:
            result.append(h)
            continue
        if any(kw in iss for kw in LEFT_FRETTING_KW):
            logger.info(f"right_hand_false_positive: {who} {iss[:50]}")
            continue
        result.append(h)
    return result


def filter_non_right_mode_right_issues(issues: List[dict], ctx: FilterContext) -> List[dict]:
    """非右手技巧模式：过滤标记为右手的问题（只有左手出镜时 MP 误判 handedness）"""
    if ctx.is_right_hand_tech:
        return issues
    return [
        h for h in issues
        if "右手" not in h.get("哪只手", "") and "右手" not in h.get("问题", "")
        and "右手" not in h.get("handedness", "")
    ]


def dedup_by_content(issues: List[dict]) -> List[dict]:
    """内容去重：同一只手同一个问题同一秒内只保留一条（保留不同时间点的检测）"""
    seen = set()
    result = []
    for h in issues:
        who = h.get("哪只手", h.get("handedness", ""))
        iss = h.get("问题", h.get("description", ""))
        t = h.get("时间", h.get("timestamp", 0))
        bucket = int(t) if t else 0
        key = f"{who}|{iss[:60]}|{bucket}"
        if key not in seen:
            seen.add(key)
            result.append(h)
    return result


def _unique_list(items):
    """去重但保持顺序。"""
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _classify_issue(issue_text: str):
    """将单条问题描述归类为 (category, finger) 元组。

    category 用于分组：同 category 的问题可在同一时间合并，不同 category 的保持独立。
    """
    p_clean = issue_text.replace("💡", "").replace("⚠️", "").strip()
    finger = ""
    for f in ["食指", "中指", "无名指", "小指", "拇指", "手腕"]:
        if f in p_clean:
            finger = f
            break

    if finger == "手腕":
        if "扣" in p_clean or "塌" in p_clean or "凹" in p_clean:
            return ("wrist_collapsed", "手腕")
        if "太直" in p_clean or "太直" in p_clean:
            return ("wrist_straight", "手腕")
        if "太弯" in p_clean or "弯得" in p_clean:
            return ("wrist_bent", "手腕")
        return ("wrist_other", "手腕")

    if finger == "拇指":
        if "捏" in p_clean or "紧" in p_clean:
            return ("thumb_tense", "拇指")
        if "太直" in p_clean:
            return ("thumb_straight", "拇指")
        if "高" in p_clean or "包覆" in p_clean:
            return ("thumb_high", "拇指")
        return ("thumb_other", "拇指")

    if "太直" in p_clean or "太竖直" in p_clean:
        return ("finger_straight", finger)
    if "太弯" in p_clean:
        return ("finger_bent", finger)
    if "趴" in p_clean or "塌" in p_clean:
        return ("finger_flat", finger)
    if "紧张" in p_clean or "放松" in p_clean:
        return ("finger_tense", finger)
    return ("other", finger)


def _summarize_problems(problems: List[str]) -> str:
    """将同一类别的多个手型问题归纳为简洁的整体描述。"""
    if len(problems) <= 1:
        return problems[0] if problems else ""

    has_wrist = False
    wrist_straight = False
    wrist_bent = False
    has_thumb = False
    too_straight = []
    too_bent = []
    too_flat = []
    too_tense = []
    other = []

    for p in problems:
        p_clean = p.replace("💡", "").replace("⚠️", "").strip()
        finger = ""
        for f in ["食指", "中指", "无名指", "小指", "拇指", "手腕"]:
            if f in p_clean:
                finger = f
                break

        if finger == "手腕":
            has_wrist = True
            if "太直" in p_clean or "太直" in p_clean:
                wrist_straight = True
            elif "太弯" in p_clean or "弯得" in p_clean:
                wrist_bent = True
            continue

        if finger == "拇指":
            has_thumb = True
            if "捏" in p_clean or "紧" in p_clean:
                too_tense.append(finger)
            elif "太直" in p_clean:
                too_straight.append(finger)
            continue

        if "太直" in p_clean or "太竖直" in p_clean:
            too_straight.append(finger)
        elif "太弯" in p_clean:
            too_bent.append(finger)
        elif "趴" in p_clean or "塌" in p_clean:
            too_flat.append(finger)
        elif "紧张" in p_clean or "放松" in p_clean:
            too_tense.append(finger)
        else:
            other.append(p_clean[:40])

    parts = []
    finger_issues = []

    if too_straight and too_bent:
        overlap = [f for f in too_straight if f in too_bent]
        if overlap:
            if len(too_straight) >= len(too_bent):
                too_bent = [f for f in too_bent if f not in overlap]
            else:
                too_straight = [f for f in too_straight if f not in overlap]
        if too_straight and too_bent:
            finger_issues.append("部分手指太直、部分太弯")
        elif too_straight:
            n_s = len(too_straight)
            if n_s >= 3:
                finger_issues.append("大部分手指站得太直")
            else:
                names = "、".join(_unique_list(too_straight)[:2])
                finger_issues.append(f"{names}站得太直")
        elif too_bent:
            n_b = len(too_bent)
            if n_b >= 3:
                finger_issues.append("大部分手指太弯")
            else:
                names = "、".join(_unique_list(too_bent)[:2])
                finger_issues.append(f"{names}太弯了")
    elif too_straight:
        n_s = len(too_straight)
        if n_s >= 3:
            finger_issues.append("大部分手指站得太直")
        else:
            names = "、".join(_unique_list(too_straight)[:2])
            finger_issues.append(f"{names}站得太直")
    elif too_bent:
        n_b = len(too_bent)
        if n_b >= 3:
            finger_issues.append("大部分手指太弯")
        else:
            names = "、".join(_unique_list(too_bent)[:2])
            finger_issues.append(f"{names}太弯了")

    if too_flat:
        names = "、".join(_unique_list(too_flat)[:2])
        finger_issues.append(f"{names}有点趴")

    if too_tense:
        unique_tense = _unique_list(too_tense)
        names = "、".join(unique_tense[:3])
        if len(unique_tense) >= 3:
            finger_issues.append("多根手指紧张")
        else:
            finger_issues.append(f"{names}太紧张")

    # 输出：拇指/手腕一组，手指一组（最多2条）
    output_parts = []
    wrist_thumb_parts = []

    if len(finger_issues) >= 2:
        output_parts.append("手指：" + "；".join(finger_issues))
    elif finger_issues:
        output_parts.append("手指：" + finger_issues[0])

    if has_wrist:
        if wrist_straight and not wrist_bent:
            wrist_thumb_parts.append("手腕太直")
        elif wrist_bent and not wrist_straight:
            wrist_thumb_parts.append("手腕太弯")
        else:
            wrist_thumb_parts.append("手腕往里扣了")

    if has_thumb and not too_tense:
        wrist_thumb_parts.append("拇指位置不对")

    if wrist_thumb_parts:
        output_parts.append("拇指/手腕：" + "，".join(wrist_thumb_parts))

    if not output_parts:
        return "；".join(problems[:3])

    return "；".join(output_parts[:2])


def dedup_by_time(issues: List[dict]) -> List[dict]:
    """时间去重：同一类问题在同一秒内合并，不同类问题保留独立条目。

    分组策略：
    - 同手指+同问题类型 → 合并（保留最高置信度的那条）
    - 不同手指 → 归类后合并（如：食指太直+中指太直 → 食指、中指站得太直）
    - 不同问题类别（手指太直 vs 手腕内扣 vs 手指太弯）→ 保留独立条目
    """
    # 过滤 t<0.3s 的调整期问题（还没开始弹奏）
    issues = [h for h in issues if h.get("时间", h.get("timestamp", 0)) >= 0.3]
    if not issues:
        return issues

    # 按时间桶分组
    time_buckets = {}
    for h in issues:
        t = h.get("时间", h.get("timestamp", 0))
        bucket = int(t) if t else 0
        if bucket not in time_buckets:
            time_buckets[bucket] = []
        time_buckets[bucket].append(h)

    result = []
    for bucket in sorted(time_buckets.keys()):
        bucket_issues = time_buckets[bucket]

        # 在时间桶内按问题类别再分组
        cat_groups = {}  # category → list of issues
        for h in bucket_issues:
            text = h.get("问题", h.get("issue", ""))
            cat, _ = _classify_issue(text)
            if cat not in cat_groups:
                cat_groups[cat] = []
            cat_groups[cat].append(h)

        # 同类别内去重+合并
        for cat, cat_issues in cat_groups.items():
            cat_issues.sort(key=lambda x: x.get("置信度", x.get("handedness_confidence", 0.0)), reverse=True)
            base = dict(cat_issues[0])
            all_problems = []
            seen_problem = set()
            for h in cat_issues:
                text = h.get("问题", h.get("issue", ""))
                if text and text not in seen_problem:
                    seen_problem.add(text)
                    all_problems.append(text)
            if len(all_problems) > 1:
                base["问题"] = _summarize_problems(all_problems)
            result.append(base)

    return result


def merge_vision_issues(issues: List[dict], vision_issues: List[dict]) -> List[dict]:
    """合并 Vision AI 问题到已有列表，然后时间去重"""
    if not vision_issues:
        return issues
    combined = list(issues) + list(vision_issues)
    return dedup_by_time(combined)


def correct_hand_labels(issues: List[dict], ctx: FilterContext) -> List[dict]:
    """用 Vision AI 的 handedness 修正 MediaPipe 不可靠的手标签"""
    if not ctx.vision_handedness or ctx.vision_handedness not in ("左手", "右手"):
        return issues

    for h in issues:
        who = h.get("哪只手", "")
        mp_conf = h.get("置信度", 1.0)
        disagrees = (
            (ctx.vision_handedness == "右手" and "左手" in who) or
            (ctx.vision_handedness == "左手" and "右手" in who)
        )
        if disagrees and mp_conf < 0.7:
            corrected = who.replace("左手", "右手") if "左手" in who else who.replace("右手", "左手")
            logger.info(f"hand label fix (conf={mp_conf}): {who} → {corrected}")
            h["哪只手"] = corrected
        elif disagrees:
            logger.info(f"hand label unchanged (conf={mp_conf}>=0.7, vision={ctx.vision_handedness}): {who}")

    return issues


# ====== 诊断优先级与因果合并引擎 ======

# 问题描述关键词 → 知识库问题 ID 映射
# 按匹配优先级排序：更具体的关键词排在前面
_PROBLEM_KEYWORD_MAP = [
    ("thumb-gripping-too-tight", ["拇指.*捏.*紧", "虎口.*闭合", "拇指.*力度.*大", "拇指.*死.*捏", "拇指.*锁死"], "left"),
    ("thumb-too-high", ["拇指.*高", "拇指.*包覆", "拇指.*琴颈上方伸出", "拇指.*超出.*琴颈"], "left"),
    ("thumb-no-support", ["拇指.*无支撑", "拇指.*未接触", "拇指.*悬空", "拇指.*没.*靠"], "left"),
    ("collapsed-wrist", ["手腕.*往里扣", "手腕.*内扣", "手腕.*塌", "手腕.*<.*140", "手腕.*角度.*小"], "left"),
    ("flat-fingers", ["手指.*放平", "指腹按弦", "手指.*平", "PIP.*>.*155", "第一个关节放平", "手指趴", "没立起来"], "left"),
    ("too-vertical-fingers", ["手指.*竖直", "手指.*垂直", "站得太直", "MCP.*<.*50", "MCP.*过小", "太直.*戳", "筷子"], "left"),
    ("fingers-bunched-together", ["手指.*缩.*一起", "手指.*挤", "手指.*间距.*小", "手指.*打不开", "指尖.*距离.*<"], "left"),
    ("palm-perpendicular", ["手掌.*垂直", "手掌.*贴.*琴颈", "手掌.*平行.*指板"], "left"),
    ("pinky-flying", ["小指.*飞", "小指.*翘", "小指.*PIP.*>", "小指.*伸直"], "left"),
    ("excessive-pressure", ["按弦.*力度.*大", "过度.*用力", "压力.*过大", "捏.*太.*用力"], "left"),
    ("string-buzzing", ["杂音", "闷音", "打品", "buzz", "蹭.*品"], "left"),
    ("right-hand-tension", ["右手.*紧张", "右手.*僵硬", "右手.*锁死"], "right"),
    ("right-hand-off-soundhole", ["右手.*不在.*音孔", "右手.*偏离", "右手.*位置.*不对"], "right"),
    ("right-hand-fingers-not-curved", ["右手.*手指.*直", "右手.*PIP.*>", "右手.*不.*弯曲", "右手.*没有.*弯曲", "半握拳"], "right"),
    ("picking-uneven", ["拨弦.*不均", "力度.*不均.*右手", "右手.*力度.*不均"], "right"),
    ("strumming-rhythm", ["扫弦.*不稳", "扫弦.*节奏", "右手.*节奏.*不稳"], "right"),
    ("sitting-posture", ["坐姿", "持琴.*姿势", "身体.*弯曲", "驼背"], "both"),
]

# 因果合并规则: (root_cause_id, [symptom_ids_that_merge_into_root])
_CAUSAL_MERGE_RULES = [
    ("collapsed-wrist", ["flat-fingers", "fingers-bunched-together",
                          "palm-perpendicular", "excessive-pressure", "string-buzzing"]),
    ("thumb-too-high", ["flat-fingers", "excessive-pressure", "pinky-flying"]),
    ("thumb-gripping-too-tight", ["flat-fingers", "excessive-pressure", "pinky-flying",
                                   "fingers-bunched-together", "string-buzzing"]),
    ("thumb-no-support", ["flat-fingers", "excessive-pressure", "too-vertical-fingers"]),
    ("sitting-posture", ["collapsed-wrist", "right-hand-off-soundhole"]),
    ("right-hand-tension", ["picking-uneven", "strumming-rhythm", "right-hand-fingers-not-curved"]),
]

# 问题优先级 (数字越小优先级越高)
_PROBLEM_PRIORITY = {
    "thumb-gripping-too-tight": 0,
    "string-buzzing": 0,
    "sitting-posture": 1,
    "collapsed-wrist": 1,
    "thumb-too-high": 2,
    "thumb-no-support": 2,
    "right-hand-tension": 2,
    "right-hand-off-soundhole": 2,
    "flat-fingers": 3,
    "too-vertical-fingers": 2,  # 用户要求作为重点检测项
    "fingers-bunched-together": 3,
    "palm-perpendicular": 3,
    "pinky-flying": 3,
    "right-hand-fingers-not-curved": 3,
    "excessive-pressure": 4,
    "picking-uneven": 4,
    "strumming-rhythm": 4,
}

_MAX_ISSUES_BY_LEVEL = {"beginner": 5, "intermediate": 8, "advanced": 12}


def _map_to_problem_id(issue_desc: str, hand_side: str) -> Optional[str]:
    import re
    desc = issue_desc.replace("\n", " ").strip()
    if not desc:
        return None
    best_match = None
    best_score = 0
    for pid, keywords, side in _PROBLEM_KEYWORD_MAP:
        if side == "left" and "右手" in hand_side:
            continue
        if side == "right" and "左手" in hand_side:
            continue
        score = 0
        for kw in keywords:
            if re.search(kw, desc):
                score += 1
        if score > best_score:
            best_score = score
            best_match = pid
    return best_match if best_score > 0 else None


def _apply_causal_merge(problem_ids: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    merged = dict(problem_ids)
    for root_id, symptom_ids in _CAUSAL_MERGE_RULES:
        if root_id not in merged:
            continue
        for sym_id in symptom_ids:
            if sym_id in merged and sym_id != root_id:
                merged[root_id].extend(merged.pop(sym_id))
                logger.info(f"causal_merge: {sym_id} -> {root_id}")
    return merged


def _sort_by_priority(problem_ids: Dict[str, List[dict]]) -> List[tuple]:
    def sort_key(item):
        pid, issues = item
        priority = _PROBLEM_PRIORITY.get(pid, 5)
        max_conf = max((i.get("置信度", 0.5) for i in issues), default=0.5)
        return (priority, -max_conf)
    return sorted(problem_ids.items(), key=sort_key)


def prioritize_issues(issues: List[dict], ctx: FilterContext) -> List[dict]:
    """诊断优先级与因果合并。

    1. 将 issue 描述映射为问题 ID
    2. 同 ID 去重（保留置信度最高的2条）
    3. 按因果链合并症状 -> 根因
    4. 按优先级排序
    5. 按用户等级限制输出数量（Tier 0 致命问题不限数量）
    6. 未映射的 issue 保持原样放在最后
    """
    if len(issues) <= 1:
        return issues

    # 1. 映射到问题 ID
    mapped: Dict[str, List[dict]] = {}
    unmapped: List[dict] = []

    for h in issues:
        desc = h.get("问题", h.get("description", ""))
        hand = h.get("哪只手", h.get("handedness", ""))
        pid = _map_to_problem_id(desc, hand)
        if pid:
            if pid not in mapped:
                mapped[pid] = []
            mapped[pid].append(h)
        else:
            unmapped.append(h)

    if not mapped:
        return issues

    # 2. 同 ID 去重：保留所有实例（时间点不同），不丢弃任何检测结果
    for pid in mapped:
        mapped[pid].sort(key=lambda x: x.get("置信度", 0.5), reverse=True)

    # 3. 因果合并
    merged = _apply_causal_merge(mapped)

    # 4. 排序
    sorted_problems = _sort_by_priority(merged)

    # 5. 构建输出
    max_issues = _MAX_ISSUES_BY_LEVEL.get(ctx.user_level, 2)
    result = []
    tier0_count = 0

    for pid, orig_issues in sorted_problems:
        prio = _PROBLEM_PRIORITY.get(pid, 5)
        if prio == 0:
            tier0_count += 1
        if prio > 0 and len(result) - tier0_count >= max_issues:
            logger.info(f"prioritize: {pid} dropped (max={max_issues})")
            continue

        orig_issues.sort(key=lambda x: x.get("置信度", 0.5), reverse=True)
        main = dict(orig_issues[0])
        main["诊断ID"] = pid
        main["诊断优先级"] = prio

        # 收集所有检测到的时间点（保留完整时间线）
        all_times = sorted(set(
            h.get("时间", h.get("timestamp", 0)) for h in orig_issues
        ))
        reported_time = all_times[0] if all_times else 0
        main["时间"] = reported_time
        main["所有检测时间点"] = all_times
        main["出现次数"] = len(all_times)

        # 选择最佳截图：优先选取时间最接近报告时间的截图，而非置信度最高的
        if len(orig_issues) > 1:
            best_snap = ""
            best_snap_dist = float("inf")
            for o in orig_issues:
                snap = o.get("截图", "")
                ot = o.get("时间", o.get("timestamp", 0))
                if snap and abs(ot - reported_time) < best_snap_dist:
                    best_snap_dist = abs(ot - reported_time)
                    best_snap = snap
            if best_snap:
                main["截图"] = best_snap

        if len(orig_issues) > 1:
            sym_descs = [
                o.get("问题", o.get("description", ""))[:40]
                for o in orig_issues[1:]
            ]
            sym_descs = _unique_list(sym_descs)
            main["关联表现"] = sym_descs[:5]
        main["来源数"] = len(orig_issues)

        result.append(main)

    # 6. 未映射的放在最后
    for h in unmapped:
        h["诊断ID"] = None
        h["诊断优先级"] = 5
        h["来源数"] = 1
        if len(result) - tier0_count < max_issues + 1:
            result.append(h)

    logger.info(
        f"prioritize: {len(issues)} -> {len(result)} issues "
        f"(mapped={len(mapped)}, merged={len(merged)})"
    )
    return result


def flatten_issues(issues: List[dict]) -> List[dict]:
    """将嵌套的 issues 列表格式展平为统一格式 {哪只手, 问题, 时间, 置信度}"""
    result = []
    seen = set()
    for h in issues:
        who = h.get("哪只手", h.get("handedness", ""))
        t = h.get("时间", h.get("timestamp", 0))
        conf = h.get("置信度", h.get("handedness_confidence", 0.0))

        if isinstance(h.get("issues"), list):
            for sub_iss in h["issues"]:
                key = f"{who}|{sub_iss}|{t}"
                if key not in seen:
                    seen.add(key)
                    result.append({
                        "哪只手": who, "问题": sub_iss, "时间": t, "置信度": conf,
                    })
        else:
            iss = h.get("问题", h.get("description", ""))
            if iss:
                key = f"{who}|{iss}|{t}"
                if key not in seen:
                    seen.add(key)
                    result.append(h)
    return result


# ====== 预设管道 ======

def create_freeplay_pipeline() -> HandIssueFilter:
    """自由演奏模式的过滤管道"""
    f = HandIssueFilter()
    f.add_stage("flatten", lambda issues, ctx: flatten_issues(issues))
    f.add_stage("pre_music", filter_pre_music)
    f.add_stage("harmonics", filter_harmonic_segments)
    f.add_stage("normal", filter_normal_issues)
    f.add_stage("content_dedup", lambda issues, ctx: dedup_by_content(issues))
    f.add_stage("time_dedup", lambda issues, ctx: dedup_by_time(issues))
    f.add_stage("technique_allowed", filter_technique_allowed)
    f.add_stage("right_hand_fp", filter_right_hand_false_positives)
    f.add_stage("correct_labels", correct_hand_labels)
    f.add_stage("prioritize", prioritize_issues)
    return f


def create_chord_check_pipeline() -> HandIssueFilter:
    """和弦/技巧检查模式的过滤管道"""
    f = HandIssueFilter()
    f.add_stage("flatten", lambda issues, ctx: flatten_issues(issues))
    f.add_stage("early_frames", filter_early_frames)
    f.add_stage("harmonics", filter_harmonic_segments)
    f.add_stage("normal", filter_normal_issues)
    f.add_stage("technique_allowed", filter_technique_allowed)
    f.add_stage("right_hand_fp", filter_right_hand_false_positives)
    f.add_stage("non_right_mode", filter_non_right_mode_right_issues)
    f.add_stage("content_dedup", lambda issues, ctx: dedup_by_content(issues))
    f.add_stage("time_dedup", lambda issues, ctx: dedup_by_time(issues))
    f.add_stage("prioritize", prioritize_issues)
    return f
