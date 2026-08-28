"""和弦对转移难度表（Batch-1 Item 1，FretboardFlow 风格）。

从 KB 手指位置预计算所有和弦对的转移难度，用于在报告中标注「难点切换」
并驱动练习建议。纯计算模块，无副作用，可单测。

难度因子（对和弦对 A→B）：
1. 锚点共享：同指同弦同品的共享手指数（越多越简单，负分）
2. 逐指移位：公共手指 |Δfret| + |Δstring| 之和
3. 新增/释放手指：只出现在一侧的手指数
4. 跨度变化：max(fret)-min(fret) 的差值
5. 横按惩罚：任一侧为横按和弦（≥ 多弦同品覆盖）
6. 开放/按弦切换：两侧空弦数差

产出：{(A, B): {"score": 0-10, "anchor_fingers": [...], "factors": {...}}}
"""
import functools
import os
import re

from chord_analyzer import ChordAnalyzer


# 难度合成系数（经验式，可在标定后调整）
_ANCHOR_BONUS = 1.5      # 每个锚点手指减难度
_SHIFT_COST = 0.4        # 每单位 (|Δfret|+|Δstring|) 加难度
_NEW_COST = 1.5          # 每根新增手指
_RELEASE_COST = 0.6      # 每根释放手指
_SPAN_COST = 0.3         # 跨度变化每 1 品
_BARRE_PENALTY = 2.0     # 大横按（覆盖 ≥4 弦）惩罚
_SMALL_BARRE_PENALTY = 0.8  # 小横按（覆盖 2-3 弦）惩罚
_OPEN_COST = 0.3         # 空弦数差每 1 弦


def _extract_barre(fd: dict):
    """从手指数据提取横按覆盖范围 (min弦, max弦)，非横按返回 None。

    KB 里横按有两种表示：
    1. 范围格式 string="1-5"/"5-1"（Bm/Bm7/Dm7）
    2. 单弦 + tip_contact/desc 含「横按」关键词（B/F，如「指侧横按5弦」「横按全6弦」）
    """
    s = fd.get("string")
    if isinstance(s, str) and "-" in s:
        try:
            lo, hi = sorted(int(x) for x in s.split("-"))
            return (lo, hi)
        except ValueError:
            pass
    text = " ".join(str(fd.get(k, "")) for k in ("tip_contact", "desc", "note"))
    if "横按" in text:
        # 关键词横按（B/F 均为 5-6 弦大横按），默认覆盖①-⑤弦
        return (1, 5)
    return None


@functools.lru_cache(maxsize=1)
def load_chords_from_kb(knowledge_dir: str = None) -> dict:
    """加载所有和弦的 strings + fingers + barre 数据（含横按和弦）。

    横按手指在 KB 里 string 形如 "1-5"/"5-1"，`ChordAnalyzer._load_chord_templates_with_finger_data`
    用 int() 解析会抛异常并整和弦跳过（Bm/Bm7/Dm7 因此丢失）。这里单独处理：
    横按范围解析成 barre_fingers[手指] = (min弦, max弦)，fingers 里用 min 弦作代表。

    Returns:
        {chord_id: {"strings": [str]*6, "fingers": {手指: (弦, 品)},
                    "barre_fingers": {手指: (min弦, max弦)}, "difficulty": str}}
    """
    if knowledge_dir is None:
        knowledge_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "knowledge", "chords"
        )
    knowledge_dir = os.path.normpath(knowledge_dir)
    chords = {}
    if not os.path.isdir(knowledge_dir):
        return chords

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
            fm = ChordAnalyzer._parse_frontmatter(parts[1])

            chord_id = fm.get("id", fname[:-3])
            if "chord-" in chord_id:
                chord_id = chord_id.replace("chord-", "")
            strings_raw = fm.get("strings", [])
            if not strings_raw or len(strings_raw) != 6:
                continue

            fingers_raw = fm.get("fingers", {})
            if isinstance(fingers_raw, str):
                fingers_raw = ChordAnalyzer._parse_inline_dict(fingers_raw)

            finger_map = {}
            barre_fingers = {}
            if isinstance(fingers_raw, dict):
                for finger_name, fd in fingers_raw.items():
                    if isinstance(fd, str):
                        fd = ChordAnalyzer._parse_inline_dict(fd)
                    if not isinstance(fd, dict):
                        continue
                    s = fd.get("string")
                    fret = fd.get("fret")
                    if fret is None:
                        continue
                    fret = int(fret)
                    barre = _extract_barre(fd)
                    if barre is not None:
                        barre_fingers[finger_name] = barre
                        finger_map[finger_name] = (barre[0], fret)
                    elif s is not None:
                        finger_map[finger_name] = (int(s), fret)

            if not finger_map:
                continue

            chords[chord_id] = {
                "strings": [str(x) for x in strings_raw],
                "fingers": finger_map,
                "barre_fingers": barre_fingers,
                "difficulty": fm.get("difficulty", ""),
            }
        except Exception:
            continue
    return chords


def lookup_chord(chords: dict, chord_id: str):
    """按 chord_id 查和弦数据，大小写不敏感兜底（如 fmaj7↔Fmaj7）。"""
    if not chord_id:
        return None
    if chord_id in chords:
        return chords[chord_id]
    cid = chord_id.casefold()
    for k, v in chords.items():
        if k.casefold() == cid:
            return v
    return None


def _finger_position(finger, chord):
    """返回手指在 chord 中的 (弦, 品)；不存在返回 None。"""
    return chord["fingers"].get(finger)


def _span(chord):
    """按弦手指的品位跨度（max fret - min fret）。"""
    frets = [f for _, f in chord["fingers"].values()]
    if not frets:
        return 0.0
    return float(max(frets) - min(frets))


def _open_count(chord):
    """空弦（strings 里 "0"）的数量。"""
    return sum(1 for s in chord["strings"] if s == "0")


def _barre_penalty(chord):
    """横按惩罚：大横按（覆盖 ≥4 弦）2.0，小横按（2-3 弦）0.8。"""
    max_span = 0
    for lo, hi in chord["barre_fingers"].values():
        max_span = max(max_span, hi - lo + 1)
    if max_span >= 4:
        return _BARRE_PENALTY
    if max_span >= 2:
        return _SMALL_BARRE_PENALTY
    return 0.0


def compute_transition(chord_a: dict, chord_b: dict) -> dict:
    """计算单对和弦 A→B 的转移难度。"""
    fingers_a = chord_a["fingers"]
    fingers_b = chord_b["fingers"]
    all_fingers = set(fingers_a) | set(fingers_b)

    anchor_fingers = []
    shift_cost = 0.0
    new_count = 0
    release_count = 0

    for finger in all_fingers:
        pos_a = fingers_a.get(finger)
        pos_b = fingers_b.get(finger)
        if pos_a is not None and pos_b is not None:
            if pos_a == pos_b:
                anchor_fingers.append(finger)
            else:
                shift_cost += abs(pos_b[0] - pos_a[0]) + abs(pos_b[1] - pos_a[1])
        elif pos_b is not None:
            new_count += 1
        else:
            release_count += 1

    span_change = abs(_span(chord_a) - _span(chord_b))
    barre_penalty = max(_barre_penalty(chord_a), _barre_penalty(chord_b))
    open_shift = abs(_open_count(chord_a) - _open_count(chord_b))

    raw = (
        -_ANCHOR_BONUS * len(anchor_fingers)
        + _SHIFT_COST * shift_cost
        + _NEW_COST * new_count
        + _RELEASE_COST * release_count
        + _SPAN_COST * span_change
        + barre_penalty
        + _OPEN_COST * open_shift
    )
    score = round(max(0.0, min(10.0, raw)), 2)

    return {
        "score": score,
        "anchor_fingers": sorted(anchor_fingers),
        "factors": {
            "anchor_count": len(anchor_fingers),
            "shift_cost": round(shift_cost, 2),
            "new_count": new_count,
            "release_count": release_count,
            "span_change": round(span_change, 2),
            "barre_penalty": round(barre_penalty, 2),
            "open_shift": open_shift,
        },
    }


def build_difficulty_table(chords: dict) -> dict:
    """预计算所有有序和弦对的转移难度。"""
    ids = sorted(chords.keys())
    table = {}
    for a in ids:
        for b in ids:
            if a == b:
                continue
            table[(a, b)] = compute_transition(chords[a], chords[b])
    return table
