"""
报告后处理：强制章节内容一致性。

统一的 force_section() 替代原来三个几乎相同的 _force_hand_section,
_force_audio_section, _force_audio_section_nosound 函数。
"""

import re as _re
from typing import Optional

# 章节配置：每个 section 的标题匹配模式和边界匹配模式
SECTION_CONFIG = {
    "hand": {
        "header_patterns": [
            r'^(?:##\s*)?(?:✋|🤚|🖐|🙌|👋)?\s*(?:手型|手指|手部|手型效率|手指检查)',
            r'^(?:##\s*)?✋',
        ],
        "boundary_patterns": [
            r'^(?:##\s*)?(?:🎵|📝|🎯|💡|总评|音频|练习|作业|精练|建议|小作业|本周|耳朵)',
            r'^---',
        ],
    },
    "audio": {
        "header_patterns": [
            r'^(?:##\s*)?(?:🎵|🎼|🎶)?\s*(?:音频|耳朵检查|音频细节|音频分析)',
        ],
        "boundary_patterns": [
            r'^(?:##\s*)?(?:✋|🤚|🖐|🙌|👋|📝|🎯|💡|总评|手型|手指|手部|练习|作业|精练|建议|小作业|本周|耳朵)',
            r'^---',
        ],
    },
}


def _is_section_header(line: str, patterns: list) -> bool:
    stripped = line.strip()
    return any(bool(_re.match(p, stripped)) for p in patterns)


def force_section(md: str, section: str, replacement: str) -> str:
    """在 markdown 报告中替换指定章节的内容。

    Args:
        md: 原始 markdown 报告
        section: 章节名 ("hand" 或 "audio")
        replacement: 替换内容（不含章节标题，force_section 会自动处理格式）

    Returns:
        处理后的 markdown
    """
    config = SECTION_CONFIG[section]
    lines = md.split('\n')
    result_lines = []
    in_section = False
    section_added = False

    for line in lines:
        stripped = line.strip()

        # 进入目标章节
        if _is_section_header(stripped, config["header_patterns"]) and not in_section:
            in_section = True
            continue  # 跳过原标题行

        if in_section:
            # 遇到下一个章节 → 结束目标章节，插入替换内容
            if _is_section_header(stripped, config["boundary_patterns"]):
                result_lines.append(replacement.strip())
                section_added = True
                in_section = False
                result_lines.append(line)
            # 否则跳过目标章节的内容
        else:
            result_lines.append(line)

    # 目标章节是文末最后一个章节
    if in_section and not section_added:
        result_lines.append(replacement.strip())
        section_added = True

    # 全文没有目标章节，追加到末尾
    if not section_added:
        if result_lines and result_lines[-1] != '':
            result_lines.append('')
        result_lines.append(replacement.strip())

    return '\n'.join(result_lines)


def force_hand_section(md: str, status: str, score: int = None) -> str:
    """根据手型状态替换手型章节"""
    if status == "nHand":
        replacement = "## ✋ 手型\n\n> 📷 本次未捕捉到手部画面，下次录制时请确保手部在镜头内清晰可见\n"
    elif status == "ok":
        if score is not None and score < 90:
            replacement = (
                "## ✋ 手型\n\n"
                "> ✅ 手型检测正常，没有发现明显的手型问题\n"
                ">\n"
                "> 💡 但手型得分还有提升空间（当前手型评分未达优秀线），建议：\n"
                "> 1. 检查每根手指是否用指尖（而非指腹）按弦\n"
                "> 2. 拇指是否自然搭在琴颈上方，不要捏太紧\n"
                "> 3. 手腕保持自然弧度，不要往里扣或往外拱\n"
                "> 4. 手指尽量靠近品柱但不碰到品柱，省力又干净\n"
            )
        else:
            replacement = "## ✋ 手型\n\n> ✅ 手部检测正常，手型姿势规范，继续保持\n"
    elif status == "ai_found":
        # AI 在视频中看到了引擎未检测到的手型细节问题 → 保留 AI 的分析
        return md
    else:
        # Unknown status: treat as no hand data to avoid false "OK" messages
        replacement = "## ✋ 手型\n\n> ⚠️ 手型检测状态未知，请确保手部在镜头内清晰可见\n"
        return force_section(md, "hand", replacement)
    return force_section(md, "hand", replacement)


def force_audio_section(md: str, audio_errors: list, audio_diagnosis=None) -> str:
    """根据音频错误列表替换音频章节。"""
    if not audio_errors:
        replacement = "## 🎵 音频\n\n> ✅ 音准节奏整体良好，未检测到明显问题\n"
    else:
        icon_map = {'wrong_note': '🎵', 'rhythm_fault': '⏱', 'transition_gap': '🔄', 'extra_note': '➕', 'missed_note': '➖', 'dead_note': '🔇'}
        lines = []
        for e in audio_errors:
            detail = e.get('detail') or e.get('msg', '')
            t = e.get('time', 0)
            icon = icon_map.get(e.get('type', ''), '🔍')
            lines.append(f"> **{icon} {detail}**（{t:.1f}秒）")
        replacement = "## 🎵 音频\n\n" + "\n".join(lines) + "\n"
    return force_section(md, "audio", replacement)


def fix_placeholders(md: str, detected_chords: list = None, notes: list = None) -> str:
    """移除报告中残留的占位符文本。

    AI 有时会忽略 prompt 指令，仍输出 [品位待核实] 等占位符。
    此函数在报告生成后做最后一道清理。
    """
    import re

    placeholder_patterns = [
        r'\[品位待核实\]',
        r'\[待核实\]',
        r'\[X弦X品\]',
        r'\[具体品位[^\]]*\]',
        r'XX\s*BPM',
    ]
    for pattern in placeholder_patterns:
        md = re.sub(pattern, '', md)

    # 清理可能留下的空括号或多余空格
    md = re.sub(r'\[\s*\]', '', md)
    md = re.sub(r'在\s*品\s*上', '在低把位', md)
    md = re.sub(r'在\s*品\s*到\s*品', '在指板上', md)
    md = re.sub(r'  +', ' ', md)

    return md


def force_audio_section_nosound(md: str) -> str:
    """无音频信号时覆盖音频章节"""
    replacement = "## 🎵 音频\n\n> 🔇 未检测到有效音频信号，请确认录制的视频包含清晰的乐器声音\n"
    return force_section(md, "audio", replacement)
