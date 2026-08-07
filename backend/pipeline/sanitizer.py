"""
报告后处理过滤器：拦截 AI 输出中的弦/品引用。

在 DeepSeek 生成报告后、返回前端前，对 report_markdown 全文扫描，
命中弦/品模式的文本用占位符替换。

为什么需要：之前三次（7/20、7/21、7/24）从源头修 prompt 来阻止弦/品泄露，
但每次加新数据字段都会出现新的泄露路径。源头拦截是打地鼠，输出端过滤是一次性方案。
"""

import re
from logging_config import get_logger

logger = get_logger(__name__)

FRET_PATTERNS = [
    (re.compile(r'\d+弦\s*\d+品'), '[弦/品引用已过滤]'),
    (re.compile(r'\d+弦\s*空弦'), '[弦/品引用已过滤]'),
    (re.compile(r'[一二三四五六七八]弦\s*[一二三四五六七八九十\d]+品'), '[弦/品引用已过滤]'),
    (re.compile(r'[一二三四五六]弦\s*[零一二三四五六七八九十]+品'), '[弦/品引用已过滤]'),
    (re.compile(r'第\s*\d+\s*品'), '[品位引用已过滤]'),
    (re.compile(r'[一二三四五六]弦\s*空弦'), '[弦/品引用已过滤]'),
]

# 音名+八度模式：G3, A#3, F4, D#3, Eb2, C#4 等
NOTE_NAME_PATTERN = re.compile(r'(?<![A-Za-z])[A-G](?:#|b)?\d(?![A-Za-z])')


def sanitize_report(text: str) -> str:
    """扫描并替换 report_markdown 中的弦/品引用和音名引用。"""
    if not text:
        return text

    replaced = 0
    for pattern, replacement in FRET_PATTERNS:
        new_text, count = pattern.subn(replacement, text)
        if count:
            replaced += count
            text = new_text

    # 过滤音名+八度（如 G3、A#3、F4）
    note_count = len(NOTE_NAME_PATTERN.findall(text))
    if note_count:
        text = NOTE_NAME_PATTERN.sub('[音名]', text)
        replaced += note_count

    if replaced:
        logger.info(f"报告后处理: 过滤了 {replaced} 处弦/品/音名引用")

    return text
