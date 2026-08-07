#!/usr/bin/env python
"""VirtuCoach 知识库一致性校验。

用法: python validate_kb.py

检查 knowledge/ 下所有 markdown 文件的:
- frontmatter 完整性
- 必需章节（与实际 KB 约定一致，而非硬编码模板）
- 和弦: chord_name + strings + fingers
- 问题: 表现 + AI检测/铁律
- 技巧: 与手型错误的区分 + AI 检测豁免
"""

import re
import sys
from pathlib import Path
from typing import Optional

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PROJECT_ROOT = Path(__file__).parent.parent
KB_DIR = PROJECT_ROOT / "knowledge"

CHECKS_PASSED = 0
CHECKS_FAILED = 0


def ok(msg: str):
    global CHECKS_PASSED
    CHECKS_PASSED += 1
    print(f"  [PASS] {msg}")


def fail(msg: str):
    global CHECKS_FAILED
    CHECKS_FAILED += 1
    print(f"  [FAIL] {msg}")


def parse_frontmatter(text: str) -> Optional[dict]:
    m = re.match(r'^---\s*\n(.*?)\n---', text, re.DOTALL)
    if not m:
        return None
    result = {}
    for line in m.group(1).strip().split("\n"):
        if ":" in line:
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip().strip('"').strip("'")
    return result


def check_chords():
    print("\n[和弦 KB] — 期望: chord_name + strings + fingers")
    chord_dir = KB_DIR / "chords"
    files = sorted(chord_dir.glob("*.md"))

    for f in files:
        text = f.read_text(encoding="utf-8")
        fm = parse_frontmatter(text)

        if fm is None:
            fail(f"{f.name}: 缺少 frontmatter")
            continue

        for field in ["chord_name", "strings", "fingers"]:
            if field not in fm:
                fail(f"{f.name}: frontmatter 缺少 '{field}'")

        strings_val = fm.get("strings", "")
        if strings_val:
            try:
                import ast
                cleaned = strings_val.replace("x", "-1").replace("X", "-1")
                parsed = ast.literal_eval(cleaned)
                if not isinstance(parsed, list) or len(parsed) != 6:
                    fail(f"{f.name}: strings 应为6元素列表")
            except (ValueError, SyntaxError):
                fail(f"{f.name}: strings 格式无法解析")

        ok(f.name)

    print(f"  和弦: {len(files)} 个文件检查完成")


def check_problems():
    print("\n[问题 KB] — 期望: frontmatter(id/type/severity) + 表现 + AI检测/铁律")
    prob_dir = KB_DIR / "problems"
    files = sorted(prob_dir.glob("*.md"))

    for f in files:
        text = f.read_text(encoding="utf-8")
        fm = parse_frontmatter(text)

        if fm is None:
            fail(f"{f.name}: 缺少 frontmatter")
            continue

        for field in ["id", "type", "severity"]:
            if field not in fm:
                fail(f"{f.name}: frontmatter 缺少 '{field}'")

        if "## 表现" not in text and "## 描述" not in text and "## 概述" not in text:
            fail(f"{f.name}: 缺少问题描述章节 (## 表现)")

        if "AI检测/铁律" not in text and "AI 检测/铁律" not in text and "铁律" not in text:
            fail(f"{f.name}: 缺少 'AI检测/铁律' 章节")

        ok(f.name)

    print(f"  问题: {len(files)} 个文件检查完成")


def check_techniques():
    print("\n[技巧 KB] — 期望: frontmatter + 与手型错误的区分 + AI 检测豁免")
    tech_dir = KB_DIR / "techniques"
    files = sorted(tech_dir.rglob("*.md"))

    for f in files:
        text = f.read_text(encoding="utf-8")
        fm = parse_frontmatter(text)

        if fm is None:
            fail(f"{f.name}: 缺少 frontmatter")
            continue

        for field in ["id", "type", "difficulty"]:
            if field not in fm:
                fail(f"{f.name}: frontmatter 缺少 '{field}'")

        if "与手型错误的区分" not in text:
            fail(f"{f.relative_to(tech_dir)}: 缺少 '与手型错误的区分' 对比表")

        if "AI 检测豁免" not in text and "AI检测豁免" not in text:
            fail(f"{f.relative_to(tech_dir)}: 缺少 'AI 检测豁免' 章节")

        ok(f"{f.parent.name}/{f.name}" if f.parent != tech_dir else f.name)

    print(f"  技巧: {len(files)} 个文件检查完成")


def check_other():
    print("\n[其他 KB]")
    expected = [
        ("chord-transition.md", "和弦转换"),
        ("diagnosis-priority.md", "诊断优先级"),
        ("problem-exercise-mapping.md", "问题-练习映射"),
    ]
    for name, label in expected:
        path = KB_DIR / name
        if path.is_file():
            ok(f"{name} ({label})")
        else:
            fail(f"{name} ({label}) 不存在")


def main():
    print("=" * 50)
    print("  VirtuCoach 知识库校验")
    print("=" * 50)

    check_chords()
    check_problems()
    check_techniques()
    check_other()

    print("\n" + "=" * 50)
    total = CHECKS_PASSED + CHECKS_FAILED
    if CHECKS_FAILED == 0:
        print(f"  ALL CLEAN: {CHECKS_PASSED} 项全部通过")
    else:
        print(f"  Results: {CHECKS_PASSED} 通过, {CHECKS_FAILED} 失败 (共 {total} 项)")
    print("=" * 50)

    return CHECKS_FAILED == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
