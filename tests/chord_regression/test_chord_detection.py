"""
和弦检测回归测试。

验证修改 chord detection 参数后，已知测试视频的和弦序列不退化。
基线来源于 2026-08-03 会话的 8.03 修复 + 8.05 七和弦验证。

用法:
    cd F:/VirtuCoach-Graduate
    python -m pytest tests/chord_regression/ -v
    （路径解析见 tests/conftest.py）
"""

import sys
from pathlib import Path

import pytest

# sys.path 由 tests/conftest.py 统一配置

# (video_filename, capo, must_contain_chords, must_not_contain_chords, tolerance)
# must_contain: 这些和弦必须在检测结果中出现（允许时序容差）
# must_not_contain: 这些和弦绝对不能出现（假阳性检测）
TEST_CASES = [
    {
        "name": "标准-15634125-capo3",
        "file": "15634125-标准.mp4",
        "capo": 3,
        "must_contain": ["C", "G", "Am", "Em", "F", "Dm7"],
        "must_not_contain": ["Cmaj7", "Fmaj7", "G7"],
        "min_chord_count": 5,
    },
    {
        "name": "间断-15634125-capo3",
        "file": "15634125-间断.mp4",
        "capo": 3,
        "must_contain": ["C", "G", "F"],
        "must_not_contain": ["Cmaj7"],
        "min_chord_count": 4,
    },
    {
        "name": "正常-G调-capo0",
        "file": "7.29-G-15634125-正常.mp4",
        "capo": 0,
        "must_contain": ["G", "D", "Em", "Bm7", "C"],
        "must_not_contain": ["Gmaj7", "Bmaj7", "Dmaj7"],
        "min_chord_count": 4,
    },
]

VIDEO_SEARCH_PATHS = [
    Path("E:/VirtuCoach/demo/视频分析"),
    Path("E:/VirtuCoach/demo/手型检查"),
    Path("E:/VirtuCoach/backend/uploads"),
    Path(__file__).parent / "videos",
]

# ── helpers ──

def _find_video(filename: str) -> Path:
    for base in VIDEO_SEARCH_PATHS:
        p = base / filename
        if p.exists():
            return p
    raise FileNotFoundError(f"找不到测试视频: {filename}")


def _extract_audio(video_path: Path):
    """从视频提取音频为 numpy array (16kHz mono)。优先用 librosa。"""
    import librosa
    import numpy as np
    audio, sr = librosa.load(str(video_path), sr=16000, mono=True)
    if len(audio) < sr * 0.5:
        raise ValueError(f"音频太短: {len(audio)/sr:.1f}s")
    return audio.astype(np.float32), sr


def _detect_chords(audio, sr, capo):
    """运行 chroma 和弦检测，返回检测到的和弦名列表。"""
    from audio_analyzer import AudioAnalyzer
    analyzer = AudioAnalyzer()
    detected = analyzer._detect_chords_chroma(audio, sr=sr, capo=capo)
    # chord_name 格式为 "C和弦"，去掉"和弦"后缀方便比对
    return [d["chord_name"].replace("和弦", "") for d in detected], detected


# ── tests ──

@pytest.mark.slow
@pytest.mark.parametrize("case", TEST_CASES, ids=[c["name"] for c in TEST_CASES])
def test_chord_detection(case):
    """验证和弦检测：必须检出的和弦出现，禁止假阳性，至少检出 min_chord_count 个和弦。"""
    video_path = _find_video(case["file"])
    audio, sr = _extract_audio(video_path)

    chord_names, raw = _detect_chords(audio, sr, capo=case["capo"])
    detected_set = set(chord_names)

    failures = []

    # 必须检出
    missing = [c for c in case["must_contain"] if c not in detected_set]
    if missing:
        failures.append(f"缺失和弦: {missing}")

    # 禁止假阳性
    false_pos = [c for c in case["must_not_contain"] if c in detected_set]
    if false_pos:
        failures.append(f"假阳性: {false_pos}")

    # 数量
    if len(chord_names) < case["min_chord_count"]:
        failures.append(f"检出数量不足: {len(chord_names)} < {case['min_chord_count']}")

    if failures:
        detail = "\n".join(failures)
        detail += f"\n  实际检出 ({len(chord_names)}个): {chord_names}"
        pytest.fail(f"{case['name']} ({case['file']}):\n{detail}")

    print(f"\n  {case['name']}: 检出 {len(chord_names)} 个和弦 → {chord_names}")


@pytest.mark.slow
@pytest.mark.parametrize("case", TEST_CASES, ids=[c["name"] for c in TEST_CASES])
def test_no_maj7_false_positives(case):
    """验证没有 maj7 假阳性（5th 泛音物理上无法区分，必须屏蔽）。"""
    video_path = _find_video(case["file"])
    audio, sr = _extract_audio(video_path)

    _, raw = _detect_chords(audio, sr, capo=case["capo"])
    chord_names = [d["chord_name"] for d in raw]

    maj7_hits = [c for c in chord_names if "maj7" in c.lower()]
    if maj7_hits:
        pytest.fail(f"{case['name']}: maj7 假阳性: {maj7_hits}")
