"""
扫弦分类回归测试。

验证修改分类器参数后，已知视频的扫弦/非扫弦判定不退化。
基线来源于 7.30-31 和 8.03 会话的分类器修复。

检测路径：
- detect_technique_segments（音符+activation交叉验证）：扫弦 yes/no（主检测源）
- _detect_strumming_pattern（频谱匹配法）：类型细分 (strumming/arpeggio/block_chord)

用法:
    cd F:/VirtuCoach-Graduate
    python -m pytest tests/strum_regression/ -v
    （路径解析见 tests/conftest.py）
"""

import sys
from pathlib import Path

import pytest

# sys.path 由 tests/conftest.py 统一配置

TEST_CASES = [
    {
        "name": "标准扫弦-15634125",
        "file": "15634125-扫弦.mp4",
        "must_be": "strumming",
    },
    {
        "name": "纯扫弦",
        "file": "扫弦.mp4",
        "must_be": "strumming",
    },
    {
        "name": "分解和弦-不应判扫弦",
        "file": "7.29-15634125-错误.mp4",
        "must_not_be": "strumming",
        "xfail": "分解和弦的快速音符偶尔聚类为扫弦 (known limitation)",
    },
    {
        "name": "柱式和弦",
        "file": "7.29-1645-柱.mp4",
        "must_be": "strumming",
        "xfail": "柱式和弦音符同时触发，音符级检测漏检；频谱级可检测但主检测源不覆盖",
    },
    {
        "name": "单音旋律-不应判扫弦",
        "file": "单音小星星.mp4",
        "must_not_be": "strumming",
    },
]

VIDEO_PATHS = [
    Path("E:/VirtuCoach/demo/视频分析"),
    Path("E:/VirtuCoach/demo/手型检查"),
    Path(__file__).parent / "videos",
]


def _find_video(filename):
    for base in VIDEO_PATHS:
        p = base / filename
        if p.exists():
            return p
    raise FileNotFoundError(f"找不到: {filename}")


def _extract_audio(video_path):
    import numpy as np
    import librosa
    audio, sr = librosa.load(str(video_path), sr=16000, mono=True)
    return audio.astype(np.float32), sr


def _run_pipeline(video_path):
    """运行技巧检测管线，返回 (technique_segments, onset_clusters)。

    technique_segments: 音符+activation交叉验证 → 扫弦 yes/no（主检测源）
    onset_clusters: 频谱匹配法 → 类型分类（补充检测源）
    """
    import numpy as np
    from audio_analyzer import AudioAnalyzer, AudioFeatures
    audio, sr = _extract_audio(video_path)
    analyzer = AudioAnalyzer()

    notes = analyzer._transcribe_notes(audio, capo=0)
    if len(notes) < 3:
        return [], []

    mo = getattr(analyzer, '_last_model_output', None)
    if mo is None or (hasattr(mo, 'shape') and mo.shape[0] < 5):
        return [], []

    # 主检测源：音符+activation 交叉验证
    audio_features = AudioFeatures()
    technique_segments = analyzer.detect_technique_segments(mo, audio_features, notes, sr=16000)

    # 补充检测源：频谱匹配法
    duration = len(audio) / sr
    onset_clusters = analyzer._detect_strumming_pattern(
        audio, sr=sr,
        strum_windows=[{"start_time": 0, "end_time": duration}],
        notes=notes,
    )

    return technique_segments, onset_clusters


@pytest.mark.slow
@pytest.mark.parametrize("case", TEST_CASES, ids=[c["name"] for c in TEST_CASES])
def test_strumming_classification(case):
    video_path = _find_video(case["file"])
    tech_segs, onset_clusters = _run_pipeline(video_path)

    # 主检测源：technique_id 是扫弦 yes/no 的权威
    tech_ids = {s.get("technique_id", "unknown") for s in tech_segs}

    # 补充检测源：onset_clusters 的 type 字段（过滤低置信度）
    onset_types = {oc["type"] for oc in onset_clusters
                   if oc.get("type") != "unknown" and oc.get("confidence", 0) >= 0.40}

    # 合并：任一源检测到即为命中（提高召回率，匹配 production 代码行为）
    all_types = tech_ids | onset_types

    xfail_reason = case.get("xfail")

    if "must_be" in case:
        cond = case["must_be"] in all_types
        if not cond and xfail_reason:
            pytest.xfail(f"KNOWN: {xfail_reason}")
        assert cond, \
            f"期望检测到 {case['must_be']}，tech={tech_ids} onset={onset_types}"

    if "must_not_be" in case:
        # 假阳性检查只看主检测源（更保守，减少频谱误判）
        cond = case["must_not_be"] not in tech_ids
        if not cond and xfail_reason:
            pytest.xfail(f"KNOWN: {xfail_reason}")
        assert cond, \
            f"不应检测到 {case['must_not_be']}，tech={tech_ids}"

    print(f"\n  {case['name']}: tech={tech_ids} onset={onset_types}")
