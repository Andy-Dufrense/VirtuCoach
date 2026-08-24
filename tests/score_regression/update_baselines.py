"""
更新评分基线。

用法:
    cd E:/VirtuCoach/backend
    PYTHONPATH="E:/VirtuCoach-Lib" python tests/score_regression/update_baselines.py

当前基线跑完后，将实际分数写入 baselines.json，用于后续回归对比。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from test_baselines import BASELINES, _resolve_video, _run_analysis, BASELINES_PATH


def main():
    print("=" * 50)
    print("  更新评分基线")
    print("=" * 50)

    new_baselines = {}
    for test_name, config in BASELINES.items():
        try:
            video_path = _resolve_video(config["file"])
            scores = _run_analysis(video_path)
        except FileNotFoundError as e:
            print(f"  ✗ {test_name}: {e}")
            continue

        if scores is None:
            print(f"  ✗ {test_name}: 分析失败")
            continue

        margin = 5
        new_baselines[test_name] = {
            "file": config["file"],
            "ranges": {
                "pitch": [max(0, scores["pitch"] - margin), min(100, scores["pitch"] + margin)],
                "rhythm": [max(0, scores["rhythm"] - margin), min(100, scores["rhythm"] + margin)],
                "technique": [max(0, scores["technique"] - margin), min(100, scores["technique"] + margin)],
                "overall": [max(0, scores["overall"] - margin), min(100, scores["overall"] + margin)],
            },
        }
        print(f"  {config['file']}: pitch={scores['pitch']} rhythm={scores['rhythm']} technique={scores['technique']} overall={scores['overall']} ✓")

    BASELINES_PATH.write_text(json.dumps(new_baselines, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  已写入: {BASELINES_PATH}")


if __name__ == "__main__":
    main()
