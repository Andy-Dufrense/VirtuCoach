"""
VirtuCoach 评分回归测试。

用法:
    # 跑全部基线测试
    cd E:/VirtuCoach/backend
    PYTHONPATH="E:/VirtuCoach-Lib" python -m pytest tests/score_regression/ -v

    # 跑单个
    PYTHONPATH="E:/VirtuCoach-Lib" python -m pytest tests/score_regression/test_baselines.py::test_freeplay_clean -v

    # 更新基线（评分公式有意调整后）
    python tests/score_regression/update_baselines.py

基线视频放在 tests/score_regression/videos/ 下。
每个视频的基线分数在 baselines.json 中定义。
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

import pytest

TESTS_DIR = Path(__file__).parent
VIDEOS_DIR = TESTS_DIR / "videos"
BASELINES_PATH = TESTS_DIR / "baselines.json"

BASELINES = {
    "freeplay_clean": {
        "file": "15634125-标准.mp4",
        "ranges": {
            "pitch": (80, 95),
            "rhythm": (78, 95),
            "technique": (60, 95),
            "overall": (78, 95),
        },
    },
    "freeplay_flawed": {
        "file": "15634125-瑕疵.mp4",
        "ranges": {
            "pitch": (80, 95),
            "rhythm": (72, 92),
            "technique": (55, 95),
            "overall": (72, 92),
        },
    },
}

# 保存基线到 JSON
def save_baselines():
    baselines_path = Path(__file__).parent / "baselines.json"
    # convert frozenset etc.
    baselines_path.write_text(json.dumps(BASELINES, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_analysis(video_path: Path) -> Optional[dict]:
    """同步跑完整的分析流水线（复用 main.py 的生产 DI 装配），返回分数字典。

    Graduate 的 AnalysisService.run() 签名：
        run(task_id, video_path, instrument, level, title, tasks, capo=0)
    结果不返回，写入 tasks[task_id]["result"]，与路由的 BackgroundTasks 调用一致。
    """
    import uuid
    import main as main_module
    import asyncio

    service = main_module.analysis_service
    task_id = f"regression_{uuid.uuid4().hex[:8]}"
    tasks = {
        task_id: {
            "id": task_id,
            "status": "uploaded",
            "progress": 0,
            "message": "",
            "video_path": str(video_path),
            "instrument": "guitar",
            "level": "intermediate",
            "title": video_path.stem,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "result": None,
            "user_id": None,  # 回归测试不落练习记录
        }
    }

    asyncio.run(service.run(
        task_id, str(video_path), "guitar", "intermediate",
        video_path.stem, tasks, 0,
    ))

    result = tasks[task_id].get("result")
    if not result or "score" not in result:
        return None

    return {
        "pitch": result["score"].get("pitch") or 0,
        "rhythm": result["score"].get("rhythm") or 0,
        "technique": result["score"].get("technique") or 0,
        "overall": result["score"].get("overall") or 0,
    }


def _resolve_video(filename: str) -> Path:
    """Find a test video by filename. Checks tests/score_regression/videos/ first, then demo/ folders."""
    local = VIDEOS_DIR / filename
    if local.exists():
        return local

    demo_paths = [
        Path("E:/VirtuCoach/demo/视频分析") / filename,
        Path("E:/VirtuCoach/demo/手型检查") / filename,
        Path("E:/VirtuCoach/backend/uploads") / filename,
    ]
    for p in demo_paths:
        if p.exists():
            return p

    raise FileNotFoundError(f"找不到测试视频: {filename}")


@pytest.mark.slow
@pytest.mark.parametrize("test_name,config", BASELINES.items())
def test_baseline(test_name, config):
    """验证视频分析的评分在基线范围内。"""
    video_path = _resolve_video(config["file"])
    scores = _run_analysis(video_path)

    assert scores is not None, f"分析失败，无返回数据: {config['file']}"

    failures = []
    for metric, (lo, hi) in config["ranges"].items():
        actual = scores.get(metric)
        if actual is None:
            failures.append(f"  {metric}: 缺失")
        elif not (lo <= actual <= hi):
            failures.append(f"  {metric}: {actual} 超出 [{lo}, {hi}]")

    if failures:
        msg = f"{config['file']} 评分回归:\n" + "\n".join(failures)
        msg += f"\n  实际分数: pitch={scores['pitch']} rhythm={scores['rhythm']} technique={scores['technique']} overall={scores['overall']}"
        pytest.fail(msg)

    print(f"\n{config['file']}: pitch={scores['pitch']} rhythm={scores['rhythm']} technique={scores['technique']} overall={scores['overall']} ✓")
