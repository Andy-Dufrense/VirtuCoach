"""
VirtuCoach - 音频分离模块
使用 demucs 分离人声和伴奏，提取纯净吉他音轨
"""
import os
import subprocess
import tempfile
import numpy as np
from typing import Optional

from logging_config import get_logger

logger = get_logger(__name__)


class AudioSeparator:
    """人声/伴奏分离器"""

    def __init__(self):
        self.available = self._check_demucs()

    def _check_demucs(self):
        try:
            import demucs
            return True
        except ImportError:
            logger.info("demucs 未安装，跳过人声分离")
            return False

    def separate(self, audio_path: str) -> Optional[str]:
        """分离人声，返回无伴奏（吉他+其他乐器）音轨路径"""
        if not self.available:
            return None

        try:
            # 创建临时输出目录
            out_dir = tempfile.mkdtemp(prefix="demucs_")

            cmd = [
                "python", "-m", "demucs",
                "-d", "cpu",
                "--two-stems", "vocals",
                "-o", out_dir,
                "--mp3",
                audio_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            logger.info(f"demucs stdout: {result.stdout[-200:]}")
            if result.stderr:
                logger.info(f"demucs stderr: {result.stderr[-200:]}")

            # demucs 输出结构: out_dir/htdemucs/filename/no_vocals.mp3
            import glob
            pattern = os.path.join(out_dir, "**", "no_vocals.*")
            matches = glob.glob(pattern, recursive=True)
            if matches:
                acc_path = matches[0]
                logger.info(f"分离成功: {acc_path}")
                return acc_path

            logger.info("未找到分离结果文件")
            return None

        except subprocess.TimeoutExpired:
            logger.info("demucs 超时")
            return None
        except Exception as e:
            logger.warning(f"分离失败: {e}")
            return None
