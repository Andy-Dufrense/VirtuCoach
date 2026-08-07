"""
VirtuCoach - 音频分析模块
两层架构:
  1. 信号分析层 (Python 确定性计算): 节拍稳定性 / 动态范围 / 音色 / 起音 / 杂音
  2. AI 解读层 (DeepSeek): 接收量化指标 → 生成描述性语言
"""

import os
import json
import struct
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv
import traceback

from logging_config import get_logger

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

logger = get_logger(__name__)

from guitar_tab_assigner import batch_assign

# scipy 1.15+ 移除了 scipy.signal.hann（改用 scipy.signal.windows.hann）
# librosa 0.10.x 的部分路径仍引用旧位置，monkey-patch 兼容
import scipy.signal
if not hasattr(scipy.signal, 'hann'):
    scipy.signal.hann = scipy.signal.windows.hann

# librosa 用于 chroma 和弦识别、onset 检测、节拍跟踪
try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.info("librosa 未安装，和弦识别将回退到 basic-pitch 方案")

# DeepChroma CNN（madmom）—— 弱信号场景辅助校验，不替代 chroma_cqt 主链路
# madmom 0.16.1 依赖已废弃的 np.float 别名，需 monkey-patch
_DC_AVAILABLE = False
_CNNChordFeatureProcessor = None
try:
    import builtins as _builtins
    for _dep in ('float', 'int', 'bool', 'complex'):
        if not hasattr(np, _dep):
            setattr(np, _dep, getattr(_builtins, _dep, float))
    from madmom.features.chords import CNNChordFeatureProcessor as _CNNChordFeatureProcessor
    _DC_AVAILABLE = True
    logger.info("DeepChroma (madmom CNN) 已加载，将用于弱信号交叉验证")
except Exception as _e:
    logger.info(f"DeepChroma 不可用 ({_e})，仅使用 chroma_cqt 主链路")


@dataclass
class AudioFeatures:
    """音频信号分析层输出的量化指标。

    所有值由 Python 确定性计算得出，不可被 LLM 覆盖。
    """
    # 节拍稳定性
    tempo_bpm: float = 0.0
    ioi_cv: float = 0.0                # IOI 变异系数（含离群值）, <0.1=良好, >0.2=不稳定
    ioi_cv_core: float = 0.0           # 核心 CV（仅统计 P90 以内 IOI），排除乐句呼吸干扰
    local_rush_regions: List[Dict] = field(default_factory=list)  # 加速区域
    local_drag_regions: List[Dict] = field(default_factory=list)  # 拖拍区域

    # 动态范围
    velocity_p5: float = 0.0
    velocity_p50: float = 0.0
    velocity_p95: float = 0.0
    dynamic_range_db: float = 0.0

    # 音色 (频谱特征)
    spectral_centroid_avg: float = 0.0   # Hz, 越高越亮
    attack_time_avg_ms: float = 0.0      # ms, 起音上升时间
    harmonic_decay_slope: float = 0.0    # 泛音衰减斜率, 越负衰减越快

    # 杂音
    inharmonicity_ratio: float = 0.0     # 非谐波能量比
    high_freq_noise_ratio: float = 0.0   # >8kHz 能量占比

    # 演奏特征描述 (供 LLM 解读用)
    feature_summary: str = ""

def _midi_to_note_name(midi_pitch: int) -> str:
    """MIDI音高 → 音名（如 C4, F#3）"""
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (midi_pitch // 12) - 1
    return f"{names[midi_pitch % 12]}{octave}"


class AudioAnalyzer:
    """音频分析器：从视频中提取音符并进行 DeepSeek 智能纠错"""

    def __init__(self):
        self.model_loaded = False
        self.ds_available = False
        self._load_model()
        self._init_deepseek()
        self._deep_chroma_scores = None

    def _load_model(self):
        """加载 basic-pitch 模型"""
        try:
            import basic_pitch
            self.model_loaded = True
            logger.info("basic-pitch 就绪")
        except ImportError:
            logger.info("basic-pitch 未安装")
        except Exception as e:
            logger.warning(f"模型加载失败: {e}")

    def _init_deepseek(self):
        """初始化 DeepSeek 客户端（用于智能音频分析）"""
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if api_key and api_key != "sk-your-deepseek-api-key-here":
            try:
                from openai import OpenAI
                self.ds_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
                self.ds_available = True
                logger.info("DeepSeek 智能分析就绪")
            except Exception as e:
                logger.warning(f"DeepSeek 初始化失败: {e}")

    # ==================== 音频提取（纯 Python，不依赖 ffmpeg） ====================

    def _read_audio_python(self, video_path: str) -> Optional[np.ndarray]:
        """
        纯 Python 读取视频中的音频轨道
        策略：
        1. 先尝试 librosa（支持通过 audioread 后端读视频）
        2. 失败则尝试用 scipy 或 soundfile 等
        """
        try:
            import librosa
            # librosa 可以用 audioread 后端读部分视频格式
            y, sr = librosa.load(video_path, sr=16000, mono=True, duration=None)
            if len(y) > 0:
                return y
        except Exception as e1:
            pass

        try:
            import subprocess
            # 尝试用系统已装的 ffmpeg（如果有的话）
            result = subprocess.run(
                ["ffmpeg", "-i", video_path, "-f", "wav", "-acodec", "pcm_s16le",
                 "-ac", "1", "-ar", "16000", "-"],
                capture_output=True, timeout=60
            )
            if result.returncode == 0 and len(result.stdout) > 100:
                import soundfile as sf
                import io
                y, sr = sf.read(io.BytesIO(result.stdout))
                return y
        except:
            pass

        return None

    # ==================== 核心分析流程 ====================

    def analyze(self, video_path: str, instrument: str = "piano", capo: int = 0,
                hand_zone_center: float = None) -> Dict:
        """
        分析视频中的音频

        返回:
        {
            "notes": [...],          # 原始转录音符
            "errors": [...],         # DeepSeek 判断的错误列表
            "error_timestamps": [],  # 错误时间戳
            "stats": {...},          # 统计数据
            "note_text": "音符文字描述（供DeepSeek分析）",
        }
        """
        result = {
            "notes": [],
            "errors": [],
            "error_timestamps": [],
            "note_text": "",
            "detected_capo": 0,
            "technique_segments": [],
            "stats": {
                "total_notes": 0,
                "pitch_accuracy": 0,
                "rhythm_accuracy": 0,
                "overall_score": 70,
            }
        }

        try:
            # 1. 读取音频
            logger.info(f"开始分析: {video_path}")
            audio = self._read_audio_python(video_path)
            if audio is None:
                logger.warning("音频提取失败")
                result["errors"].append({
                    "type": "audio_extraction_failed",
                    "msg": "无法从视频中提取音频。请确保视频包含音频轨道。"
                })
                return result
            logger.info(f"音频提取成功, 长度={len(audio)}")

            # 静音检测：RMS 能量过低视为无音频
            rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
            logger.info(f"RMS={rms:.6f}")
            if rms < 0.003:
                logger.info("音频能量过低，视为静音")
                result["errors"].append({
                    "type": "audio_extraction_failed",
                    "msg": "视频音频轨道为静音，未检测到有效声音。"
                })
                return result

            # 1.5 起始检测：找到演奏开始位置
            onset_time = self._detect_playing_onset(audio)
            # 如果 onset 检测失败（返回 0），用 RMS 兜底
            if onset_time <= 0.01 and len(audio) > 8000:
                frame_len = 800  # 50ms @ 16kHz
                hop = 400
                n_frames_rms = (len(audio) - frame_len) // hop + 1
                rms_frames = np.array([
                    np.sqrt(np.mean(audio[i*hop:i*hop+frame_len]**2))
                    for i in range(n_frames_rms)
                ])
                rms_thresh = np.mean(rms_frames) * 1.3
                above = np.where(rms_frames > rms_thresh)[0]
                if len(above) > 0:
                    onset_time = round(float(above[0] * 400 / 16000), 2)
                    logger.info(f"RMS起始检测(兜底): {onset_time:.2f}s")
            result["playing_onset"] = onset_time

            # 1.6 节拍跟踪
            beat_info = self._track_beats(audio, onset_time=onset_time)
            result["beat_info"] = beat_info

            # 2. basic-pitch 转录
            notes = self._transcribe_notes(audio, hand_zone_center, capo)
            result["stats"]["total_notes"] = len(notes)
            logger.info(f"转录音符数: {len(notes)}")

            if not notes:
                result["errors"].append({
                    "type": "no_notes_detected",
                    "msg": "未检测到明显的音符。请确认视频中有清晰的乐器声。"
                })
                return result

            # 2.5 Chroma 和弦识别（演奏开始前的静音清零，避免噪底匹配七和弦）
            onset_samples = int(onset_time * 16000) if onset_time > 0 else 0
            if onset_samples > 0:
                audio_chord = audio.copy()
                audio_chord[:onset_samples] = 0
            else:
                audio_chord = audio
            detected_chords = self._detect_chords_chroma(audio_chord, capo=capo, notes=notes)
            best_capo = capo
            result["detected_capo"] = best_capo

            # 2.55 和弦意图解析（多源证据纠正 chroma 纯声学匹配）
            if detected_chords and notes:
                try:
                    detected_chords = self._resolve_chord_intent(
                        detected_chords, notes, capo=best_capo
                    )
                except Exception as e:
                    logger.warning(f"和弦意图解析失败: {e}")

            # 2.6 音符反向验证 + 补充纠正
            if detected_chords:
                detected_chords = self._verify_chords_with_notes(
                    detected_chords, notes, capo=best_capo)
            detected_chords = self._refine_chords_with_notes(
                detected_chords or [], notes, capo=best_capo)
            result["detected_chords"] = detected_chords

            # 2.7 和弦驱动音符过滤（用和弦结论过滤幻音，必须在使用 notes 之前）
            phantom_notes = []
            if detected_chords:
                filtered_notes, phantom_notes = self._filter_notes_by_chords(notes, detected_chords)
                result["phantom_notes"] = phantom_notes
            else:
                filtered_notes = notes
            result["notes"] = filtered_notes

            # 3. 信号分析层：提取定量特征（确定性计算，使用过滤后音符）
            audio_features = self.extract_audio_features(audio, sr=16000, notes=filtered_notes)
            result["audio_features"] = asdict(audio_features)

            # 4. 构建音符文本描述（使用过滤后音符）
            note_text = self._build_note_text(filtered_notes)
            result["note_text"] = note_text

            # 4.5 乐句间隔检测（使用过滤后音符，避免幻音产生假间隔）
            gap_analysis = self._detect_phrase_gaps(filtered_notes)
            result["gap_analysis"] = gap_analysis

            # 5. DeepSeek 智能解读（使用过滤后音符）
            if self.ds_available:
                ds_errors, ds_stats = self._deepseek_analyze_errors(
                    note_text, filtered_notes, instrument, audio_features, gap_analysis
                )
                ds_pitch = ds_stats.get("pitch_accuracy", 0)
                ds_rhythm = ds_stats.get("rhythm_accuracy", 0)
                if ds_pitch == 0 and ds_rhythm == 0 and len(filtered_notes) > 0:
                    logger.warning("DS audio returned zero scores, falling back to rule-based (ds_pitch=%s, ds_rhythm=%s, notes=%d)", ds_pitch, ds_rhythm, len(filtered_notes))
                    errors, timestamps = self._rule_based_errors(filtered_notes, instrument)
                    stats = {
                        "total_notes": len(filtered_notes),
                        "pitch_accuracy": self._calc_pitch_accuracy(errors),
                        "rhythm_accuracy": self._calc_rhythm_accuracy(errors),
                        "overall_score": self._calc_overall_score(errors, len(filtered_notes)),
                    }
                else:
                    errors = ds_errors
                    stats = ds_stats
                    stats["total_notes"] = len(filtered_notes)
                    ioi_cv_core = getattr(audio_features, 'ioi_cv_core', 0) if audio_features else 0
                    ds_rhythm = stats.get("rhythm_accuracy", 80)
                    if ioi_cv_core > 0:
                        if ioi_cv_core < 0.45:
                            if ds_rhythm < 82:
                                stats["rhythm_accuracy"] = 82
                                stats["overall_score"] = max(stats.get("overall_score", 82), 82)
                                logger.info("CV-rhythm锚定: CV(core)=%.3f→良好, rhythm %s→82",
                                          ioi_cv_core, ds_rhythm)
                        else:
                            if ds_rhythm > 81:
                                stats["rhythm_accuracy"] = 81
                                stats["overall_score"] = min(stats.get("overall_score", 81), 81)
                                logger.info("CV-rhythm锚定: CV(core)=%.3f→偏不稳, rhythm %s→81",
                                          ioi_cv_core, ds_rhythm)
            else:
                errors, timestamps = self._rule_based_errors(filtered_notes, instrument)
                stats = {
                    "total_notes": len(filtered_notes),
                    "pitch_accuracy": self._calc_pitch_accuracy(errors),
                    "rhythm_accuracy": self._calc_rhythm_accuracy(errors),
                    "overall_score": self._calc_overall_score(errors, len(filtered_notes)),
                }

            # 4.5 闷音/哑音检测（使用过滤后音符）
            dead_errors = self._detect_dead_notes(filtered_notes, audio_features)
            if dead_errors:
                existing_dead_times = {e.get("time", 0) for e in errors if e.get("type") == "dead_note"}
                for de in dead_errors:
                    t = de.get("time", 0)
                    if not any(abs(t - edt) < 1.5 for edt in existing_dead_times):
                        errors.append(de)
                logger.info(f"闷音检测合并: 新增 {len([de for de in dead_errors if not any(abs(de.get('time',0)-edt)<1.5 for edt in existing_dead_times)])} 个闷音区域")

            # 4.6 乐句间隔→transition_gap 转换（确定性检测，绕过 LLM 避免漏报）
            gap_errors = self._gaps_to_errors(gap_analysis)
            if gap_errors:
                existing_gap_times = {e.get("time", 0) for e in errors if e.get("type") == "transition_gap"}
                for ge in gap_errors:
                    t = ge.get("time", 0)
                    if not any(abs(t - edt) < 1.5 for edt in existing_gap_times):
                        errors.append(ge)
                logger.info(f"间隔检测合并: 新增 {len([ge for ge in gap_errors if not any(abs(ge.get('time',0)-edt)<1.5 for edt in existing_gap_times)])} 个 transition_gap")

            # 按时间排序所有错误
            errors.sort(key=lambda e: e.get("time", 0))

            result["errors"] = errors
            result["error_timestamps"] = [e["time"] for e in errors if "time" in e]
            result["stats"] = stats

            # 5.5 技巧检测：从 model_output + audio_features 推断技巧时间段
            # 使用原始 notes（未过滤）做扫弦检测，避免和弦过滤误删扫弦簇
            mo = getattr(self, '_last_model_output', None)
            if mo is not None and notes:
                try:
                    technique_segments = self.detect_technique_segments(
                        mo, audio_features, notes, sr=16000
                    )
                    result["technique_segments"] = technique_segments
                    logger.info(f"技巧检测: {len(technique_segments)} 个时间段")
                except Exception as e:
                    logger.warning(f"技巧检测失败: {e}")
                    result["technique_segments"] = []

            # 5.6 技巧模式检测：匹配法分类（扫弦/琶音/分解和弦）
            # 频谱优先：基于 raw audio onset + pitch ordering 方向 + 匹配法分类
            strum_tech_segs = [
                s for s in result.get("technique_segments", [])
                if s.get("technique_id") == "strumming" and s.get("confidence", 0) >= 0.5
            ]

            onset_t = result.get("playing_onset", 0)
            audio_end = len(audio) / 16000
            full_window = [{"start_time": onset_t, "end_time": audio_end}]
            onset_clusters = []
            try:
                onset_clusters = self._detect_strumming_pattern(
                    audio, sr=16000, strum_windows=full_window, notes=notes
                )
                if onset_clusters:
                    type_summary = ", ".join(
                        f"{c['type']}(conf={c.get('type_confidence',0):.2f})" for c in onset_clusters[:3]
                    )
                    logger.info(f"Onset cluster分类(频谱): {len(onset_clusters)} 段 → {type_summary}")

                    # 只把 strumming 类型的 cluster 补充到 technique_segments
                    spec_strum_segs = []
                    for pat in onset_clusters:
                        if pat.get("type") == "strumming":
                            spec_strum_segs.append({
                                "start_time": pat["start_time"],
                                "end_time": pat["end_time"],
                                "technique_id": "strumming",
                                "confidence": pat.get("confidence", 0.7),
                                "source": "spectral",
                            })
                    # 合并去重
                    existing_ids = set()
                    for s in strum_tech_segs:
                        key = (round(s["start_time"], 1), round(s["end_time"], 1))
                        existing_ids.add(key)
                    for s in spec_strum_segs:
                        key = (round(s["start_time"], 1), round(s["end_time"], 1))
                        if key not in existing_ids:
                            strum_tech_segs.append(s)
                            result["technique_segments"].append(s)
                    logger.info(f"扫弦段合并: 音符{len([s for s in strum_tech_segs if s.get('source')!='spectral'])} + 频谱{len(spec_strum_segs)} = {len(strum_tech_segs)}")
            except Exception as e:
                logger.warning(f"Onset cluster分类失败: {e}")

            has_strumming = any(p.get("type") == "strumming" for p in onset_clusters)
            result["onset_clusters"] = onset_clusters  # 新字段：匹配法分类结果
            result["strumming_patterns"] = onset_clusters  # 向后兼容
            if onset_clusters:
                types_seen = set(c["type"] for c in onset_clusters)
                logger.info(f"技巧分类: {len(onset_clusters)} 段, 类型={types_seen}")

            # 5.6.5 扫弦质量评估（仅评估type="strumming"的cluster）
            try:
                if onset_clusters and any(p.get("type") == "strumming" for p in onset_clusters):
                    strumming_quality = self._evaluate_strumming_quality(
                        onset_clusters, audio, technique_segments=result.get("technique_segments", []), sr=16000
                    )
                else:
                    strumming_quality = None
                result["strumming_quality"] = strumming_quality
                if strumming_quality:
                    logger.info(f"扫弦质量: overall={strumming_quality['overall_score']:.2f}, "
                                f"rhythm_cv={strumming_quality['rhythm_cv']:.3f}, "
                                f"{strumming_quality['feedback_summary'][:80]}")
            except Exception as e:
                logger.warning(f"扫弦质量评估失败: {e}")
                result["strumming_quality"] = None

            # 5.6b 扫弦段和弦校正 — 禁用
            # basic-pitch 音符转录在复音场景下不可靠，用它覆写 chroma 会引入错误
            # if has_strumming and filtered_notes:
            #     try:
            #         detected_chords = self._correct_chords_during_strumming(
            #             detected_chords, strum_tech_segs, filtered_notes
            #         )
            #         result["detected_chords"] = detected_chords
            #     except Exception as e:
            #         logger.warning(f"扫弦和弦校正失败: {e}")

            # 5.7 技巧检测 Phase 3: 击弦/勾弦/滑音/推弦/揉弦
            try:
                midi_data = getattr(self, '_last_midi_data', None)
                note_techniques = self._detect_techniques_from_notes(
                    filtered_notes, midi_data=midi_data
                )
                result["note_techniques"] = note_techniques
                if note_techniques:
                    logger.info(f"音符技巧检测: {len(note_techniques)} 个")
            except Exception as e:
                logger.warning(f"音符技巧检测失败: {e}")
                result["note_techniques"] = []

        except Exception as e:
            result["errors"].append({"type": "analysis_error", "msg": str(e)})
            logger.warning(f"分析异常: {traceback.format_exc()}")

        return result

    # ==================== 信号分析层：确定性特征提取 ====================

    def extract_audio_features(self, audio: np.ndarray, sr: int = 16000,
                                notes: Optional[List[Dict]] = None) -> AudioFeatures:
        """从音频波形中提取多维定量特征。

        Args:
            audio: 音频波形 (1D numpy array)
            sr: 采样率, 默认 16000
            notes: basic-pitch 转录音符列表（可选，用于节拍+动态分析）

        Returns:
            AudioFeatures 数据类，包含所有定量指标
        """
        features = AudioFeatures()

        # 1. 节拍稳定性 (基于 basic-pitch 转录的 IOI)
        if notes and len(notes) >= 4:
            self._compute_tempo_features(notes, features)

        # 2. 动态范围 (基于 velocity 数据)
        if notes and len(notes) >= 4:
            self._compute_dynamic_features(notes, features)

        # 3. 频谱特征 (基于原始音频波形)
        try:
            self._compute_spectral_features(audio, sr, features)
        except Exception as e:
            logger.warning(f"频谱特征提取失败: {e}")

        # 4. 构建文字摘要供 LLM 解读
        features.feature_summary = self._build_feature_summary(features)

        return features

    def _compute_tempo_features(self, notes: List[Dict], f: AudioFeatures):
        """计算节拍稳定性特征。"""
        onsets = sorted(set(n["start_time"] for n in notes))
        if len(onsets) < 4:
            return

        # Filter near-simultaneous onsets (chord strums): keep only first onset in each 50ms cluster
        filtered_onsets = [onsets[0]]
        for t in onsets[1:]:
            if t - filtered_onsets[-1] > 0.05:  # 50ms gap = distinct musical event
                filtered_onsets.append(t)

        # Trim pre-playing silence: find the first cluster of 3+ notes within 1.5s
        # This skips isolated noise detections before actual playing starts
        if len(filtered_onsets) >= 6:
            for i in range(len(filtered_onsets) - 2):
                if filtered_onsets[i + 2] - filtered_onsets[i] < 1.5:
                    if i > 0:
                        filtered_onsets = filtered_onsets[i:]
                    break

        # Skip tempo analysis if total time span is too short (<2s) or too few distinct events
        total_span = filtered_onsets[-1] - filtered_onsets[0]
        if len(filtered_onsets) < 4 or total_span < 2.0:
            f.ioi_cv = 0.0  # 数据不足，跳过节奏评估
            f.tempo_bpm = 0
            return

        iois = np.diff(filtered_onsets)
        mean_ioi = float(np.mean(iois))
        std_ioi = float(np.std(iois))

        if mean_ioi > 0.05:
            f.ioi_cv = round(std_ioi / mean_ioi, 4)
            f.tempo_bpm = round(60.0 / mean_ioi, 1)
            # 核心 CV：仅用 P90 以内的 IOI，排除乐句呼吸等离群值对整体节奏评估的干扰
            p90 = float(np.percentile(iois, 90))
            core_iois = iois[iois <= p90]
            if len(core_iois) >= 4:
                core_mean = float(np.mean(core_iois))
                core_std = float(np.std(core_iois))
                f.ioi_cv_core = round(core_std / core_mean, 4) if core_mean > 0 else 0

        # 局部加速/拖拍检测：滑动窗口 (窗口大小=5)
        # 吉他演奏中扫弦(IOI极小)和旋律(IOI较大)自然交替，阈值需放宽以避免误报
        win_size = min(5, len(iois) - 1)
        if win_size >= 3:
            rush_regions = []
            drag_regions = []
            for i in range(len(iois) - win_size + 1):
                window = iois[i:i + win_size]
                win_mean = float(np.mean(window))
                # rush: 窗口均值比全局均值快40%以上（放宽避免扫弦误报）
                if win_mean < mean_ioi * 0.60:
                    start_t = filtered_onsets[i]
                    end_t = filtered_onsets[i + win_size]
                    rush_regions.append({
                        "start": round(start_t, 1),
                        "end": round(end_t, 1),
                        "ratio": round(win_mean / mean_ioi, 2),
                    })
                # drag: 窗口均值比全局均值慢60%以上
                elif win_mean > mean_ioi * 1.6:
                    start_t = filtered_onsets[i]
                    end_t = filtered_onsets[i + win_size]
                    drag_regions.append({
                        "start": round(start_t, 1),
                        "end": round(end_t, 1),
                        "ratio": round(win_mean / mean_ioi, 2),
                    })
            # 合并重叠/相邻的 rush+drag 对（扫弦→旋律的自然交替不算异常）
            # 重叠或间隔 < 1s 的 rush→drag 对视为吉他演奏的正常交替，一并剔除
            f.local_rush_regions, f.local_drag_regions = self._filter_alternating_regions(
                rush_regions, drag_regions)

    @staticmethod
    def _filter_alternating_regions(rush_regions, drag_regions):
        """过滤扫弦→旋律自然交替造成的虚假 rush/drag 对。

        吉他的扫弦产生极短 IOI（看起来像加速），紧接着旋律段落 IOI 变长（看起来像拖拍）。
        这种交替是正常演奏模式切换，不应报告为节奏问题。
        合并重叠或相邻(间隔<1s)的 rush+drag 对。"""
        if not rush_regions or not drag_regions:
            return rush_regions, drag_regions

        # 找出需要移除的对：rush 和 drag 区域在时间上重叠或紧密相邻
        paired_rush = set()
        paired_drag = set()
        for ri, r in enumerate(rush_regions):
            for di, d in enumerate(drag_regions):
                # 检查时间重叠或相邻（间隔 < 1s）
                gap = max(0, d["start"] - r["end"], r["start"] - d["end"])
                overlap = min(r["end"], d["end"]) - max(r["start"], d["start"])
                if overlap > -1.0:  # 重叠或间隔 < 1s
                    paired_rush.add(ri)
                    paired_drag.add(di)

        filtered_rush = [r for i, r in enumerate(rush_regions) if i not in paired_rush]
        filtered_drag = [d for i, d in enumerate(drag_regions) if i not in paired_drag]
        return filtered_rush, filtered_drag

    def _compute_dynamic_features(self, notes: List[Dict], f: AudioFeatures):
        """计算动态范围特征。"""
        velocities = [n.get("velocity", 0) for n in notes if n.get("velocity", 0) > 0]
        if len(velocities) < 4:
            return

        sorted_v = sorted(velocities)
        n = len(sorted_v)
        f.velocity_p5 = round(float(sorted_v[int(n * 0.05)]), 1)
        f.velocity_p50 = round(float(sorted_v[int(n * 0.50)]), 1)
        f.velocity_p95 = round(float(sorted_v[int(n * 0.95)]), 1)

        # 动态范围 (dB): 20 * log10(P95 / P5)
        if f.velocity_p5 > 0.5:
            f.dynamic_range_db = round(20 * np.log10(f.velocity_p95 / f.velocity_p5), 1)

    def _compute_spectral_features(self, audio: np.ndarray, sr: int, f: AudioFeatures):
        """计算频谱/音色/杂音特征。"""
        try:
            import librosa
        except ImportError:
            logger.info("librosa 未安装，跳过频谱分析")
            return

        # 频谱质心 (亮度指标)
        cent = librosa.feature.spectral_centroid(y=audio.astype(np.float32), sr=sr)
        f.spectral_centroid_avg = round(float(np.mean(cent)), 0)

        # 起音速度：onset strength 上升时间
        onset_env = librosa.onset.onset_strength(y=audio.astype(np.float32), sr=sr)
        if len(onset_env) > 5:
            # 检测起音事件
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            if len(onset_frames) > 1:
                hop_length = 512
                attack_times = []
                for of in onset_frames[:20]:  # 取前20个起音
                    start = max(0, of * hop_length - int(0.02 * sr))
                    end = min(len(audio), of * hop_length + int(0.15 * sr))
                    segment = np.abs(audio[start:end])
                    if len(segment) > 100:
                        peak_idx = np.argmax(segment)
                        if peak_idx > 5:
                            rise_10pct = np.where(segment[:peak_idx] > segment[peak_idx] * 0.1)[0]
                            rise_90pct = np.where(segment[:peak_idx] > segment[peak_idx] * 0.9)[0]
                            if len(rise_10pct) > 0 and len(rise_90pct) > 0:
                                attack_time_ms = (rise_90pct[0] - rise_10pct[0]) / sr * 1000
                                if 0 < attack_time_ms < 200:
                                    attack_times.append(attack_time_ms)
                if attack_times:
                    f.attack_time_avg_ms = round(float(np.median(attack_times)), 1)

        # 泛音衰减斜率: 谐波 vs 冲击成分分离
        try:
            y_harm, y_perc = librosa.effects.hpss(audio.astype(np.float32))
            harm_energy = float(np.sum(y_harm ** 2))
            perc_energy = float(np.sum(y_perc ** 2))
            total = harm_energy + perc_energy
            if total > 0:
                f.inharmonicity_ratio = round(perc_energy / total, 4)

            # 高频噪声: > 8kHz 能量占比
            if sr >= 22050:
                D = np.abs(np.fft.rfft(audio.astype(np.float32)))
                freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
                high_mask = freqs > 8000
                if np.sum(D) > 0:
                    f.high_freq_noise_ratio = round(float(np.sum(D[high_mask]) / np.sum(D)), 4)
        except Exception as e:
            logger.warning(f"HPSS/噪声分析失败: {e}")

    def _build_feature_summary(self, f: AudioFeatures) -> str:
        """将定量特征转换为文字摘要，供 LLM 解读使用。"""
        parts = []

        # 节拍 (only report when there's enough data — 5+ distinct onset events)
        if f.tempo_bpm > 0:
            parts.append(f"速度: {f.tempo_bpm:.0f} BPM")
            # 吉他阈值：扫弦+旋律混合天然 CV 偏高，阈值比一般乐器宽松
            parts.append(f"节拍稳定性: CV(全局)={f.ioi_cv:.3f} " +
                          ("(优秀)" if f.ioi_cv < 0.20 else
                           "(良好)" if f.ioi_cv < 0.35 else
                           "(一般)" if f.ioi_cv < 0.50 else "(不稳定)"))
            if f.ioi_cv_core > 0:
                parts.append(f"核心节奏稳定性: CV(核心)={f.ioi_cv_core:.3f} " +
                              ("(优秀, 排除离群后节奏均匀)" if f.ioi_cv_core < 0.25 else
                               "(良好/正常, 扫弦+旋律的正常混合)" if f.ioi_cv_core < 0.45 else
                               "(偏不稳)" if f.ioi_cv_core < 0.55 else "(不稳定)"))
        if f.local_rush_regions:
            parts.append(f"加速段落: {len(f.local_rush_regions)}处")
        if f.local_drag_regions:
            parts.append(f"拖拍段落: {len(f.local_drag_regions)}处")

        # 动态
        parts.append(f"动态范围: {f.dynamic_range_db:.1f}dB, " +
                      f"velocity P5={f.velocity_p5:.0f} P50={f.velocity_p50:.0f} P95={f.velocity_p95:.0f} " +
                      ("(表现力丰富)" if f.dynamic_range_db > 15 else
                       "(变化适中)" if f.dynamic_range_db > 8 else "(较平淡)"))
        # 弱音/哑音检测：P5<20 说明存在明显没弹响的音
        if f.velocity_p5 < 15:
            parts.append("⚠️ 存在极弱音/哑音(velocity P5<15)，部分音符可能没弹响")
        elif f.velocity_p5 < 25:
            parts.append("⚠️ 存在偏弱音符(velocity P5<25)，可能有少数音不够清晰")

        # 音色
        if f.spectral_centroid_avg > 0:
            parts.append(f"音色亮度: 频谱质心={f.spectral_centroid_avg:.0f}Hz " +
                          ("(明亮)" if f.spectral_centroid_avg > 2000 else
                           "(适中)" if f.spectral_centroid_avg > 1000 else "(偏暗)"))
        if f.attack_time_avg_ms > 0:
            parts.append(f"起音速度: {f.attack_time_avg_ms:.0f}ms " +
                          ("(清晰利落)" if f.attack_time_avg_ms < 30 else
                           "(正常)" if f.attack_time_avg_ms < 60 else "(偏慢/偏软)"))

        # 杂音
        parts.append(f"非谐波能量比: {f.inharmonicity_ratio:.3f} " +
                      ("(干净)" if f.inharmonicity_ratio < 0.15 else
                       "(正常)" if f.inharmonicity_ratio < 0.30 else "(杂音较多)"))
        if f.high_freq_noise_ratio > 0:
            parts.append(f"高频噪声占比: {f.high_freq_noise_ratio:.3f} " +
                          ("(低)" if f.high_freq_noise_ratio < 0.05 else
                           "(中)" if f.high_freq_noise_ratio < 0.10 else "(高)"))

        return "；".join(parts)

    # ==================== 技巧检测：从 model_output + audio_features ====================

    def detect_technique_segments(self, model_output, audio_features: AudioFeatures,
                                   notes: List[Dict], sr: int = 16000) -> List[Dict]:
        """从 pitch activation matrix + audio features 推断技巧时间段。

        Args:
            model_output: basic-pitch 的 model_output, shape [n_frames, 88] (88 MIDI pitches)
            audio_features: 已计算的 AudioFeatures
            notes: basic-pitch 转录音符
            sr: 采样率

        Returns:
            [{start_time, end_time, technique_id, confidence}, ...]
        """
        segments = []
        mo = np.array(model_output)  # [n_frames, 88]
        n_frames, n_bins = mo.shape
        if n_frames < 10 or len(notes) < 3:
            return segments

        # frame → time mapping: basic-pitch hop_length=512 @16000Hz
        hop_length = 512
        frame_time = lambda f: f * hop_length / sr
        time_to_frame = lambda t: int(t * sr / hop_length)

        # Helper: get activation at a specific MIDI pitch over time
        midi_offset = 21  # basic-pitch bins start at MIDI 21 (A0)
        def pitch_activation(midi_pitch):
            bin_idx = midi_pitch - midi_offset
            if 0 <= bin_idx < n_bins:
                return mo[:, bin_idx]
            return np.zeros(n_frames)

        # ── 泛音检测 (Harmonics) ──
        self._detect_harmonics_segments(mo, notes, audio_features, frame_time, midi_offset, segments)

        # ── 连奏检测 (Legato: hammer-on / pull-off / slide) ──
        self._detect_legato_segments(notes, segments)

        # ── PM/闷音检测 (Palm Muting) ──
        self._detect_pm_segments(mo, notes, audio_features, frame_time, midi_offset, segments)

        # ── 扫弦检测 (Strumming) — 音符级 + activation matrix 交叉验证 ──
        self._detect_strumming_segments(notes, segments, mo, frame_time)

        # Sort by start_time and merge overlapping segments of same technique
        segments.sort(key=lambda s: (s["start_time"], s["technique_id"]))
        return self._merge_segments(segments)

    def _detect_harmonics_segments(self, mo, notes, af, frame_time, midi_offset, segments):
        """检测泛音：泛音列激活模式 + 频谱特征。

        泛音的严格特征（必须同时满足多项）：
        - 基频能量极低（轻触不按实，基频大幅衰减）
        - 某个泛音列频率的能量远高于基频（≥2.5倍）
        - 起音极快（<10ms，轻触即刻发声）
        - 持续时间短（<800ms，泛音衰减快）
        - 非谐波能量极低（频谱干净）

        为了防止正常按弦被误判为泛音，阈值设得很保守。
        宁可漏判不可误判：漏判只是报告不提泛音，误判会导致AI胡说。
        """
        n_frames = mo.shape[0]
        midi_offset_val = midi_offset
        HARMONIC_INTERVALS = [12, 19, 24, 28]
        total_notes = len(notes)
        harmonic_count = 0

        for note in notes:
            midi = note.get("pitch", 60)
            t0 = max(0, int(note["start_time"] * 16000 / 512))
            t1 = min(n_frames, int(note["end_time"] * 16000 / 512) + 1)
            duration_frames = t1 - t0
            if duration_frames < 3:
                continue

            fund_bin = midi - midi_offset_val
            if fund_bin < 0 or fund_bin >= mo.shape[1]:
                continue

            fund_act = float(np.mean(mo[t0:t1, fund_bin]))
            if fund_act < 0.008:
                continue

            # 基频能量必须极低（<0.03）——正常按弦的基频能量通常在0.05以上
            if fund_act > 0.03:
                continue

            # 检查所有泛音列
            best_ratio = 0.0
            for interval in HARMONIC_INTERVALS:
                harmonic_bin = fund_bin + interval
                if harmonic_bin < mo.shape[1]:
                    harmonic_act = float(np.mean(mo[t0:t1, harmonic_bin]))
                    ratio = harmonic_act / max(fund_act, 0.001)
                    if ratio > best_ratio:
                        best_ratio = ratio

            # 泛音判定：泛音列能量必须远强于基频（≥2.5）
            if best_ratio < 2.5:
                continue

            # 持续时间必须短（泛音衰减快）
            note_duration = note.get("end_time", 0) - note.get("start_time", 0)
            if note_duration > 0.8:
                continue

            # 确认不是所有音符都被判为泛音（泛音在全曲中占比应<15%）
            harmonic_count += 1

            confidence = min(0.85, best_ratio * 0.25)
            tech = "natural-harmonics"
            segments.append({
                "start_time": note["start_time"],
                "end_time": note["end_time"],
                "technique_id": tech,
                "confidence": round(confidence, 2),
            })

        # 如果泛音占比过高（>15%），说明判断有误，全部放弃
        if total_notes > 0 and harmonic_count / total_notes > 0.15:
            logger.info(f"泛音占比过高({harmonic_count}/{total_notes})，放弃全部泛音检测结果")
            segments[:] = [s for s in segments if s.get("technique_id") not in
                          ("natural-harmonics", "artificial-harmonics")]

    def _detect_legato_segments(self, notes, segments):
        """检测连奏：短间隔连续音符 → hammer-on/pull-off/slide。
        优先用弦号验证，但DP弦号分配可能不准，因此也允许仅凭时间+音高+力度判断。"""
        for i in range(1, len(notes)):
            prev = notes[i - 1]
            curr = notes[i]
            gap = curr["start_time"] - prev["end_time"]
            pitch_diff = abs(curr["pitch"] - prev["pitch"])
            # 同弦验证：最优
            same_string = (prev.get("string") == curr.get("string")
                           and prev.get("string") is not None
                           and prev.get("string") > 0)
            # 击勾弦：必须同弦 + 无缝衔接 + 低力度（三者缺一不可）
            # 仅凭音频无法可靠区分「快速拨弦」和「击勾弦」，必须保守
            if same_string and 0 <= gap < 0.04 and pitch_diff >= 1:
                v_curr = curr.get("velocity", 0)
                v_prev = prev.get("velocity", 0)
                if v_curr < v_prev * 0.7:
                    tech = "hammer-on" if curr["pitch"] > prev["pitch"] else "pull-off"
                    segments.append({
                        "start_time": prev["start_time"],
                        "end_time": curr["end_time"],
                        "technique_id": tech,
                        "confidence": 0.70,
                    })
            # 滑音：间隔<150ms，音高差1-12半音
            if 0 <= gap < 0.15 and 1 <= abs(pitch_diff) <= 12:
                if same_string:
                    direction = "上滑" if curr["pitch"] > prev["pitch"] else "下滑"
                    segments.append({
                        "start_time": prev["start_time"],
                        "end_time": curr["end_time"],
                        "technique_id": "slide",
                        "confidence": 0.70,
                        "detail": f"{direction}: {prev.get('note_name','')}→{curr.get('note_name','')}",
                    })
                elif prev.get("velocity", 0) < 45 or curr.get("velocity", 0) < 45:
                    # 力度低说明大概率是滑音而非两次拨弦，即使DP分配了不同弦
                    direction = "上滑" if curr["pitch"] > prev["pitch"] else "下滑"
                    segments.append({
                        "start_time": prev["start_time"],
                        "end_time": curr["end_time"],
                        "technique_id": "slide",
                        "confidence": 0.50,
                        "detail": f"{direction}: {prev.get('note_name','')}→{curr.get('note_name','')}",
                    })

    def _detect_pm_segments(self, mo, notes, af, frame_time, midi_offset, segments):
        """检测 PM/闷音：快起音 + 快衰减"""
        if af.attack_time_avg_ms > 20 or af.inharmonicity_ratio < 0.35:
            return
        midi_offset_val = midi_offset
        n_frames = mo.shape[0]
        for note in notes[:30]:
            midi = note.get("pitch", 60)
            t0 = max(0, int(note["start_time"] * 16000 / 512))
            t1 = min(n_frames, t0 + int(0.15 * 16000 / 512))  # 150ms window
            if t1 - t0 < 3:
                continue
            bin_idx = midi - midi_offset_val
            if 0 <= bin_idx < mo.shape[1]:
                envelope = mo[t0:t1, bin_idx]
                if len(envelope) < 3:
                    continue
                peak = np.max(envelope)
                if peak < 0.02:
                    continue
                # Check fast decay: activation drops to <30% within 100ms
                decay_frames = min(len(envelope), int(0.10 * 16000 / 512))
                if decay_frames >= 2:
                    post_peak = envelope[1:decay_frames]
                    if len(post_peak) > 0 and np.mean(post_peak) < peak * 0.35:
                        segments.append({
                            "start_time": note["start_time"],
                            "end_time": note["end_time"],
                            "technique_id": "pm-technique",
                            "confidence": 0.65,
                        })

    def _detect_strumming_segments(self, notes, segments, mo=None, frame_time=None):
        """检测扫弦：音符级聚类 + activation matrix 交叉验证。

        两路并行：
        1. 音符级：多音符在极短时间内同时触发（用 pitch 离散度代替 string 字段）
        2. Activation级：pitch activation matrix 中多频带同时跳变（不依赖品位分配）
        两路交叉验证提高召回率和准确率。
        """
        if len(notes) < 3:
            return

        STRUM_WINDOW = 0.020  # 20ms（扫弦拨片划过6弦约需20-40ms）
        MIN_PITCH_SPREAD = 12  # 至少12半音跨度（约4根弦的跨度）
        MIN_CLUSTER = 4       # 至少4个音符集群
        MAX_SPREAD = 0.030    # cluster 内最大时间跨度（匹配STRUM_WINDOW）

        # ── 路1: 音符级检测 ──
        note_strum_regions = set()  # (start_time_rounded, end_time_rounded)
        sorted_notes = sorted(notes, key=lambda n: n["start_time"])
        i = 0
        while i < len(sorted_notes):
            cluster = [sorted_notes[i]]
            j = i + 1
            while j < len(sorted_notes) and sorted_notes[j]["start_time"] - cluster[0]["start_time"] < STRUM_WINDOW:
                cluster.append(sorted_notes[j])
                j += 1
            if len(cluster) >= MIN_CLUSTER:
                # 用 pitch 跨度代替 string 计数（basic-pitch 不输出 string 字段）
                pitches = [n.get("pitch", 60) for n in cluster]
                pitch_spread = max(pitches) - min(pitches)
                has_string_info = any(n.get("string") for n in cluster)
                if has_string_info:
                    strings = set(n.get("string") for n in cluster if n.get("string"))
                    valid = len(strings) >= 4
                else:
                    valid = pitch_spread >= MIN_PITCH_SPREAD
                if valid:
                    spread = max(n["start_time"] for n in cluster) - min(n["start_time"] for n in cluster)
                    if spread <= MAX_SPREAD:
                        velocities = [n.get("velocity", 0) for n in cluster]
                        if velocities and max(velocities) > 0:
                            mean_v = sum(velocities) / len(velocities)
                            cv = (sum((v - mean_v)**2 for v in velocities) / len(velocities))**0.5 / mean_v if mean_v > 0 else 1
                        else:
                            cv = 0.5
                        if spread < 0.006 and cv < 0.35:
                            confidence = 0.85
                        elif spread < 0.010 and cv < 0.50:
                            confidence = 0.70
                        elif spread < 0.015:
                            confidence = 0.55
                        else:
                            confidence = 0.40
                        t0 = cluster[0]["start_time"]
                        t1 = max(n["end_time"] for n in cluster)
                        note_strum_regions.add((round(t0, 1), round(t1, 1)))
                        segments.append({
                            "start_time": t0,
                            "end_time": t1,
                            "technique_id": "strumming",
                            "confidence": confidence,
                            "source": "notes",
                        })
            i = j if j > i + 1 else i + 1

        # ── 路2: Activation matrix 检测（不依赖品位分配）──
        if mo is not None and frame_time is not None:
            self._detect_strumming_from_activation(
                mo, frame_time, segments, note_strum_regions
            )

    def _detect_strumming_from_activation(self, mo, frame_time, segments, note_regions=None):
        """从 pitch activation matrix 检测扫弦。

        原理：扫弦时大量弦同时被触发，pitch activation 矩阵中相邻帧间
        激活的 MIDI pitch bin 数量会突然跳升。这不依赖品位分配，直接分析频谱激活模式。
        """
        try:
            import numpy as np
            mo_arr = np.array(mo)  # [n_frames, n_bins]
            if mo_arr.ndim != 2 or mo_arr.shape[0] < 5:
                return

            # 每帧激活的 pitch bin 数量（阈值 0.15 过滤低能量噪声）
            n_active = (mo_arr > 0.15).sum(axis=1)

            # 扫弦特征：相邻帧激活数跳变 >12 且跳变后 >6 个 bin 同时激活
            for f in range(1, len(n_active)):
                jump = n_active[f] - n_active[f - 1]
                if jump > 12 and n_active[f] > 6:
                    t = float(frame_time(f))
                    # 检查是否与音符级检测的区域重叠
                    if note_regions:
                        # 如果音符级已经在这个时间段检测到了，跳过激活级检测
                        overlapped = any(abs(t - nr[0]) < 0.3 for nr in note_regions)
                        if overlapped:
                            continue
                    segments.append({
                        "start_time": max(0, t - 0.03),
                        "end_time": t + 0.12,
                        "technique_id": "strumming",
                        "confidence": min(0.75, 0.45 + jump / 40),
                        "source": "activation",
                    })

            # 第二类信号：持续高激活（>12 bins 持续2帧以上，且前面有跳变）
            # 说明有多根弦在同时振动 → 可能是扫弦后的延音
            # 限制最大持续时长 1.5s，超过的可能是持续和弦而非扫弦
            i = 0
            while i < len(n_active) - 2:
                if n_active[i] > 12:
                    j = i + 1
                    while j < len(n_active) and n_active[j] > 12:
                        j += 1
                    sustain_frames = j - i
                    if sustain_frames >= 2:  # 至少2帧
                        t0 = float(frame_time(i))
                        t1 = float(frame_time(min(j - 1, len(n_active) - 1)))
                        duration = t1 - t0
                        # 持续太久（>1.5s）更可能是按住的持续和弦，不是扫弦
                        if duration > 1.5:
                            i = j
                            continue
                        # 检查前面 5 帧内是否有跳变（扫弦起音的特征）
                        pre_start = max(0, i - 5)
                        pre_max = max(n_active[pre_start:i]) if i > pre_start else 0
                        pre_jump = n_active[i] - pre_max
                        has_attack = pre_jump > 6
                        # 如果前面没有明显跳变且持续时间长，跳过（更可能是持续和弦）
                        if not has_attack and duration > 0.5:
                            i = j
                            continue
                        # 检查是否与已有检测重叠
                        overlaps = False
                        for seg in segments:
                            if seg.get("technique_id") == "strumming":
                                if abs(seg["start_time"] - t0) < 0.15:
                                    overlaps = True
                                    break
                        if not overlaps:
                            confidence = min(0.75, 0.35 + sustain_frames * 0.08 + (0.1 if has_attack else 0))
                            segments.append({
                                "start_time": t0,
                                "end_time": t1,
                                "technique_id": "strumming",
                                "confidence": confidence,
                                "source": "activation_sustain",
                            })
                    i = j
                else:
                    i += 1
        except Exception as e:
            logger.debug(f"activation strumming detection skipped: {e}")

    def _extract_cluster_features(self, seg: List[Dict]) -> Dict:
        """提取单个onset cluster的特征向量，用于技巧指纹匹配。

        Returns:
            {n_strokes, avg_ioi, ioi_cv, sixteenth_ratio,
             avg_energy, energy_cv, stroke_density, avg_notes_per_stroke, ...}
        """
        n_total = len(seg)
        iois = []
        for i in range(1, n_total):
            ioi = seg[i]["time"] - seg[i - 1]["time"]
            if 0.03 < ioi < 2.0:
                iois.append(ioi)

        avg_ioi = sum(iois) / len(iois) if iois else 0.5
        ioi_cv = 0.0
        if len(iois) >= 3 and avg_ioi > 0:
            std_ioi = (sum((x - avg_ioi) ** 2 for x in iois) / len(iois)) ** 0.5
            ioi_cv = std_ioi / avg_ioi

        duration = seg[-1]["time"] - seg[0]["time"]
        stroke_density = n_total / duration if duration > 0 else 0

        labels = [s.get("duration_label", "") for s in seg]
        n_sixteenth = labels.count("十六分")
        n_labeled = sum(1 for l in labels if l and l != "?")
        sixteenth_ratio = n_sixteenth / n_labeled if n_labeled > 0 else 0.0

        energies = [s.get("_energy", 0.0) for s in seg]
        avg_energy = sum(energies) / n_total if n_total > 0 else 0.0
        energy_cv = 0.0
        if avg_energy > 1e-9 and n_total >= 4:
            std_e = (sum((e - avg_energy) ** 2 for e in energies) / n_total) ** 0.5
            energy_cv = std_e / avg_energy

        # 频谱特征: spectral spread + peak count（区分多弦 vs 单弦）
        spreads = [s.get("_spectral_spread", 0.0) for s in seg]
        peaks = [s.get("_peak_count", 0) for s in seg]
        avg_spread = sum(spreads) / n_total if n_total > 0 else 0.0
        avg_peaks = sum(peaks) / n_total if n_total > 0 else 0.0
        peak_cv = 0.0
        if avg_peaks > 0.5 and n_total >= 4:
            std_p = (sum((p - avg_peaks) ** 2 for p in peaks) / n_total) ** 0.5
            peak_cv = std_p / avg_peaks

        # basic-pitch 音符特征: 直接度量多弦同时发声（替代频谱猜测）
        # n_notes: stroke 内同时触发的音符数 → 1=单弦, 2-4=柱式, 4-6=扫弦
        # pitch_spread: stroke 内音高跨度（半音）→ 大=多弦跨度大
        notes_per_stroke = [s.get("_n_notes", 0) for s in seg]
        pitch_spreads = [s.get("_pitch_spread", 0) for s in seg]
        avg_notes = sum(notes_per_stroke) / n_total if n_total > 0 else 0.0
        avg_pitch_spread = sum(pitch_spreads) / n_total if n_total > 0 else 0.0
        notes_cv = 0.0
        if avg_notes > 0.5 and n_total >= 4:
            std_n = (sum((x - avg_notes) ** 2 for x in notes_per_stroke) / n_total) ** 0.5
            notes_cv = std_n / avg_notes

        # 包络攻击时长: 扫弦=15-40ms逐步建立, 单音/琶音=3-10ms瞬间建立
        attack_durations = [s.get("_attack_duration", 0.0) for s in seg]
        valid_atk = [a for a in attack_durations if a > 0.5]
        avg_attack_dur = sum(valid_atk) / len(valid_atk) if valid_atk else 0.0

        # 音符时间散布: 扫弦stroke内音符有5-20ms时差(拨片划过), 柱式≈0ms
        note_spreads = [s.get("_note_time_spread", 0.0) for s in seg]
        avg_note_spread = sum(note_spreads) / n_total if n_total > 0 else 0.0

        return {
            "n_strokes": n_total,
            "avg_ioi": round(avg_ioi, 4),
            "ioi_cv": round(ioi_cv, 3),
            "sixteenth_ratio": round(sixteenth_ratio, 3),
            "avg_energy": round(avg_energy, 6),
            "energy_cv": round(energy_cv, 3),
            "stroke_density": round(stroke_density, 1),
            "avg_spectral_spread": round(avg_spread, 1),
            "avg_peak_count": round(avg_peaks, 2),
            "peak_count_cv": round(peak_cv, 2),
            "avg_notes_per_stroke": round(avg_notes, 2),
            "notes_cv": round(notes_cv, 2),
            "avg_pitch_spread": round(avg_pitch_spread, 1),
            "avg_attack_duration": round(avg_attack_dur, 1),
            "avg_within_stroke_spread_ms": round(avg_note_spread, 1),
        }

    def _compute_onset_bursts(self, audio: np.ndarray, sr: int,
                              t_start: float, t_end: float):
        """Analyze onset burst pattern in raw audio segment.

        Strumming physically produces tight clusters of string onsets (pick
        sweeps across 6 strings in 20-40ms), while decomposition produces
        evenly-spaced single-note onsets. This function detects the pattern
        directly from raw audio onsets, bypassing basic-pitch entirely.

        Returns:
            (burst_ratio, avg_burst_size, n_bursts, n_onsets)
            - burst_ratio: proportion of onsets that belong to bursts (≥3 onsets)
            - avg_burst_size: mean onsets per burst
        """
        start_sample = max(0, int(t_start * sr))
        end_sample = min(len(audio), int(t_end * sr))
        if end_sample - start_sample < int(sr * 0.5):
            return 0.0, 1.0, 0, 0

        seg = audio[start_sample:end_sample].astype(np.float32)

        try:
            onset_frames = librosa.onset.onset_detect(
                y=seg, sr=sr, hop_length=512,
                backtrack=True,
            )
            onset_times = librosa.frames_to_time(
                onset_frames, sr=sr, hop_length=512
            )
            onsets = [t_start + float(t) for t in onset_times]
        except Exception:
            return 0.0, 1.0, 0, 0

        if len(onsets) < 4:
            return 0.0, 1.0, 0, len(onsets)

        # Group onsets within 50ms into bursts (pick sweep across 6 strings = 20-40ms,
        # but librosa onset detection may merge very close onsets, so widen to 50ms)
        bursts = []
        current = [onsets[0]]
        for t in onsets[1:]:
            if t - current[0] <= 0.050:
                current.append(t)
            else:
                bursts.append(current)
                current = [t]
        bursts.append(current)

        burst_sizes = [len(b) for b in bursts]
        # Count bursts of size >= 2 (librosa may merge adjacent string onsets)
        total_in_bursts = sum(s for s in burst_sizes if s >= 2)
        burst_ratio = total_in_bursts / len(onsets) if len(onsets) > 0 else 0.0
        avg_burst_size = sum(burst_sizes) / len(bursts) if bursts else 1.0

        return burst_ratio, avg_burst_size, len(bursts), len(onsets)

    def _compute_onset_peak_widths(self, audio: np.ndarray, sr: int,
                                    t_start: float, t_end: float):
        """高时间分辨率 onset 峰宽分析。

        扫弦时多弦在20-40ms内依次触发，onset strength envelope 在峰附近
        持续偏高（宽峰）。单音/琶音瞬间触发，峰窄。
        此特征不依赖方向检测，不依赖 basic-pitch 准确性。

        Returns:
            (wide_peak_ratio, avg_peak_width_ms)
            - wide_peak_ratio: 宽峰(>18ms)占所有峰的比例
            - avg_peak_width_ms: 所有峰的平均半峰宽(ms)
        """
        start_sample = max(0, int(t_start * sr))
        end_sample = min(len(audio), int(t_end * sr))
        if end_sample - start_sample < int(sr * 1.0):
            return 0.0, 0.0

        seg = audio[start_sample:end_sample].astype(np.float32)
        hop_length = 128  # 8ms at 16kHz

        try:
            onset_env = librosa.onset.onset_strength(y=seg, sr=sr, hop_length=hop_length)
        except Exception:
            return 0.0, 0.0

        if len(onset_env) < 4:
            return 0.0, 0.0

        onset_arr = np.array(onset_env)
        env_max = float(np.max(onset_arr))
        if env_max < 1e-9:
            return 0.0, 0.0

        # Find peaks: local maxima above 25% of global max
        peak_frames = []
        threshold = env_max * 0.25
        for i in range(1, len(onset_arr) - 1):
            if onset_arr[i] > threshold and onset_arr[i] > onset_arr[i - 1] and onset_arr[i] > onset_arr[i + 1]:
                peak_frames.append(i)

        if len(peak_frames) < 2:
            return 0.0, 0.0

        # Measure half-height width for each peak
        widths_ms = []
        for pi in peak_frames:
            half_h = onset_arr[pi] * 0.5
            left = pi
            while left > 0 and onset_arr[left] > half_h:
                left -= 1
            right = pi
            while right < len(onset_arr) - 1 and onset_arr[right] > half_h:
                right += 1
            width_frames = right - left
            width_ms = width_frames * hop_length / sr * 1000
            if width_ms > 0:
                widths_ms.append(width_ms)

        if not widths_ms:
            return 0.0, 0.0

        avg_width = sum(widths_ms) / len(widths_ms)
        n_wide = sum(1 for w in widths_ms if w > 18)
        wide_ratio = n_wide / len(widths_ms)

        return round(wide_ratio, 3), round(avg_width, 1)

    def _compute_centroid_slope(self, audio: np.ndarray, sr: int, center_sample: int) -> float:
        """频谱质心漂移斜率 (Hz/ms)，区分依次拨弦(扫弦) vs 同时击弦(柱式)。

        扫弦: 拨片依次划过6根弦 → 频谱质心随时间漂移（下扫低→高，上扫高→低）
        柱式: 所有弦同时击中 → 频谱质心基本恒定

        Returns:
            abs(slope) in Hz/ms. >30 表示明显漂移 → 倾向于扫弦。
        """
        ws = max(0, center_sample - int(sr * 0.010))
        we = min(len(audio), center_sample + int(sr * 0.030))
        if we - ws < int(sr * 0.015):
            return 0.0

        seg = audio[ws:we].astype(np.float64)
        win_len = int(sr * 0.008)
        hop = int(sr * 0.002)
        n_windows = max(1, (len(seg) - win_len) // hop)
        if n_windows < 3:
            return 0.0

        centroids = []
        for i in range(n_windows):
            s = i * hop
            chunk = seg[s:s + win_len]
            if len(chunk) < 16:
                continue
            fft = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
            freqs = np.fft.rfftfreq(len(chunk), d=1.0 / sr)
            total = float(np.sum(fft))
            if total > 1e-12:
                centroids.append(float(np.sum(freqs * fft) / total))

        if len(centroids) < 3:
            return 0.0

        times = np.arange(len(centroids)) * 2.0
        centroids = np.array(centroids)
        slope = np.polyfit(times, centroids, 1)[0]
        return round(abs(float(slope)), 1)

    def _compute_hf_tail_ratio(self, audio: np.ndarray, sr: int, center_sample: int) -> float:
        """高频拖尾比：扫弦拨片噪声持续15-40ms，柱式瞬间消失。

        在 attack 窗口内，比较高频(>4kHz)能量在 15-30ms 段 vs 0-5ms 段。
        扫弦: >0.5 (高频噪声持续)   柱式: <0.3 (高频集中在前5ms)

        Returns:
            hf_tail_ratio (0.0-2.0). 越高越像扫弦。
        """
        ws = max(0, center_sample - int(sr * 0.010))
        we = min(len(audio), center_sample + int(sr * 0.035))
        if we - ws < int(sr * 0.020):
            return 1.0

        seg = audio[ws:we].astype(np.float64)
        hop = max(1, int(sr * 0.0015))
        win_len = hop * 2
        n_frames = max(1, (len(seg) - win_len) // hop)

        hf_early = 0.0
        hf_tail = 0.0
        for i in range(n_frames):
            t_ms = i * 1.5
            s = i * hop
            chunk = seg[s:s + win_len]
            if len(chunk) < 8:
                continue
            fft = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
            freqs = np.fft.rfftfreq(len(chunk), d=1.0 / sr)
            hf_mask = freqs >= 4000
            hf_energy = float(np.sum(fft[hf_mask]))

            if t_ms < 5:
                hf_early += hf_energy
            elif 15 <= t_ms <= 30:
                hf_tail += hf_energy

        if hf_early < 1e-12:
            return 1.0  # no HF at all → can't tell
        return round(hf_tail / hf_early, 2)

    def _compute_spectral_peak_density(self, audio: np.ndarray, sr: int,
                                        strokes: List[Dict]) -> float:
        """频谱峰值密度：区分单音(能量集中少数谐波)和多弦(能量分散)。

        核心指标：top 5 最强频率峰占总能量的比例(能量集中度)。
        单音→0.50-0.80(能量集中在基频+前几个谐波)
        多弦(4-6)→0.10-0.30(能量分散在多个弦的谐波中)

        使用150ms窗口(而非60ms)，频率分辨率~6.7Hz，足够分辨谐波结构差异。
        """
        concentration_vals = []
        for s in strokes:
            t = s.get("time", 0)
            center = int(t * sr)
            ws = max(0, center - int(sr * 0.075))
            we = min(len(audio), center + int(sr * 0.075))
            n_samples = we - ws
            if n_samples < 256:
                continue

            seg = audio[ws:we].astype(np.float64)
            spec = np.abs(np.fft.rfft(seg * np.hanning(n_samples))) ** 2
            # Skip DC and only look at guitar-relevant range (60Hz-4000Hz)
            bin_min = max(1, int(60 * n_samples / sr))
            bin_max = min(len(spec), int(4000 * n_samples / sr))
            guitar_spec = spec[bin_min:bin_max]
            if len(guitar_spec) < 20 or np.max(guitar_spec) < 1e-12:
                continue

            # Energy concentration: ratio of top-5 peaks to total energy
            total_energy = float(np.sum(guitar_spec))
            if total_energy < 1e-12:
                continue
            sorted_spec = np.sort(guitar_spec)[::-1]
            top5_energy = float(np.sum(sorted_spec[:5]))
            concentration = top5_energy / total_energy
            concentration_vals.append(float(np.clip(concentration, 0.0, 1.0)))

        if not concentration_vals:
            return 0.50  # default: assume tonal/single note

        avg_concentration = sum(concentration_vals) / len(concentration_vals)
        return round(avg_concentration, 3)

    def _classify_onset_cluster(self, fp: Dict, is_single: bool, global_median_energy: float,
                                  tempo_bpm: float = 0) -> Dict:
        """技巧分类：能量为主（扫弦6弦=2-3x单音能量），频谱为辅。"""
        scores = {"strumming": 0.0, "arpeggio": 0.0, "broken_chord": 0.0, "block_chord": 0.0, "double_stop": 0.0}
        n = fp["n_strokes"]
        avg_energy = fp.get("avg_energy", 0.0)

        # ── Energy gate ──
        if global_median_energy > 1e-9 and avg_energy < global_median_energy * 0.2:
            return {
                "type": "unknown", "confidence": 0.0,
                "scores": {"strumming": 0.0},
                "_reason": f"low_energy({avg_energy:.6f})",
            }

        # ── Energy ratio (within-video relative) ──
        if global_median_energy > 1e-9:
            energy_ratio = avg_energy / global_median_energy
        else:
            energy_ratio = 1.0

        # ── PRIMARY: energy signal ──
        # Tier 1: absolute high energy (>0.10 = likely strumming on phone recordings)
        # Tier 2: relative standout within video (>1.5x median)
        is_high_energy = avg_energy > 0.12 or energy_ratio > 1.5
        is_low_energy = avg_energy < 0.05 and energy_ratio < 0.8

        if is_high_energy:
            atk = fp.get("avg_attack_duration", 0.0)
            if atk >= 1.0:
                if atk > 12:
                    scores["strumming"] += 0.35
                    scores["block_chord"] += 0.15
                elif atk > 5:
                    scores["strumming"] += 0.20
                    scores["block_chord"] += 0.25
                else:
                    scores["block_chord"] += 0.35
                    scores["strumming"] += 0.05
            else:
                scores["strumming"] += 0.35
                scores["block_chord"] += 0.15

            # Counterweight: high energy but single-note plucks → strong fingerpicking,
            # not strumming. 根3231323 with loud bass notes should not be strumming.
            avg_notes = fp.get("avg_notes_per_stroke", 0)
            if avg_notes < 2.0:
                scores["arpeggio"] += 0.25
                scores["strumming"] -= 0.15

            if n >= 5 and fp.get("ioi_cv", 1.0) < 0.50:
                scores["strumming"] += 0.15
        elif is_low_energy:
            scores["arpeggio"] += 0.35
            scores["strumming"] -= 0.20
        else:
            # Moderate energy: ambiguous
            scores["arpeggio"] += 0.15
            scores["strumming"] += 0.10

        # ── PRIMARY-B: onset peak width (direction-independent) ──
        wide_ratio = fp.get("wide_peak_ratio", 0.0)
        avg_peak_w = fp.get("avg_peak_width_ms", 0.0)
        avg_notes = fp.get("avg_notes_per_stroke", 0)
        if wide_ratio >= 0.30:
            scores["strumming"] += 0.25
            scores["arpeggio"] -= 0.15
        elif wide_ratio >= 0.20 and avg_peak_w > 15:
            scores["strumming"] += 0.12
        if wide_ratio < 0.10 and avg_notes < 2.0:
            scores["strumming"] -= 0.15

        # ── SECONDARY: spectral spread (reduced weight) ──
        spread = fp.get("avg_spectral_spread", 0.0)
        if spread > 1500:
            scores["strumming"] += 0.10
        elif spread > 1200 and is_high_energy:
            scores["strumming"] += 0.05
        elif spread < 400:
            scores["strumming"] -= 0.15

        # ── TERTIARY: burst + peaks ──
        burst_ratio = fp.get("onset_burst_ratio", 0.0)
        peaks = fp.get("avg_peak_count", 0.0)
        ioi_cv = fp.get("ioi_cv", 1.0)

        if burst_ratio > 0.60 and is_high_energy:
            scores["strumming"] += 0.10
        if peaks > 5:
            scores["strumming"] += 0.10

        # ── Rhythm bonus ──
        if ioi_cv < 0.35 and n >= 5:
            if is_high_energy:
                scores["strumming"] += 0.10
            else:
                scores["arpeggio"] += 0.15

        # ── Anti-strumming ──
        if is_low_energy and spread < 600:
            scores["strumming"] -= 0.25
        if burst_ratio < 0.15 and not is_high_energy:
            scores["strumming"] -= 0.15
        if peaks < 3 and not is_high_energy:
            scores["strumming"] -= 0.10

        # ── Specific types ──
        if fp["sixteenth_ratio"] > 0.50 and not is_high_energy:
            scores["arpeggio"] += 0.15
        if n > 12 and not is_high_energy:
            scores["broken_chord"] += 0.15
        if n <= 6 and is_high_energy:
            scores["block_chord"] += 0.15

        # ── Double-stop detection ──
        # 双音/三音：音符数介于琶音(1)和扫弦(4+)之间，瞬时触弦(窄峰)而非扫过(宽峰)
        avg_notes = fp.get("avg_notes_per_stroke", 0)
        note_spread = fp.get("avg_within_stroke_spread_ms", 0)
        wide_ratio_ds = fp.get("wide_peak_ratio", 0.0)
        if 1.3 <= avg_notes <= 3.5 and not is_high_energy:
            if note_spread < 25 and wide_ratio_ds < 0.25:
                scores["double_stop"] += 0.40
                scores["arpeggio"] -= 0.15
                scores["strumming"] -= 0.15
                if avg_notes >= 2.0:
                    scores["double_stop"] += 0.15
                if note_spread < 12:
                    scores["double_stop"] += 0.10
        elif avg_notes >= 2.0 and avg_notes <= 4.0 and is_high_energy:
            if note_spread < 20 and wide_ratio_ds < 0.30:
                scores["double_stop"] += 0.25
                scores["block_chord"] += 0.10

        # ── Pick best ──
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        if best_score <= 0.0:
            best_type = "unknown"

        return {
            "type": best_type,
            "confidence": round(min(1.0, max(0.0, best_score)), 2),
            "scores": {t: round(s, 2) for t, s in sorted(scores.items(), key=lambda x: -x[1])},
        }

    def _split_mixed_cluster(self, seg: List[Dict]) -> List[List[Dict]]:
        """检测cluster内的技巧切换点，拆分成独立子段。

        滑动窗口对比相邻5-stroke窗口的IOI和能量变化：
        - IOI突然缩>40%: 琶音→扫弦过渡
        - 能量突然增>50%: 琶音→扫弦过渡
        """
        n = len(seg)
        if n < 12:
            return [seg]  # 太短，不拆

        window = 5
        split_at = None
        for i in range(window, n - window + 1):
            left = seg[i - window:i]
            right = seg[i:i + window]

            left_iois = [left[j]["time"] - left[j - 1]["time"] for j in range(1, len(left))]
            right_iois = [right[j]["time"] - right[j - 1]["time"] for j in range(1, len(right))]
            avg_left = sum(left_iois) / len(left_iois) if left_iois else 0.1
            avg_right = sum(right_iois) / len(right_iois) if right_iois else 0.1

            left_energies = [s.get("_energy", 0) for s in left]
            right_energies = [s.get("_energy", 0) for s in right]
            avg_e_left = sum(left_energies) / len(left_energies) if left_energies else 0
            avg_e_right = sum(right_energies) / len(right_energies) if right_energies else 0

            ioi_shrink = avg_left > 0.02 and avg_right > 0 and avg_right < avg_left * 0.6
            energy_jump = avg_e_left > 1e-9 and avg_e_right > avg_e_left * 1.5

            if ioi_shrink or energy_jump:
                split_at = i
                logger.info(f"段内切分: ioi {avg_left*1000:.0f}→{avg_right*1000:.0f}ms, "
                            f"energy {avg_e_left:.4f}→{avg_e_right:.4f}")
                break

        if split_at and split_at >= 3 and (n - split_at) >= 3:
            return [seg[:split_at], seg[split_at:]]
        return [seg]

    def _detect_strumming_pattern(self, audio: np.ndarray, sr: int = 16000,
                                    strum_windows: List[Dict] = None,
                                    notes: List[Dict] = None) -> List[Dict]:
        """检测扫弦节奏型：识别拨弦方向 + IOI量化 + 节奏型识别。

        改进：
        - 频谱质心（spectral centroid）替代固定频带比判断方向
        - IOI 量化到音乐时值（四分/八分/十六分等）
        - 自适应节奏型匹配

        Returns:
            [{start_time, end_time,
              strokes: [{time, direction, confidence, duration_label}],
              rhythm_pattern: str,  # e.g. "四分 四分 前八后十六"
              pattern_label: str,    # e.g. "下 下 下上"
              confidence: float}, ...]
        """
        if len(audio) < sr * 1.0:
            return []

        if not strum_windows:
            return []

        try:
            audio_f32 = audio.astype(np.float32)

            # 1. 基于 basic-pitch 音符进行 stroke 分组（替代 librosa onset）
            # basic-pitch 提供精确的音符起始时间 + 音高，直接区分单弦/多弦
            if not notes or len(notes) < 3:
                return []
            sorted_notes = sorted(notes, key=lambda n: n["start_time"])

            # 2. 将 20ms 内的音符归为同一个 stroke（同时触弦 = 同一 stroke）
            # 扫弦拨片划过6根弦约需20-40ms，12ms太窄会拆分单次扫弦为多个stroke
            strokes_raw = []
            i = 0
            while i < len(sorted_notes):
                cluster_notes = [sorted_notes[i]]
                j = i + 1
                while j < len(sorted_notes) and sorted_notes[j]["start_time"] - cluster_notes[0]["start_time"] < 0.020:
                    cluster_notes.append(sorted_notes[j])
                    j += 1

                t = sum(n["start_time"] for n in cluster_notes) / len(cluster_notes)
                pitches = [n["pitch"] for n in cluster_notes]
                n_notes = len(cluster_notes)
                pitch_spread = max(pitches) - min(pitches)

                # 3. 对每个 stroke 做 pitch ordering 方向判定（使用 raw audio）
                # 原理：扫弦瞬间物理传播有5-15ms时序差
                #       下扫: 6弦→1弦, 低音先到; 上扫: 1弦→6弦, 高音先到
                center = int(t * sr)
                sub_windows = []
                for offset_ms in [0, 10, 20]:
                    ws = center + int(sr * offset_ms / 1000)
                    we = ws + int(sr * 0.005)
                    if 0 <= ws < len(audio_f32) and we <= len(audio_f32) and (we - ws) >= 16:
                        seg = audio_f32[ws:we]
                        fft = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
                        freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)
                        low_mask = freqs < 800
                        high_mask = freqs > 2000
                        low_energy = float(np.sum(fft[low_mask]))
                        high_energy = float(np.sum(fft[high_mask]))
                        sub_windows.append((low_energy, high_energy))
                    else:
                        sub_windows.append((0.0, 0.0))

                low_vals = [sw[0] for sw in sub_windows]
                high_vals = [sw[1] for sw in sub_windows]
                if max(low_vals) > 1e-9 and max(high_vals) > 1e-9:
                    low_peak_idx = low_vals.index(max(low_vals))
                    high_peak_idx = high_vals.index(max(high_vals))
                    if low_peak_idx < high_peak_idx:
                        direction = "down"
                        conf = 0.75 if high_peak_idx - low_peak_idx >= 1 else 0.60
                    elif high_peak_idx < low_peak_idx:
                        direction = "up"
                        conf = 0.75 if low_peak_idx - high_peak_idx >= 1 else 0.60
                    else:
                        total_low = sum(low_vals)
                        total_high = sum(high_vals)
                        ratio = total_high / (total_low + 1e-9)
                        if ratio > 1.3:
                            direction = "up"
                            conf = 0.50
                        elif ratio < 0.7:
                            direction = "down"
                            conf = 0.50
                        else:
                            direction = "?"
                            conf = 0.35
                else:
                    direction = "?"
                    conf = 0.30

                # 4. 计算该 stroke 的 RMS 能量
                ws_full = max(0, center - int(sr * 0.015))
                we_full = min(len(audio_f32), center + int(sr * 0.015))
                if we_full > ws_full:
                    seg_full = audio_f32[ws_full:we_full]
                    energy = float(np.sqrt(np.mean(seg_full.astype(np.float64) ** 2)))
                else:
                    energy = 0.0

                # 5. 计算频谱特征: spectral spread + peak count
                spectral_spread = 0.0
                peak_count = 0
                if we_full > ws_full and (we_full - ws_full) >= 16:
                    seg_spec = audio_f32[ws_full:we_full]
                    fft_spec = np.abs(np.fft.rfft(seg_spec * np.hanning(len(seg_spec))))
                    freqs_spec = np.fft.rfftfreq(len(seg_spec), d=1.0 / sr)
                    total_mag = float(np.sum(fft_spec))
                    if total_mag > 1e-12:
                        centroid = float(np.sum(freqs_spec * fft_spec)) / total_mag
                        spread = float(np.sqrt(np.sum(((freqs_spec - centroid) ** 2) * fft_spec) / total_mag))
                        spectral_spread = round(spread, 1)
                    if len(fft_spec) >= 5 and total_mag > 1e-12:
                        mag_max = float(np.max(fft_spec))
                        threshold = mag_max * 0.20
                        peaks = 0
                        for k in range(2, len(fft_spec) - 2):
                            if (fft_spec[k] > threshold
                                    and fft_spec[k] > fft_spec[k-1]
                                    and fft_spec[k] > fft_spec[k-2]
                                    and fft_spec[k] > fft_spec[k+1]
                                    and fft_spec[k] > fft_spec[k+2]):
                                peaks += 1
                        peak_count = peaks

                # 6. 包络攻击时长：扫弦拨片划过6弦需20-40ms能量逐步建立，
                #    单音/琶音拨弦能量瞬间(3-8ms)建立，物理上不可混淆
                attack_duration = 0.0
                aw_start = max(0, center - int(sr * 0.025))
                aw_end = min(len(audio_f32), center + int(sr * 0.050))
                if aw_end - aw_start >= int(sr * 0.010):
                    aw_seg = audio_f32[aw_start:aw_end]
                    env_hop = max(1, int(sr * 0.0015))
                    n_env = max(2, (len(aw_seg) - env_hop) // env_hop)
                    rms_env = np.zeros(n_env)
                    for ei in range(n_env):
                        s0 = ei * env_hop
                        rms_env[ei] = np.sqrt(np.mean(aw_seg[s0:s0 + env_hop].astype(np.float64) ** 2))
                    rms_max = float(np.max(rms_env))
                    if rms_max > 1e-9:
                        lo_th = rms_max * 0.15
                        hi_th = rms_max * 0.85
                        t_lo = None
                        t_hi = None
                        for ei in range(n_env):
                            if t_lo is None and rms_env[ei] >= lo_th:
                                t_lo = ei
                            if t_lo is not None and t_hi is None and rms_env[ei] >= hi_th:
                                t_hi = ei
                                break
                        if t_lo is not None and t_hi is not None and t_hi > t_lo:
                            attack_duration = round((t_hi - t_lo) * 0.0015 * 1000, 1)

                # 计算 stroke 内音符的时间散布（ms）：扫弦5-20ms，柱式0-2ms
                note_times = [n["start_time"] for n in cluster_notes]
                note_time_spread = round((max(note_times) - min(note_times)) * 1000, 1)


                strokes_raw.append({
                    "time": round(float(t), 3),
                    "direction": direction,
                    "confidence": conf,
                    "_energy": energy,
                    "_spectral_spread": spectral_spread,
                    "_peak_count": peak_count,
                    "_n_notes": n_notes,
                    "_pitch_spread": pitch_spread,
                    "_attack_duration": attack_duration,
                    "_note_time_spread": note_time_spread,
                })
                i = j

            if len(strokes_raw) < 4:
                return []

            # stroke-level dedup: 相邻 stroke 时间差 < 30ms 且同方向 → 保留能量最大的
            strokes = [strokes_raw[0]]
            for s in strokes_raw[1:]:
                if s["time"] - strokes[-1]["time"] < 0.030 and s["direction"] == strokes[-1]["direction"]:
                    if s["_energy"] > strokes[-1]["_energy"]:
                        strokes[-1] = s
                else:
                    strokes.append(s)

            if len(strokes) < 4:
                return []

            # 6. 方向交替逻辑校验：连续3个同向 → 中间那次可能是误判
            for i in range(1, len(strokes) - 1):
                if (strokes[i]["direction"] == strokes[i-1]["direction"] == strokes[i+1]["direction"]
                        and strokes[i]["direction"] != "?"
                        and strokes[i]["confidence"] < 0.65):
                    strokes[i]["direction"] = "up" if strokes[i]["direction"] == "down" else "down"
                    strokes[i]["confidence"] = 0.45

            # 4. 聚类（相邻 stroke 间隔 > 1.2s 断开）
            if not strokes:
                return []
            clusters = []
            current = [strokes[0]]
            for s in strokes[1:]:
                if s["time"] - current[-1]["time"] < 1.2:
                    current.append(s)
                else:
                    if len(current) >= 3:
                        clusters.append(current)
                    current = [s]
            if len(current) >= 3:
                clusters.append(current)

            if not clusters:
                return []

            # 4b. 段内切分检测（琶音→扫弦无缝过渡）
            split_clusters = []
            for seg in clusters:
                split_clusters.extend(self._split_mixed_cluster(seg))
            clusters = split_clusters

            # 5. 全局统计（用于匹配法）
            is_single_cluster = len(clusters) == 1
            all_energies = [s.get("_energy", 0.0) for seg in clusters for s in seg]
            global_median_energy = float(np.median(all_energies)) if all_energies else 0.0

            # 6. 匹配法分类：每个 cluster 独立量化节奏型 + 提取特征 + 指纹匹配
            results = []
            for seg in clusters:
                # 6a. Per-cluster IOI 量化
                seg_iois = []
                for i in range(1, len(seg)):
                    ioi = seg[i]["time"] - seg[i-1]["time"]
                    if 0.03 < ioi < 2.0:
                        seg_iois.append(ioi)

                rhythm_pattern = ""
                duration_labels = []
                tempo_bpm = 0
                if len(seg_iois) >= 3:
                    rhythm_pattern, duration_labels, tempo_bpm = self._quantize_rhythm(seg_iois)
                    # 为 stroke 打时值标签
                    for i, label in enumerate(duration_labels):
                        if i < len(seg):
                            seg[i]["duration_label"] = label

                # 6b. 提取特征向量
                fp = self._extract_cluster_features(seg)

                # 6b2. Raw audio onset burst analysis (primary strumming signal)
                t0_seg = seg[0]["time"]
                t1_seg = seg[-1]["time"]
                br, abs_val, n_bursts, n_onsets = self._compute_onset_bursts(
                    audio_f32, sr, t0_seg, t1_seg
                )
                fp["onset_burst_ratio"] = br
                fp["avg_onset_burst_size"] = abs_val

                # 6b3. Onset peak width analysis (direction-independent strumming signal)
                wide_ratio, avg_width_ms = self._compute_onset_peak_widths(
                    audio_f32, sr, t0_seg, t1_seg
                )
                fp["wide_peak_ratio"] = wide_ratio
                fp["avg_peak_width_ms"] = avg_width_ms

                # 6b4. Spectral peak density: energy concentration → single vs multi-string
                fp["avg_band_count"] = self._compute_spectral_peak_density(
                    audio_f32, sr, seg
                )

                # 6c. 匹配法分类
                classification = self._classify_onset_cluster(
                    fp, is_single=is_single_cluster,
                    global_median_energy=global_median_energy,
                    tempo_bpm=tempo_bpm,
                )
                ctype = classification["type"]
                cconf = classification["confidence"]
                scores_str = ", ".join(f"{t}={s:.2f}" for t, s in classification["scores"].items())

                logger.info(f"cluster分类: {ctype} (conf={cconf:.2f}) [{scores_str}] "
                            f"n={fp['n_strokes']} energy={fp.get('avg_energy',0):.4f} "
                            f"atk={fp.get('avg_attack_duration',0):.0f}ms "
                            f"spread={fp.get('avg_spectral_spread',0):.0f}Hz "
                            f"peaks={fp.get('avg_peak_count',0):.1f} burst={fp.get('onset_burst_ratio',0):.0%}")

                # 6d. 组装结果
                directions = [s.get("direction", "?") for s in seg]
                n_down = directions.count("down")
                n_up = directions.count("up")
                n_unknown = directions.count("?")

                direction_str = " ".join("下" if d == "down" else ("上" if d == "up" else "?") for d in directions)
                if len(direction_str) > 60:
                    direction_str = direction_str[:60] + "..."

                result = {
                    "type": ctype,
                    "type_confidence": cconf,
                    "type_scores": classification["scores"],
                    "start_time": seg[0]["time"],
                    "end_time": seg[-1]["time"],
                    "strokes": [{k: v for k, v in s.items() if not k.startswith("_")} for s in seg],
                    "pattern_label": direction_str,
                    "pattern_summary": f"{n_down}下{n_up}上" + (f"({n_unknown}?)" if n_unknown else ""),
                    "confidence": cconf,
                }
                if rhythm_pattern:
                    result["rhythm_pattern"] = rhythm_pattern
                    result["tempo_bpm"] = round(tempo_bpm, 1)
                if duration_labels:
                    result["duration_labels"] = duration_labels
                results.append(result)

            # 6e. 全局上下文重新评定（仅抑制明显误判：stroke 数少且孤立）
            # 不再做大范围降级——频谱分类已有保守阈值，全局重分类容易误杀
            total_strokes_all = sum(len(seg) for seg in clusters)
            for i, r in enumerate(results):
                if r["type"] == "strumming" and r["type_confidence"] < 0.50:
                    n_strokes = len(clusters[i]) if i < len(clusters) else 0
                    # 只有置信度极低（<0.50）且 stroke 极少的孤例才降级
                    if n_strokes <= 4 and len(results) > 3:
                        logger.info(f"全局重分类(极低置信+孤立): strumming→unknown "
                                    f"(n={n_strokes} conf={r['type_confidence']:.2f})")
                        r["type"] = "unknown"

            type_summary = ", ".join(f"{r['type']}({r.get('rhythm_pattern', '?')[:20]})" for r in results[:3])
            logger.info(f"Onset cluster分类: {len(results)} 段 → {type_summary}")
            return results

        except Exception as e:
            logger.warning(f"扫弦节奏型检测失败: {e}")
            return []

    def _quantize_rhythm(self, iois: List[float]) -> tuple:
        """将 IOI 序列量化到音乐时值网格。

        步骤：
        1. 找最小 IOI 作为候选十六分音符
        2. 计算 tempo（假设 min_ioi≈十六分音符，min_ioi*4=四分音符时长）
        3. 将每个 IOI 映射到最近的标准时值
        4. 生成中文节奏型描述

        Returns:
            (rhythm_pattern_str, duration_labels_list, tempo_bpm)
            e.g. ("四分 四分 前八后十六", ["四分","四分","八分","十六分","十六分"], 90.0)
        """
        import numpy as np

        if len(iois) < 3:
            return "", [], 0

        # 去掉极端离群值（可能是漏检或误检）
        sorted_iois = sorted(iois)
        # 使用中位数附近的 IOI 估算基本单位
        mid_iois = sorted_iois[len(sorted_iois)//4 : 3*len(sorted_iois)//4]
        if not mid_iois:
            return "", [], 0
        min_ioi = mid_iois[0]  # 估计的最小单位

        # 假设 min_ioi 是十六分音符 → 四分音符 = min_ioi * 4
        quarter_duration = min_ioi * 4
        tempo_bpm = 60.0 / quarter_duration if quarter_duration > 0 else 0

        # 合理 tempo 范围 (40-200 BPM)
        if tempo_bpm > 200:
            # min_ioi 可能是八分音符
            quarter_duration = min_ioi * 2
            tempo_bpm = 60.0 / quarter_duration
        elif tempo_bpm < 40:
            # min_ioi 可能是三十二分音符
            quarter_duration = min_ioi * 8
            tempo_bpm = 60.0 / quarter_duration

        if tempo_bpm <= 0 or tempo_bpm > 250:
            return "", [], 0

        # 标准时值（以四分音符 = 1.0 为单位）
        DURATIONS = {
            "十六分": 0.25,
            "八分": 0.50,
            "附点八分": 0.75,
            "四分": 1.0,
            "附点四分": 1.5,
            "二分": 2.0,
        }
        # 中文标签映射
        LABEL_MAP = {
            "十六分": "十六分",
            "八分": "八分",
            "附点八分": "附点八分",
            "四分": "四分",
            "附点四分": "附点四分",
            "二分": "二分",
        }

        duration_labels = []
        for ioi in iois:
            ratio = ioi / quarter_duration
            best_label = "?"
            best_diff = float("inf")
            for name, target in DURATIONS.items():
                diff = abs(ratio - target)
                if diff < best_diff and diff < 0.35:  # max tolerance
                    best_diff = diff
                    best_label = LABEL_MAP.get(name, "?")
            duration_labels.append(best_label)

        # 生成节奏型描述（合并重复模式）
        # e.g. ["四分","四分","八分","十六分","十六分"] → "四分 四分 前八后十六"
        pattern_parts = []
        i = 0
        while i < len(duration_labels):
            # 检测 "八分 十六分 十六分" → "前八后十六"
            if (i + 2 < len(duration_labels)
                    and duration_labels[i] == "八分"
                    and duration_labels[i+1] == "十六分"
                    and duration_labels[i+2] == "十六分"):
                pattern_parts.append("前八后十六")
                i += 3
                continue
            # 检测 "十六分 十六分 八分" → "前十六后八"
            if (i + 2 < len(duration_labels)
                    and duration_labels[i] == "十六分"
                    and duration_labels[i+1] == "十六分"
                    and duration_labels[i+2] == "八分"):
                pattern_parts.append("前十六后八")
                i += 3
                continue
            # 检测 "十六分 八分 十六分" → "切分"
            if (i + 2 < len(duration_labels)
                    and duration_labels[i] == "十六分"
                    and duration_labels[i+1] == "八分"
                    and duration_labels[i+2] == "十六分"):
                pattern_parts.append("切分音")
                i += 3
                continue
            # 检测 "附点八分 十六分" → "附点"
            if (i + 1 < len(duration_labels)
                    and duration_labels[i] == "附点八分"
                    and duration_labels[i+1] == "十六分"):
                pattern_parts.append("附点节奏")
                i += 2
                continue
            pattern_parts.append(duration_labels[i])
            i += 1

        # 压缩连续重复：["四分","四分","四分"] → "四分×3"
        compressed = []
        j = 0
        while j < len(pattern_parts):
            count = 1
            while j + count < len(pattern_parts) and pattern_parts[j+count] == pattern_parts[j]:
                count += 1
            if count > 1:
                compressed.append(f"{pattern_parts[j]}×{count}")
            else:
                compressed.append(pattern_parts[j])
            j += count

        rhythm_pattern = " ".join(compressed)

        # 如果 pattern 太长，简化描述
        if len(rhythm_pattern) > 60:
            unique_parts = list(dict.fromkeys(compressed))  # 去重保持顺序
            rhythm_pattern = " ".join(unique_parts[:6])
            if len(unique_parts) > 6:
                rhythm_pattern += " …"

        return rhythm_pattern, duration_labels, tempo_bpm

    def _evaluate_strumming_quality(self, strumming_patterns, audio, technique_segments=None, sr=16000):
        """评估扫弦质量：基于节奏型量化的精准度分析。

        改进 v3:
        - 只评估 type="strumming" 的 cluster
        - Per-cluster 节奏型合并（不再只读 patterns[0]）
        - dynamic_variation_score 替代 dynamic_consistency（U型曲线）

        Returns:
            dict: 质量评估结果，或 None（无扫弦段时）
        """
        if not strumming_patterns:
            return None

        # 只取扫弦类型的 cluster
        strum_only = [p for p in strumming_patterns if p.get("type") == "strumming"]
        if not strum_only:
            logger.info(f"扫弦质量评估: 无 strumming 类型 cluster（"
                        f"已有类型: {set(p.get('type','?') for p in strumming_patterns)}），跳过")
            return None

        all_strokes = []
        for pat in strum_only:
            all_strokes.extend(pat.get("strokes", []))

        if len(all_strokes) < 4:
            return None

        all_strokes.sort(key=lambda s: s["time"])

        # 1. 节奏精度：per-cluster 独立计算后再综合
        iois = []
        for i in range(1, len(all_strokes)):
            ioi = all_strokes[i]["time"] - all_strokes[i - 1]["time"]
            if 0.03 < ioi < 2.0:
                iois.append(ioi)

        # Per-cluster: 每段独立做 timing 评估，然后加权平均
        merged_rhythm_patterns = []
        merged_tempo_bpms = []
        all_timing_errors = []  # [(error, weight), ...]
        all_details_parts = []

        for pat in strum_only:
            rp = pat.get("rhythm_pattern", "")
            dl = pat.get("duration_labels", [])
            tb = pat.get("tempo_bpm", 0)
            seg_strokes = pat.get("strokes", [])

            if rp:
                merged_rhythm_patterns.append(rp)
            if tb > 0:
                merged_tempo_bpms.append(tb)

            # Per-cluster IOI
            seg_iois = []
            for i in range(1, len(seg_strokes)):
                sioi = seg_strokes[i]["time"] - seg_strokes[i-1]["time"]
                if 0.03 < sioi < 2.0:
                    seg_iois.append(sioi)

            if dl and len(dl) >= 2 and tb > 0 and seg_iois:
                quarter_dur = 60.0 / tb
                DURATION_IDEAL = {
                    "十六分": quarter_dur * 0.25, "八分": quarter_dur * 0.50,
                    "附点八分": quarter_dur * 0.75, "四分": quarter_dur * 1.00,
                    "附点四分": quarter_dur * 1.50, "二分": quarter_dur * 2.00,
                }
                WEIGHTS = {"十六分": 0.6, "八分": 0.8, "附点八分": 1.0, "四分": 1.0, "附点四分": 1.0, "二分": 0.8}
                cat_errors = {}
                n_pairs = min(len(dl), len(seg_iois))
                for j in range(n_pairs):
                    label = dl[j]
                    if label in DURATION_IDEAL:
                        ideal = DURATION_IDEAL[label]
                        actual = seg_iois[j]
                        error_ratio = abs(actual - ideal) / ideal if ideal > 0 else 0
                        cat_errors.setdefault(label, []).append(error_ratio)

                for cat, errors in cat_errors.items():
                    if errors:
                        avg_err = sum(errors) / len(errors)
                        w = WEIGHTS.get(cat, 0.5)
                        all_timing_errors.append((avg_err, w * len(errors)))
                        if avg_err < 0.12:
                            level = "精准"
                        elif avg_err < 0.22:
                            level = "良好"
                        elif avg_err < 0.35:
                            level = "有偏差"
                        else:
                            level = "不稳定"
                        all_details_parts.append(f"{cat}音符({'×'+str(len(errors))}): {level}")

        rhythm_pattern = " | ".join(merged_rhythm_patterns) if merged_rhythm_patterns else ""
        tempo_bpm = sum(merged_tempo_bpms) / len(merged_tempo_bpms) if merged_tempo_bpms else 0

        timing_accuracy = 0.85  # default
        timing_label = "一般"
        rhythm_details = ""

        if all_timing_errors:
            total_error = sum(e * w for e, w in all_timing_errors)
            total_weight = sum(w for _, w in all_timing_errors)
            if total_weight > 0:
                avg_error = total_error / total_weight
                timing_accuracy = max(0.1, 1.0 - avg_error * 1.2)
                if avg_error < 0.10:
                    timing_label = "精准"
                elif avg_error < 0.18:
                    timing_label = "良好"
                elif avg_error < 0.30:
                    timing_label = "有偏差"
                else:
                    timing_label = "不稳"
            rhythm_details = ", ".join(all_details_parts) if all_details_parts else ""

        # fallback: 原始 CV 计算（无节奏型信息时使用）
        rhythm_cv = 0.0
        mean_ioi = 0.0
        if len(iois) >= 3:
            mean_ioi = sum(iois) / len(iois)
            std_ioi = (sum((x - mean_ioi) ** 2 for x in iois) / len(iois)) ** 0.5
            rhythm_cv = std_ioi / mean_ioi if mean_ioi > 0 else 0

        # 2. 力度变化感知评估（U型曲线）
        # 使用 pre-computed _energy（pitch ordering 阶段已计算）
        energy_list = [s.get("_energy", 0.0) for s in all_strokes]
        # fallback: 重新计算 RMS
        if all(e < 1e-9 for e in energy_list):
            energy_list = []
            window_samples = int(sr * 0.03)
            for s in all_strokes:
                center = int(s["time"] * sr)
                start = max(0, center - window_samples // 2)
                end = min(len(audio), center + window_samples // 2)
                if end > start:
                    energy = float(np.sqrt(np.mean(audio[start:end].astype(np.float64) ** 2)))
                    energy_list.append(energy)

        # 6. Spectral flux: 触弦瞬态力度（对录音条件更鲁棒）
        energy_list_flux = []
        window_samples_sf = int(sr * 0.015)
        for s in all_strokes:
            center = int(s["time"] * sr)
            ws1_start = max(0, center - window_samples_sf)
            ws1_end = center
            ws2_start = center
            ws2_end = min(len(audio), center + window_samples_sf)
            if ws1_end > ws1_start and ws2_end > ws2_start:
                seg1 = audio[ws1_start:ws1_end].astype(np.float64)
                seg2 = audio[ws2_start:ws2_end].astype(np.float64)
                fft1 = np.abs(np.fft.rfft(seg1 * np.hanning(len(seg1)))) if len(seg1) > 4 else np.array([0])
                fft2 = np.abs(np.fft.rfft(seg2 * np.hanning(len(seg2)))) if len(seg2) > 4 else np.array([0])
                min_len = min(len(fft1), len(fft2))
                flux = float(np.sum(np.abs(fft2[:min_len] - fft1[:min_len])))
                energy_list_flux.append(flux)

        # RMS + Spectral Flux 加权平均作为最终力度值
        if energy_list_flux and len(energy_list_flux) >= 4 and max(energy_list_flux) > 0:
            max_flux = max(energy_list_flux)
            norm_flux = [f / max_flux for f in energy_list_flux]
            if len(energy_list) == len(norm_flux):
                energy_list = [e * 0.5 + nf * max(e, 1e-9) * 0.5
                              for e, nf in zip(energy_list, norm_flux)]

        dynamic_cv = 0.0
        dynamic_variation = 0.9  # default: good
        if len(energy_list) >= 4:
            mean_e = sum(energy_list) / len(energy_list)
            if mean_e > 0:
                std_e = (sum((e - mean_e) ** 2 for e in energy_list) / len(energy_list)) ** 0.5
                dynamic_cv = std_e / mean_e
                # U型曲线：适度变化最好，太均匀=无表现力，太不均匀=失控
                if dynamic_cv < 0.08:
                    dynamic_variation = 0.5  # 过于均匀，缺乏轻重对比
                elif dynamic_cv < 0.25:
                    dynamic_variation = 0.9  # 理想区间：有自然轻重变化
                elif dynamic_cv < 0.45:
                    dynamic_variation = 0.7  # 波动偏大
                else:
                    dynamic_variation = 0.4  # 不稳定

        # 3. 方向分析：使用改进后的方向检测结果
        directions = [s.get("direction", "?") for s in all_strokes]
        direction_issues = []
        alt_breaks = 0
        unknown_count = directions.count("?")
        # 只统计低置信度方向之间的交替断裂
        for i in range(1, len(directions)):
            di = directions[i]
            di_prev = directions[i - 1]
            if di != "?" and di_prev != "?" and di == di_prev:
                # 检查是否可能是刻意同向（如连续下扫的重音练习）
                conf_i = all_strokes[i].get("confidence", 0.5)
                conf_p = all_strokes[i - 1].get("confidence", 0.5)
                if conf_i < 0.55 or conf_p < 0.55:
                    alt_breaks += 1  # 低置信度同向 = 可能是误判

        down_count = directions.count("down")
        up_count = directions.count("up")
        # 有合理的上下比例才是好的扫弦
        if up_count == 0 and down_count > 6:
            direction_issues.append("全部识别为下扫，可能未做上扫或方向检测不准确")
        elif up_count < down_count * 0.2 and down_count > 8:
            direction_issues.append(f"上扫偏少（{up_count}上/{down_count}下），手腕可能不够灵活")
        if alt_breaks > len(directions) * 0.15:
            direction_issues.append(f"有{alt_breaks}处方向交替不自然")
        if unknown_count > len(directions) * 0.3:
            direction_issues.append(f"方向识别不清比例偏高({unknown_count}/{len(directions)})")

        # 4. 速度漂移
        tempo_drift = 0.0
        tempo_issue = ""
        if len(iois) >= 6:
            mid = len(iois) // 2
            early_mean = sum(iois[:mid]) / mid
            late_mean = sum(iois[mid:]) / len(iois[mid:])
            if early_mean > 0:
                tempo_drift = (late_mean - early_mean) / early_mean
                if abs(tempo_drift) > 0.15:
                    direction_word = "减速" if tempo_drift > 0 else "加速"
                    tempo_issue = f"明显{direction_word}（{abs(tempo_drift)*100:.0f}%），建议全程跟节拍器练习"
                elif abs(tempo_drift) > 0.08:
                    direction_word = "减速趋势" if tempo_drift > 0 else "加速趋势"
                    tempo_issue = f"略有{direction_word}（{abs(tempo_drift)*100:.0f}%）"

        # 5. 综合评分（pattern-aware 权重）
        direction_score = max(0.2, 1.0 - alt_breaks * 0.15 - unknown_count * 0.05)
        if up_count > 0 and down_count > 0:
            up_ratio = min(up_count, down_count) / max(up_count, down_count) if max(up_count, down_count) > 0 else 0
            direction_score += up_ratio * 0.2  # bonus for having both directions
        direction_score = min(1.0, direction_score)
        drift_score = 1.0 - min(0.3, abs(tempo_drift) * 0.5)

        overall = timing_accuracy * 0.35 + dynamic_variation * 0.25 + direction_score * 0.25 + drift_score * 0.15
        overall = max(0.1, min(1.0, overall))

        # 6. 教学反馈
        feedback_parts = []

        # 节奏反馈
        if rhythm_details:
            feedback_parts.append(f"计时分析[{timing_label}]: {rhythm_details}")
        elif rhythm_cv < 0.12:
            feedback_parts.append("节奏基本稳定，扫弦律动感好")
        elif rhythm_cv < 0.22:
            feedback_parts.append("节奏有轻微不均匀，建议用节拍器练习")
        else:
            feedback_parts.append(f"节奏不稳定，建议用节拍器60-70BPM单独练节奏型")

        if rhythm_pattern:
            feedback_parts.append(f"节奏型: {rhythm_pattern}")
            if tempo_bpm:
                feedback_parts.append(f"估计速度: {tempo_bpm:.0f} BPM")

        # 力度反馈（U型曲线三档 + 无变化档）
        if dynamic_cv < 0.08:
            feedback_parts.append("力度过于均匀，建议练习在强拍加重力度、弱拍减小力度，增强律动感")
        elif dynamic_cv < 0.25:
            feedback_parts.append("力度控制自然，有轻重变化，上下扫音量平衡")
        elif dynamic_cv < 0.45:
            feedback_parts.append("力度有波动，注意上下扫力度均匀")
        else:
            feedback_parts.append("力度不均匀，建议单独练习下扫和上扫的力度控制")

        if tempo_issue:
            feedback_parts.append(tempo_issue)

        for di in direction_issues:
            feedback_parts.append(di)

        quality = {
            "overall_score": round(overall, 2),
            "timing_accuracy": round(timing_accuracy, 2),
            "timing_label": timing_label,
            "rhythm_cv": round(rhythm_cv, 3),
            "mean_ioi": round(mean_ioi, 3),
            "dynamic_variation": round(dynamic_variation, 2),
            "dynamic_cv": round(dynamic_cv, 3),
            "direction_issues": direction_issues,
            "tempo_drift": round(tempo_drift, 3),
            "tempo_issue": tempo_issue,
            "alternation_breaks": alt_breaks,
            "down_count": down_count,
            "up_count": up_count,
            "stroke_count": len(all_strokes),
            "unknown_direction_count": unknown_count,
            "rhythm_details": rhythm_details,
            "feedback_summary": "；".join(feedback_parts),
            "rhythm_pattern": rhythm_pattern,
            "tempo_bpm": round(tempo_bpm, 1) if tempo_bpm else 0,
            "stroke_details": [
                {
                    "time": s["time"],
                    "direction": s.get("direction", "?"),
                    "duration_label": s.get("duration_label", ""),
                }
                for s in all_strokes
            ],
        }

        logger.info(f"扫弦质量评估: overall={overall:.2f}, timing={timing_label}, "
                    f"directions={down_count}下/{up_count}上, strokes={len(all_strokes)}, "
                    f"drift={tempo_drift:.3f}")
        return quality

    # ==================== Phase 3: 和弦以外的技巧检测 ====================

    def _detect_techniques_from_notes(self, notes: List[Dict],
                                        midi_data=None, sr: int = 16000) -> List[Dict]:
        """从 basic-pitch 音符数据检测技巧：击弦/勾弦/滑音/推弦/揉弦。

        原理：
        - 击弦(H): 新音符出现但无对应高频瞬态 → 非拨弦触发
        - 勾弦(P): 同击弦条件 + 音高下降
        - 滑音(slide): 频率连续变化（非跳变）
        - 推弦(bend): 频率偏离 12 平均律 > 40 音分且持续上升
        - 揉弦(vibrato): 频率在 ±20 音分内周期性波动

        Returns:
            [{start_time, end_time, technique_id, confidence, detail}, ...]
        """
        if not notes or len(notes) < 2:
            return []

        techniques = []
        # Sort notes by start time
        sorted_notes = sorted(notes, key=lambda n: n.get("start_time", 0))

        # 提取音符的 onset 强度特征（从 midi_data 音高轨迹分析）
        for i in range(1, len(sorted_notes)):
            prev = sorted_notes[i - 1]
            curr = sorted_notes[i]
            t_prev = prev.get("start_time", 0)
            t_curr = curr.get("start_time", 0)
            p_prev = prev.get("pitch", 60)
            p_curr = curr.get("pitch", 60)
            v_prev = prev.get("velocity", 0)
            v_curr = curr.get("velocity", 0)
            interval = t_curr - t_prev  # 音符间隔
            pitch_diff = p_curr - p_prev  # 音高差（半音）

            # 1. 击弦/勾弦检测：必须同弦 + 短间隔 + 力度骤降（确认非重新拨弦）
            # 仅凭音频无法区分「快速拨弦」和「击勾弦」，无视频确认前必须保守
            s_prev_str = prev.get("string", 0)
            s_curr_str = curr.get("string", 0)
            same_str = s_prev_str == s_curr_str and s_prev_str > 0
            if same_str and 0.02 < interval < 0.08 and v_curr < v_prev * 0.6:
                if pitch_diff > 0:
                    techniques.append({
                        "start_time": round(t_prev, 2),
                        "end_time": round(t_curr, 2),
                        "technique_id": "hammer-on",
                        "confidence": 0.65,
                        "detail": f"击弦: {prev.get('note_name','')}→{curr.get('note_name','')}",
                    })
                elif pitch_diff < 0:
                    techniques.append({
                        "start_time": round(t_prev, 2),
                        "end_time": round(t_curr, 2),
                        "technique_id": "pull-off",
                        "confidence": 0.65,
                        "detail": f"勾弦: {prev.get('note_name','')}→{curr.get('note_name','')}",
                    })

            # 2. 滑音检测：相邻音符时间连续 + 音高连续变化
            if interval < 0.15 and 1 <= abs(pitch_diff) <= 12:
                s_prev = prev.get("string", 0)
                s_curr = curr.get("string", 0)
                direction = "上滑" if pitch_diff > 0 else "下滑"
                detail = f"{direction}: {prev.get('note_name','')}→{curr.get('note_name','')}"
                if s_prev == s_curr and s_prev > 0:
                    # 同弦 → 高置信滑音
                    techniques.append({
                        "start_time": round(t_prev, 2),
                        "end_time": round(t_curr, 2),
                        "technique_id": "slide",
                        "confidence": 0.70,
                        "detail": detail,
                    })
                elif v_curr < 45:
                    # 力度低 + 短间隔 + 音高变化 → 大概率滑音，即使DP分配了不同弦
                    techniques.append({
                        "start_time": round(t_prev, 2),
                        "end_time": round(t_curr, 2),
                        "technique_id": "slide",
                        "confidence": 0.50,
                        "detail": detail,
                    })

        # 3. 推弦/揉弦检测（从 midi_data 音高轨迹）
        if midi_data is not None:
            try:
                import scipy.signal
                for note in sorted_notes:
                    t0 = note.get("start_time", 0)
                    t1 = note.get("end_time", 0)
                    dur = t1 - t0
                    if dur < 0.3:  # 太短的音符不够时间推弦/揉弦
                        continue
                    pitch = note.get("pitch", 60)
                    # 从 midi_data 提取该音符期间的音高轨迹
                    midi_offset = 21  # basic-pitch MIDI offset
                    bin_idx = pitch - midi_offset
                    if bin_idx < 0 or bin_idx >= midi_data.shape[1]:
                        continue
                    # 时间→帧
                    hop = 512
                    f0 = int(t0 * sr / hop)
                    f1 = int(t1 * sr / hop)
                    if f1 - f0 < 5:
                        continue
                    contour = np.array(midi_data[f0:f1, bin_idx], dtype=np.float64)
                    if len(contour) < 5:
                        continue

                    # 音高偏差（cents）
                    # basic-pitch contour 是激活值，需要检测峰值位置偏移
                    peak_shift = np.argmax(contour) - len(contour) // 2
                    if abs(peak_shift) > 1:
                        cents_dev = abs(peak_shift) * 50  # 粗略转换
                        if cents_dev > 40:
                            techniques.append({
                                "start_time": round(t0, 2),
                                "end_time": round(t1, 2),
                                "technique_id": "bend",
                                "confidence": min(0.85, 0.5 + cents_dev / 200),
                                "detail": f"推弦: {note.get('note_name','')} 偏离约{cents_dev:.0f}音分",
                            })
                        elif 10 < cents_dev <= 40:
                            # 揉弦：小幅度周期性波动
                            ac = np.correlate(contour - np.mean(contour),
                                              contour - np.mean(contour), mode='full')
                            ac = ac[len(ac)//2:]
                            ac = ac / (ac[0] + 1e-8)
                            if len(ac) > 3 and ac[2] > 0.3:
                                techniques.append({
                                    "start_time": round(t0, 2),
                                    "end_time": round(t1, 2),
                                    "technique_id": "vibrato",
                                    "confidence": 0.65,
                                    "detail": f"揉弦: {note.get('note_name','')}",
                                })
            except Exception:
                pass

        # 去重合并：相同技巧的相邻段合并
        if techniques:
            techniques.sort(key=lambda t: t["start_time"])
            merged = [techniques[0]]
            for t in techniques[1:]:
                last = merged[-1]
                if (t["technique_id"] == last["technique_id"]
                        and t["start_time"] - last["end_time"] < 0.3):
                    last["end_time"] = t["end_time"]
                    last["confidence"] = max(last["confidence"], t["confidence"])
                else:
                    merged.append(t)
            techniques = merged

        logger.info(f"技巧检测(Phase 3): {len(techniques)} 个 "
                     + ", ".join(f"{t['technique_id']}({t['confidence']:.2f})" for t in techniques[:5]))
        return techniques

    @staticmethod
    def _merge_segments(segments: List[Dict]) -> List[Dict]:
        """合并重叠的同类型技巧时间段"""
        if len(segments) < 2:
            return segments
        merged = []
        for seg in segments:
            if not merged:
                merged.append(seg.copy())
                continue
            last = merged[-1]
            if seg["technique_id"] == last["technique_id"] and seg["start_time"] <= last["end_time"] + 0.3:
                last["end_time"] = max(last["end_time"], seg["end_time"])
                last["confidence"] = max(last["confidence"], seg["confidence"])
            else:
                merged.append(seg.copy())
        return merged

    # ==================== basic-pitch 转录 ====================

    def _transcribe_notes(self, audio: np.ndarray, hand_zone_center: float = None, capo: int = 0) -> List[Dict]:
        """使用 basic-pitch 进行音符转录"""
        notes = []
        try:
            import basic_pitch
            from basic_pitch.inference import predict
            import tempfile

            # basic-pitch 需要文件路径，保存临时文件
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()

            import soundfile as sf
            sf.write(tmp_path, audio, 16000)

            model_output, midi_data, note_events = predict(tmp_path)
            # model_output is a dict {'note': array, 'onset': array, 'contour': array}
            # Store the note activation matrix for technique detection
            if isinstance(model_output, dict):
                self._last_model_output = model_output.get('note', None)
            else:
                self._last_model_output = model_output
            self._last_midi_data = midi_data

            for note in note_events:
                start_time, end_time, pitch, velocity, *rest = note
                dur = float(end_time - start_time)
                # basic-pitch 返回的 velocity 是归一化值 (0~1)，转为 0-127
                vel = float(velocity) * 127
                # 过滤噪音：严格的阈值减少 basic-pitch 伪影
                # dur<0.08s 过滤泛音/噪声碎片, vel<18 过滤微弱误检
                # basic-pitch 对泛音列敏感, 容易将泛音误识别为独立音符
                if dur < 0.08 or vel < 18:
                    continue
                midi = int(pitch)
                notes.append({
                    "start_time": float(round(start_time, 2)),
                    "end_time": float(round(end_time, 2)),
                    "pitch": midi,
                    "note_name": _midi_to_note_name(midi),
                    "velocity": vel,
                    "duration": float(round(dur, 2)),
                })

            # ── 八度泛音去重：同时刻(<50ms)且差12/19/24半音 → 只保留力度大的 ──
            if len(notes) >= 2:
                notes.sort(key=lambda n: (n["start_time"], -n["velocity"]))
                to_remove = set()
                for i in range(len(notes)):
                    if i in to_remove:
                        continue
                    for j in range(i + 1, len(notes)):
                        if j in to_remove:
                            continue
                        if notes[j]["start_time"] - notes[i]["start_time"] > 0.05:
                            break
                        if abs(notes[j]["pitch"] - notes[i]["pitch"]) in (12, 19, 24):
                            to_remove.add(j)
                if to_remove:
                    removed = len(to_remove)
                    total_before = len(notes)
                    notes = [n for idx, n in enumerate(notes) if idx not in to_remove]
                    logger.info(f"八度泛音去重: 移除 {removed} 个幻影音符 ({removed}/{total_before} = {removed/total_before*100:.0f}%)")

            # ── MIDI 音域过滤：吉他 E2(MIDI 40) ~ 高把位推弦上限(MIDI 79) ──
            before_range = len(notes)
            notes = [n for n in notes if 40 <= n.get("pitch", 0) <= 79]
            if len(notes) < before_range:
                logger.info(f"MIDI 音域过滤: 移除 {before_range - len(notes)} 个超出吉他音域的音符")

            # DP 批量分配最优弦+品位（直接使用用户指定的 capo，不做自动检测）
            batch_assign(notes, capo=capo, hand_zone_center=hand_zone_center)
            self._last_detected_capo = capo
            if capo > 0:
                logger.info(f"使用用户指定的变调夹: {capo}品")

            # 后置过滤：移除分配到不合理位置的音符（6弦>12品等泛音误检）
            orig_count = len(notes)
            notes[:] = [n for n in notes if not (
                n.get("string") == 6 and n.get("fret", 0) > 12
            )]
            if len(notes) < orig_count:
                removed = orig_count - len(notes)
                logger.info(f"过滤 {removed} 个不合理品位分配(如6弦>12品)")

            os.unlink(tmp_path)

        except Exception as e:
            logger.warning(f"basic-pitch 转录失败: {e}")
            logger.warning(f"basic-pitch traceback:\n{traceback.format_exc()}")

        return notes

    # ==================== Chroma 和弦识别 (librosa) ====================

    def _load_chord_chroma_templates(self):
        """从 KB 和弦文件加载 chroma 模板向量，用于和弦识别。

        Returns:
            dict: {chord_id: {"chroma": np.array(12,), "chord_name": str, "pcset": set, ...}}
        """
        templates = {}
        knowledge_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "knowledge", "chords"
        )
        if not os.path.isdir(knowledge_dir):
            return templates

        try:
            from chord_analyzer import ChordAnalyzer
        except ImportError:
            return templates

        for fname in sorted(os.listdir(knowledge_dir)):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(knowledge_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                fm = {}
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        fm = ChordAnalyzer._parse_frontmatter(parts[1])
                chord_id = fm.get("id", fname[:-3])
                if "chord-" in chord_id:
                    chord_id = chord_id.replace("chord-", "")
                strings = fm.get("strings", [])
                if strings and len(strings) == 6:
                    pcset = ChordAnalyzer._pcset_from_strings(strings, capo=0)
                    if len(pcset) == 0:
                        continue
                    chroma = np.zeros(12, dtype=np.float32)
                    for pc in pcset:
                        chroma[pc] = 1.0
                    templates[chord_id] = {
                        "chroma": chroma,
                        "chord_name": fm.get("chord_name", chord_id),
                        "pcset": pcset,
                        "strings": strings,
                        "difficulty": fm.get("difficulty", ""),
                        "related_problems": fm.get("related_problems", []),
                        "related_techniques": fm.get("related_techniques", []),
                    }
            except Exception:
                continue

        logger.info(f"加载 {len(templates)} 个和弦 chroma 模板")
        return templates

    # ==================== DeepChroma 辅助信号 ====================

    _DEEP_PROTOTYPES = None  # 类级缓存：{chord_id: np.array(128,)}

    @staticmethod
    def _synthesize_chord_audio(strings, sr=44100, duration=2.0):
        """从按弦定义合成吉他音色片段，用于 CNN 原型计算。"""
        open_freqs = [82.41, 110.0, 146.83, 196.0, 246.94, 329.63]  # 6弦→1弦
        t = np.arange(int(sr * duration), dtype=np.float64) / sr
        env = np.exp(-3.0 * t)
        signal = np.zeros(len(t), dtype=np.float64)
        for i, s in enumerate(strings):
            if isinstance(s, str) and s.lower() == 'x':
                continue
            try:
                fret = int(s)
            except (ValueError, TypeError):
                continue
            freq = open_freqs[i] * (2.0 ** (fret / 12.0))
            for h in range(1, 6):
                signal += (1.0 / h) * np.sin(2 * np.pi * freq * h * t)
        signal *= env
        mx = max(np.abs(signal).max(), 0.001)
        return (signal / mx).astype(np.float32)

    def _get_deep_prototypes(self):
        """懒加载 DeepChroma 和弦原型向量。每个和弦合成一次 CNN 特征。"""
        if AudioAnalyzer._DEEP_PROTOTYPES is not None:
            return AudioAnalyzer._DEEP_PROTOTYPES
        if not _DC_AVAILABLE:
            AudioAnalyzer._DEEP_PROTOTYPES = {}
            return {}

        templates = self._load_chord_chroma_templates()
        dcp = _CNNChordFeatureProcessor()
        prototypes = {}
        for cid, tpl in templates.items():
            strings = tpl.get("strings", [])
            if not strings or len(strings) != 6:
                continue
            try:
                synth = self._synthesize_chord_audio(strings)
                feats = dcp(synth)  # (n_frames, 128)
                proto = feats.mean(axis=0)  # (128,)
                proto = proto / (np.linalg.norm(proto) + 1e-8)
                prototypes[cid] = proto.astype(np.float32)
            except Exception as e:
                logger.warning(f"DeepChroma 原型构建失败 {cid}: {e}")

        AudioAnalyzer._DEEP_PROTOTYPES = prototypes
        logger.info(f"DeepChroma 原型已构建: {len(prototypes)}/{len(templates)} 个和弦")
        return prototypes

    def _compute_deep_chroma_scores(self, audio_f32: np.ndarray, sr: int,
                                     chord_ids: list, n_frames: int, hop_length: int):
        """用 DeepChroma CNN 为每帧计算 23 个和弦的余弦相似度分数。

        Returns:
            np.ndarray (n_chords × n_frames) 或 None（DeepChroma 不可用时）
        """
        if not _DC_AVAILABLE:
            return None
        prototypes = self._get_deep_prototypes()
        if len(prototypes) < 2:
            return None

        # 确保采样率匹配 madmom 要求（44100 Hz）
        if sr != 44100:
            import librosa as _librosa
            audio_44k = _librosa.resample(audio_f32, orig_sr=sr, target_sr=44100)
        else:
            audio_44k = audio_f32

        try:
            dcp = _CNNChordFeatureProcessor()
            feats = dcp(audio_44k)  # (n_madmom_frames, 128)
        except Exception as e:
            logger.warning(f"DeepChroma 特征提取失败: {e}")
            return None

        # L2 归一化 CNN 特征
        feats_norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)

        # 对每个和弦原型计算逐帧余弦相似度
        n_proto = len(chord_ids)
        n_dc_frames = feats_norm.shape[0]
        dc_scores = np.zeros((n_proto, n_dc_frames), dtype=np.float32)
        for i, cid in enumerate(chord_ids):
            proto = prototypes.get(cid)
            if proto is None:
                continue
            dc_scores[i, :] = np.dot(feats_norm, proto.astype(np.float32))

        # 插值到 chroma 帧率以对齐下游 frame_times
        if n_dc_frames > 1 and n_dc_frames != n_frames:
            dc_time = np.arange(n_dc_frames) * (len(audio_44k) / sr / n_dc_frames)
            chroma_time = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)
            interp = np.zeros((n_proto, n_frames), dtype=np.float32)
            for i in range(n_proto):
                interp[i, :] = np.interp(chroma_time, dc_time, dc_scores[i, :],
                                          left=dc_scores[i, 0], right=dc_scores[i, -1])
            dc_scores = interp

        return dc_scores

    # ==================== 主链路 Chroma 和弦识别 ====================

    def _detect_chords_chroma(self, audio: np.ndarray, sr: int = 16000,
                               capo: int = 0, notes: List[Dict] = None) -> List[Dict]:
        """使用 librosa chroma + 和弦模板匹配检测和弦（capo 感知版）。

        按指定 capo 将 chromagram 移位后匹配模板。不再自动枚举 capo。
        """
        if not LIBROSA_AVAILABLE:
            logger.info("librosa 不可用，跳过 chroma 和弦识别")
            return []

        if len(audio) < sr * 0.5:
            return []

        templates = self._load_chord_chroma_templates()
        if len(templates) < 2:
            return []

        try:
            audio_f32 = audio.astype(np.float32)

            # 1. CQT chromagram
            hop_length = 512
            chroma_raw = librosa.feature.chroma_cqt(
                y=audio_f32, sr=sr, hop_length=hop_length,
                bins_per_octave=36, n_chroma=12,
            )
            n_frames = chroma_raw.shape[1]

            # 2. 中值滤波去噪
            from scipy.ndimage import median_filter
            chroma_smooth = median_filter(chroma_raw, size=(1, 3))

            # 3. 按指定 capo 移位 chroma
            best_capo = capo
            chord_ids = list(templates.keys())
            n_chords = len(chord_ids)
            if best_capo > 0:
                chroma_final = np.roll(chroma_smooth, -best_capo, axis=0)
            else:
                chroma_final = chroma_smooth

            scores = np.zeros((n_chords, n_frames), dtype=np.float32)
            _C_MAJOR = {0, 2, 4, 5, 7, 9, 11}
            _diatonic_bonus = np.array([
                0.025 if templates[cid].get("pcset", set()) and templates[cid].get("pcset", set()).issubset(_C_MAJOR) else 0.0
                for cid in chord_ids
            ], dtype=np.float32)
            # 根音 rank 惩罚：模板的根音在帧内能量排名太低 → 该和弦不匹配
            _ROOT_MAP = {"C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,
                         "E":4,"F":5,"F#":6,"Gb":6,"G":7,"G#":8,
                         "Ab":8,"A":9,"A#":10,"Bb":10,"B":11}
            def _extract_root_pc(cid: str) -> int:
                name = cid.replace("chord-", "")
                if len(name) >= 2 and name[1] in ('#', 'b'):
                    return _ROOT_MAP.get(name[:2], 0)
                return _ROOT_MAP.get(name[0], 0)
            _root_pcs = [_extract_root_pc(cid) for cid in chord_ids]
            for i, cid in enumerate(chord_ids):
                tpl = templates[cid]["chroma"].astype(np.float32)
                t_norm = np.linalg.norm(tpl)
                if t_norm < 0.01:
                    continue
                bonus = _diatonic_bonus[i]
                root_pc = _root_pcs[i]
                _pcset = templates[cid].get("pcset", set())
                for f in range(n_frames):
                    frame = chroma_final[:, f]
                    f_norm = np.linalg.norm(frame)
                    if f_norm < 0.01:
                        scores[i, f] = 0.0
                    else:
                        cos_sim = float(np.dot(frame, tpl) / (f_norm * t_norm))
                        # 根音权重：根音占总能量比
                        root_val = float(frame[root_pc])
                        frame_sum = float(np.sum(frame))
                        root_ratio = root_val / max(frame_sum, 0.001)
                        if root_ratio < 0.08:
                            root_penalty = 0.15
                        elif root_ratio < 0.12:
                            root_penalty = 0.08
                        elif root_ratio < 0.16:
                            root_penalty = 0.03
                        else:
                            root_penalty = 0.0
                        root_bonus = 0.03 if root_ratio > 0.22 else 0.0
                        # 多余音符：帧内不在模板 pcset 中的能量占比 → 惩罚
                        # 对 Am↔F (共享A/C，仅差E/F) 等共享2/3音的和弦区分至关重要
                        extra_e = frame_sum - sum(float(frame[p]) for p in _pcset if p < 12)
                        extra_ratio = extra_e / max(frame_sum, 0.001)
                        if extra_ratio > 0.30:
                            extra_penalty = 0.16
                        elif extra_ratio > 0.22:
                            extra_penalty = 0.08
                        elif extra_ratio > 0.15:
                            extra_penalty = 0.04
                        else:
                            extra_penalty = 0.0
                        scores[i, f] = cos_sim + bonus + root_bonus - root_penalty - extra_penalty
                        # 复杂度惩罚：音高数>3的模板更容易"解释"泛音泄露，需轻微惩罚
                        if len(_pcset) > 3:
                            scores[i, f] -= 0.04 * (len(_pcset) - 3)

            best_indices = np.argmax(scores, axis=0).astype(np.int32)
            best_scores = np.max(scores, axis=0)
            best_indices_smooth = median_filter(
                best_indices.astype(np.float64), size=7
            ).astype(np.int32)

            frame_times = librosa.frames_to_time(
                np.arange(n_frames), sr=sr, hop_length=hop_length
            )

            # First pass: collect all raw segments (including short ones)
            min_frames = int(0.85 * sr / hop_length)
            raw_segments = []
            i = 0
            while i < n_frames:
                cid_idx = int(best_indices_smooth[i])
                j = i + 1
                while j < n_frames and int(best_indices_smooth[j]) == cid_idx:
                    j += 1
                if cid_idx < n_chords:
                    seg_chroma = np.mean(chroma_final[:, i:j], axis=1)
                    raw_segments.append({
                        "i": i, "j": j,
                        "cid_idx": cid_idx,
                        "seg_len": j - i,
                        "seg_chroma": seg_chroma,
                    })
                i = j

            # Merge adjacent segments sharing ≥2 pitch classes,
            # but ONLY when at least one segment < min_frames (prevent chain-merging long segments)
            changed = True
            while changed:
                changed = False
                for idx in range(len(raw_segments) - 1):
                    seg_a = raw_segments[idx]
                    seg_b = raw_segments[idx + 1]
                    if seg_a["seg_len"] >= min_frames and seg_b["seg_len"] >= min_frames:
                        continue
                    cid_a = chord_ids[seg_a["cid_idx"]]
                    cid_b = chord_ids[seg_b["cid_idx"]]
                    pcset_a = templates[cid_a].get("pcset", set())
                    pcset_b = templates[cid_b].get("pcset", set())
                    if len(pcset_a & pcset_b) < 2:
                        continue
                    # 和弦性质不同不合并：大三↔小三是最容易区分的特征
                    # 例: C(major)↔Am(minor), Am(minor)↔F(major) 共享2音但功能不同
                    def _chord_quality(cid: str) -> str:
                        """从 chord_id 推导和弦性质: major/minor/dim/aug"""
                        name = cid.lower().replace("chord-", "")
                        # 先检查复杂后缀
                        if "dim" in name or "°" in name:
                            return "dim"
                        if "aug" in name or "+" in name:
                            return "aug"
                        if "maj" in name:
                            return "major"  # maj7, maj9 etc
                        # "m" 但不是 "maj" → minor
                        if "m" in name:
                            return "minor"
                        # 纯数字后缀如 "7", "9" → dominant (major family)
                        if any(c.isdigit() for c in name):
                            return "major"
                        return "major"
                    if _chord_quality(cid_a) != _chord_quality(cid_b):
                        continue
                    seg_a["j"] = seg_b["j"]
                    seg_a["seg_len"] = seg_a["j"] - seg_a["i"]
                    seg_a["seg_chroma"] = np.mean(chroma_final[:, seg_a["i"]:seg_a["j"]], axis=1)
                    raw_segments.pop(idx + 1)
                    changed = True
                    break

            # Second pass: process (possibly merged) segments
            detected = []
            for seg in raw_segments:
                i = seg["i"]
                j = seg["j"]
                cid_idx = seg["cid_idx"]
                seg_chroma = seg["seg_chroma"]
                if seg["seg_len"] >= min_frames:
                    seg_scores = best_scores[i:j]
                    avg_score = float(np.mean(seg_scores))

                    # 逐帧最优可能 pcset 不匹配（如 B7 当选但 D# 能量太低）
                    # → 尝试 runner-up，直到找到 pcset 匹配的或全失败
                    # 按帧平均分排序候选（去重）
                    seg_mean_scores = np.mean(scores[:, i:j], axis=1)
                    # 降序遍历（不用 [::-1] 避免翻转稳定排序导致同分时顺序颠倒）
                    candidate_order = np.argsort(seg_mean_scores)
                    best_candidate = None
                    best_candidate_score = -999.0
                    tried = set()
                    for pos in range(len(candidate_order) - 1, -1, -1):
                        cand_idx = candidate_order[pos]
                        cand_cid = chord_ids[cand_idx]
                        if cand_cid in tried:
                            continue
                        tried.add(cand_cid)
                        cand_info = templates[cand_cid]
                        cand_pcset = cand_info.get("pcset", set())
                        if cand_pcset:
                            match_count = sum(1 for pc in cand_pcset if seg_chroma[pc] > 0.10)
                            if match_count < 2:
                                continue
                        cand_score = float(seg_mean_scores[cand_idx])
                        # 简约性：音高数多的和弦（七和弦）需要显著更高的模板得分才能胜出
                        cand_complex = len(cand_pcset) if cand_pcset else 3
                        _cand_sc = cand_score
                        _best_sc = best_candidate_score if best_candidate is not None else -999.0
                        if best_candidate is not None:
                            _best_pcset_pre = best_candidate[1].get("pcset", set())
                            best_complex = len(_best_pcset_pre) if _best_pcset_pre else 3
                            if cand_complex > best_complex:
                                _cand_sc -= 0.060 * (cand_complex - best_complex)
                            elif best_complex > cand_complex:
                                _best_sc -= 0.060 * (best_complex - cand_complex)
                        # 同 pcset 和弦：优先选简单三和弦而非斜杠/转位和弦
                        is_slash = "/" in cand_info.get("chord_name", "")
                        if best_candidate is None:
                            best_candidate = (cand_cid, cand_info, cand_score)
                            best_candidate_score = cand_score
                        elif is_slash and _cand_sc <= _best_sc + 0.03:
                            continue  # 斜杠和弦不显著优于已选非斜杠和弦
                        elif not is_slash and _best_sc <= _cand_sc + 0.03:
                            # 非斜杠候选不显著差于已选斜杠 → 替换
                            prev_is_slash = "/" in best_candidate[1].get("chord_name", "")
                            if prev_is_slash or _cand_sc > _best_sc:
                                best_candidate = (cand_cid, cand_info, cand_score)
                                best_candidate_score = cand_score
                        elif _cand_sc > _best_sc + 0.03:
                            best_candidate = (cand_cid, cand_info, cand_score)
                            best_candidate_score = cand_score
                        elif cand_score >= best_candidate_score - 0.06:
                            # 子集↑：用原始分（不受简约性偏见影响）判断三和弦→七和弦升级
                            _best_pcset_last = best_candidate[1].get("pcset", set())
                            if _best_pcset_last and cand_pcset:
                                if _best_pcset_last.issubset(cand_pcset) and _best_pcset_last != cand_pcset:
                                    _shared = _best_pcset_last & cand_pcset
                                    _extra = cand_pcset - _best_pcset_last
                                    _shared_e = sum(float(seg_chroma[pc]) for pc in _shared)
                                    _avg_shared = _shared_e / len(_shared) if _shared else 0
                                    _extra_e = sum(float(seg_chroma[pc]) for pc in _extra)
                                    _avg_extra = _extra_e / len(_extra) if _extra else 0
                                    if _avg_shared > 0.01 and _avg_extra >= _avg_shared * 0.80:
                                        logger.info(f"Chord upgrade(subset↑): "
                                                    f"{best_candidate[1].get('chord_name','?')}→"
                                                    f"{cand_info.get('chord_name', cand_cid)} "
                                                    f"@t={float(frame_times[i]):.1f}s")
                                        best_candidate = (cand_cid, cand_info, cand_score)
                                        best_candidate_score = cand_score
                                        continue

                    if best_candidate is None:
                        i = j
                        continue
                    picked_cid, picked_info, picked_score = best_candidate
                    cid = picked_cid
                    info = picked_info
                    avg_score = picked_score
                    # Guard: 不从超集七和弦降级到其三和弦子集
                    # 仅当区分音是超集和弦根音时才触发——这是最可靠的信号。
                    # 例: Bm7 vs D，区分音 B 是 Bm7 根音 → 保护 Bm7
                    # 反例: Cmaj7 vs C，区分音 B 只是七音 → 不保护 (可能是泛音噪声)
                    if cid != chord_ids[cid_idx]:
                        orig_pcset = templates[chord_ids[cid_idx]].get("pcset", set())
                        picked_pcset = picked_info.get("pcset", set())
                        if (picked_pcset and orig_pcset
                                and picked_pcset.issubset(orig_pcset)
                                and picked_pcset != orig_pcset):
                            extra_pcs = orig_pcset - picked_pcset
                            # 从 chord_id 推导根音
                            _ROOT_MAP = {"C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,
                                         "E":4,"F":5,"F#":6,"Gb":6,"G":7,"G#":8,
                                         "Ab":8,"A":9,"A#":10,"Bb":10,"B":11}
                            _orig_cid = chord_ids[cid_idx].replace("chord-","")
                            _orig_root = None
                            for _rn in sorted(_ROOT_MAP.keys(), key=len, reverse=True):
                                if _orig_cid.startswith(_rn):
                                    _orig_root = _ROOT_MAP[_rn]
                                    break
                            is_root_extra = _orig_root is not None and _orig_root in extra_pcs
                            if is_root_extra:
                                # 超集的段平均 chroma 分不能显著低于子集
                                # 否则段级 chroma 明确倾向子集，不应 guard
                                orig_seg_score = float(seg_mean_scores[cid_idx])
                                if orig_seg_score < picked_score - 0.06:
                                    is_root_extra = False
                                # 段内多样性高说明是噪声链式合并，不应 guard
                                # (例: Am7 段含 Am7/A7/Cmaj7/G7/Em/E → 6 种 →
                                #  跳过 guard)
                                if is_root_extra:
                                    # 用未平滑 raw 索引检测链式噪声合并
                                    unique_ids = len(set(
                                        int(best_indices[fi])
                                        for fi in range(i, j)
                                    ))
                                    if unique_ids > 5:
                                        is_root_extra = False
                            if is_root_extra:
                                extra_energy = sum(float(seg_chroma[pc]) for pc in extra_pcs)
                                avg_extra = extra_energy / len(extra_pcs) if extra_pcs else 0
                                shared_pcs = orig_pcset & picked_pcset
                                shared_energy = sum(float(seg_chroma[pc]) for pc in shared_pcs)
                                avg_shared = shared_energy / len(shared_pcs) if shared_pcs else 0.001
                                t0 = float(frame_times[i])
                                keep_superset = (avg_extra > 0.08 or
                                                 (avg_shared > 0.01 and avg_extra >= avg_shared * 0.40))
                                if keep_superset:
                                    cid = chord_ids[cid_idx]
                                    info = templates[chord_ids[cid_idx]]
                                    avg_score = float(seg_mean_scores[cid_idx])
                                    logger.info(f"Chord pcset guard(root): {picked_cid}→{cid} "
                                                f"@t={t0:.1f}s "
                                                f"(extra/avg_shared={avg_extra:.3f}/{avg_shared:.3f})")
                    if cid != chord_ids[cid_idx]:
                        logger.info(f"Chord pcset fallback: {chord_ids[cid_idx]}→{cid} "
                                    f"@t={float(frame_times[i]):.1f}s (pcset mismatch of top choice)")

                    if avg_score >= 0.50:
                        t0 = round(float(frame_times[i]), 2)
                        t1 = round(float(frame_times[min(j - 1, n_frames - 1)]), 2)
                        if detected and detected[-1]["chord_id"] == cid \
                                and t0 - detected[-1].get("end_time", 0) < 3.0:
                            detected[-1]["end_time"] = t1
                            detected[-1]["confidence"] = round(
                                max(detected[-1]["confidence"], avg_score), 2
                            )
                        else:
                            entry = {
                                "time": t0,
                                "end_time": t1,
                                "chord_id": cid,
                                "chord_name": info["chord_name"],
                                "confidence": round(min(0.95, avg_score), 2),
                                "related_problems": info.get("related_problems", []),
                                "related_techniques": info.get("related_techniques", []),
                                "_source": "audio",
                            }
                            if cid != chord_ids[cid_idx]:
                                entry["_pcset_fallback"] = True
                                entry["_original_template_cid"] = chord_ids[cid_idx]
                            detected.append(entry)
            # 4. 相对大小调消歧：Em↔G, Am↔C 等共享 2/3 音高，cosine 无法区分
            detected = self._disambiguate_relative_major_minor(
                detected, scores, chord_ids, chroma_final, frame_times, templates, sr, hop_length
            )

            # 4.3 两步七和弦验证：用已确定的三和弦为锚点，独立验证7音证据
            detected = self._verify_seventh_chords(
                detected, chroma_final, frame_times, templates, sr, hop_length
            )

            # 4.5 RMS + 频谱色彩过滤：移除静音区噪底匹配 + 停顿/噪音段
            if detected and len(audio) > 0:
                overall_rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
                if overall_rms > 1e-6:
                    min_rms = overall_rms * 0.08
                    filtered = []
                    prev_sp = None
                    for d in detected:
                        t0 = d.get("time", 0)
                        t1 = d.get("end_time", t0 + 1)
                        s0 = max(0, int(t0 * sr))
                        s1 = min(len(audio), int(t1 * sr))
                        if s1 > s0 + sr // 10:
                            seg_rms = float(np.sqrt(np.mean(audio[s0:s1].astype(np.float64) ** 2)))
                            if seg_rms < min_rms:
                                logger.info(f"RMS filter drop: {d['chord_name']} [{t0:.1f}s-{t1:.1f}s] "
                                            f"seg_rms={seg_rms:.4f} < {min_rms:.4f}")
                                continue
                            # 残响检测：频谱高度相似+能量显著衰减→前和弦的残响
                            # 只用极严阈值(rms衰减>30%, 质心差<150Hz)避免误杀
                            dur = t1 - t0
                            if prev_sp and dur < 2.0:
                                sp = self._segment_spectral_features(audio[s0:s1], sr)
                                if sp and prev_sp:
                                    c_sim = abs(sp["centroid_hz"] - prev_sp["centroid_hz"]) < 100
                                    rms_decay = sp["rms"] < prev_sp["rms"] * 0.70
                                    if c_sim and rms_decay:
                                        logger.info(f"Spectral drop (resonance): {d['chord_name']} "
                                                    f"[{t0:.1f}s-{t1:.1f}s] centroid={sp['centroid_hz']:.0f}/{prev_sp['centroid_hz']:.0f}Hz "
                                                    f"rms={sp['rms']:.4f}/{prev_sp['rms']:.4f}")
                                        continue
                            else:
                                sp = None
                            filtered.append(d)
                            prev_sp = self._segment_spectral_features(audio[s0:s1], sr) if sp is None else sp
                        else:
                            filtered.append(d)
                    if len(filtered) < len(detected):
                        logger.info(f"RMS filter: {len(detected)}→{len(filtered)} segments")
                    detected = filtered

            # 4.6 过渡区假阳合并：短+低置信度+与邻居共享≥2音且另一边≤1音 → 合并
            if len(detected) >= 3:
                merged = []
                i = 0
                while i < len(detected):
                    d = detected[i]
                    dur = d.get("end_time", 0) - d.get("time", 0)
                    conf = d.get("confidence", 0)
                    cid = d.get("chord_id", "")
                    cur_pcset = templates.get(cid, {}).get("pcset", set())
                    _cond = (0 < i < len(detected) - 1 and dur < 1.5 and conf < 0.80 and bool(cur_pcset))
                    if _cond:
                        prev_pcset = templates.get(detected[i-1].get("chord_id", ""), {}).get("pcset", set())
                        next_pcset = templates.get(detected[i+1].get("chord_id", ""), {}).get("pcset", set())
                        sp = len(cur_pcset & prev_pcset) if prev_pcset else 0
                        sn = len(cur_pcset & next_pcset) if next_pcset else 0
                        # 和弦性质不同不合并（如 Am/minor ↔ Fmaj7/major）
                        _prev_qual = _chord_quality(detected[i-1].get("chord_id", ""))
                        _cur_qual = _chord_quality(cid)
                        _qual_match = _prev_qual == _cur_qual
                        if sp >= 3 and sn <= 1 and _qual_match:
                            # If current is superset of prev, keep current's identity
                            if prev_pcset and prev_pcset.issubset(cur_pcset) and prev_pcset != cur_pcset:
                                detected[i-1]["end_time"] = d["end_time"]
                                detected[i-1]["chord_id"] = d["chord_id"]
                                detected[i-1]["chord_name"] = d["chord_name"]
                                detected[i-1]["confidence"] = round(max(detected[i-1]["confidence"], conf), 2)
                            else:
                                merged[-1]["end_time"] = d["end_time"]
                                merged[-1]["confidence"] = round(max(merged[-1]["confidence"], conf), 2)
                            logger.info(f"Transition merge: {d['chord_name']}[{d['time']:.1f}s-{d['end_time']:.1f}s] "
                                        f"→ {merged[-1]['chord_name']} (shared {sp}/{sn} with prev/next)")
                            i += 1
                            continue
                        elif sn >= 3 and sp <= 1 and _qual_match:
                            # If current is superset of next, merge next into current
                            if next_pcset and next_pcset.issubset(cur_pcset) and next_pcset != cur_pcset:
                                detected[i+1]["time"] = d["time"]
                                detected[i+1]["chord_id"] = d["chord_id"]
                                detected[i+1]["chord_name"] = d["chord_name"]
                            else:
                                detected[i+1]["time"] = d["time"]
                            logger.info(f"Transition merge: {d['chord_name']}[{d['time']:.1f}s-{d['end_time']:.1f}s] "
                                        f"→ {detected[i+1]['chord_name']} (shared {sp}/{sn} with prev/next)")
                            i += 1
                            continue
                    merged.append(d)
                    i += 1
                if len(merged) < len(detected):
                    logger.info(f"Transition merge: {len(detected)}→{len(merged)} segments")
                detected = merged

            # 4.7 最小持续时间过滤：短于 0.5s 的段物理上不可能形成独立和弦
            MIN_CHORD_DUR = 0.5
            if len(detected) >= 2:
                filtered = []
                for i, d in enumerate(detected):
                    dur = d.get("end_time", 0) - d.get("time", 0)
                    if dur >= MIN_CHORD_DUR:
                        filtered.append(d)
                        continue
                    # 与邻居合并：选 pcset 重叠多的
                    cid = d.get("chord_id", "")
                    cur_pc = templates.get(cid, {}).get("pcset", set())
                    prev_pc = templates.get(detected[i-1].get("chord_id", ""), {}).get("pcset", set()) if i > 0 else set()
                    next_pc = templates.get(detected[i+1].get("chord_id", ""), {}).get("pcset", set()) if i < len(detected)-1 else set()
                    sp = len(cur_pc & prev_pc) if prev_pc else 0
                    sn = len(cur_pc & next_pc) if next_pc else 0
                    if sp >= sn and i > 0 and filtered:
                        filtered[-1]["end_time"] = d["end_time"]
                        filtered[-1]["confidence"] = round(max(filtered[-1]["confidence"], d.get("confidence", 0)), 2)
                    elif sn > 0 and i < len(detected) - 1:
                        detected[i+1]["time"] = d["time"]
                    else:
                        filtered.append(d)  # 无法合并，保留
                if len(filtered) < len(detected):
                    logger.info(f"Min-duration filter: {len(detected)}→{len(filtered)} segments")
                detected = filtered

            logger.info(f"Chroma 和弦识别 (capo={best_capo}): {len(detected)} 个和弦 "
                        + ", ".join(f"{d['chord_name']}({d['confidence']:.2f})" for d in detected[:6]))
            # 存到实例，供 analyze() 读取更新 capo 和 refine_chords_with_notes 使用 chroma 评分
            self._chroma_detected_capo = best_capo
            self._chroma_scores = scores
            self._chroma_frame_times = frame_times
            self._chroma_chord_ids = chord_ids
            # DeepChroma 辅助信号（不影响主链路）
            self._deep_chroma_scores = self._compute_deep_chroma_scores(
                audio_f32, sr, chord_ids, n_frames, hop_length
            )
            return detected

        except Exception as e:
            logger.warning(f"Chroma 和弦识别失败: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            return []

    def _segment_spectral_features(self, audio_segment: np.ndarray, sr: int = 16000):
        """计算音频段的频谱色彩特征（不依赖转录，直接从波形计算）。

        Returns dict with:
            centroid_hz: 频谱质心（亮度），开放和弦通常更亮
            bandwidth_hz: 频谱带宽，七和弦>三和弦（多一个音=频谱更宽）
            flatness: 频谱平坦度，0=纯音 1=白噪声，停顿/噪音>0.5
            rolloff_hz: 85%能量所在频率
            hf_ratio: >2kHz能量占比，七和弦不协和度更高→高频更多
            rms: 均方根能量
        """
        if len(audio_segment) < sr // 10:
            return None
        try:
            S = np.abs(librosa.stft(audio_segment.astype(np.float32), n_fft=1024, hop_length=256))
            centroid = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
            bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr)))
            flatness = float(np.mean(librosa.feature.spectral_flatness(S=S)))
            rolloff = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)))
            # high-frequency ratio: energy above 2kHz / total
            freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
            hf_mask = freqs >= 2000
            total_power = float(np.sum(S**2))
            hf_power = float(np.sum(S[hf_mask, :]**2)) if np.any(hf_mask) else 0.0
            hf_ratio = hf_power / total_power if total_power > 1e-10 else 0.0
            rms = float(np.sqrt(np.mean(audio_segment.astype(np.float64)**2)))
            return dict(centroid_hz=centroid, bandwidth_hz=bandwidth,
                        flatness=flatness, rolloff_hz=rolloff,
                        hf_ratio=hf_ratio, rms=rms)
        except Exception:
            return None

    def _resolve_chord_intent(self, detected_chords: List[Dict], notes: List[Dict],
                              capo: int = 0) -> List[Dict]:
        """用多源证据纠正 chroma 纯声学匹配的明显错误。

        当 chroma 被缺失音/泛音等误导时（如 Bm7 16帧全被识别为 D），
        用音符证据 + 进行上下文来恢复演奏者的真实意图。

        Returns:
            修正后的 detected_chords（可能修改 chord_id/chord_name）。
        """
        if not detected_chords or not notes:
            return detected_chords
        if not hasattr(self, '_chroma_scores') or self._chroma_scores is None:
            return detected_chords

        scores = self._chroma_scores
        frame_times = self._chroma_frame_times
        chord_ids = list(self._chroma_chord_ids)
        templates = self._load_chord_chroma_templates()

        _ROOTS = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
                  'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
                  'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}

        def _get_root(cid: str):
            if not cid:
                return None
            if len(cid) >= 2 and cid[1] in ('#', 'b'):
                return _ROOTS.get(cid[:2])
            return _ROOTS.get(cid[0])

        def _note_evidence(cid: str, t0: float, t1: float) -> float:
            """音符证据：候选和弦的音高类覆盖率 × (1 - 多余音符惩罚)。

            多余音符 = 转录音符中存在但不在候选和弦 pcset 内的音。
            例如，转录有 B D F# A 时，D {D,F#,A} 因缺少 B 受罚。
            """
            tpl = templates.get(cid, {})
            pcset = tpl.get("pcset", set())
            if not pcset:
                return 0.0
            effective_pcset = {((pc + capo) % 12) for pc in pcset} if capo else pcset
            margin = 0.15
            note_pcs = {}
            for n in notes:
                nt = n.get("start_time", 0)
                if t0 - margin <= nt <= t1 + margin:
                    pc = n["pitch"] % 12
                    vel = n.get("velocity", 0)
                    note_pcs[pc] = max(note_pcs.get(pc, 0), vel / 127.0)
            if not note_pcs:
                return 0.0
            total_e = sum(note_pcs.values())
            if total_e < 0.001:
                return 0.0
            chord_e = sum(note_pcs.get(pc, 0) for pc in effective_pcset)
            matched = sum(1 for pc in effective_pcset if pc in note_pcs)
            coverage = matched / len(effective_pcset)
            # 多余音符惩罚：转录中不在候选和弦内的音符能量占比
            extra_e = sum(e for pc, e in note_pcs.items() if pc not in effective_pcset)
            penalty = extra_e / total_e
            return coverage * (1.0 - 0.5 * penalty)

        def _progression_bonus(prev_cid: str, cand_cid: str) -> float:
            """和弦进行常见度加分。"""
            if not prev_cid:
                return 0.0
            pr = _get_root(prev_cid)
            cr = _get_root(cand_cid)
            if pr is None or cr is None:
                return 0.0
            interval = (cr - pr) % 12
            if interval in (5, 7):       # P4/P5 — 极常见
                return 0.10
            if interval in (2, 3, 4, 8, 9, 10):  # 2nd/3rd — 常见
                return 0.05
            if interval == 0:            # 同根音
                return 0.03
            return 0.0

        resolved = []
        prev_cid = None

        # DeepChroma 辅助分数（仅在 chroma 弱信号时介入）
        dc_scores = getattr(self, '_deep_chroma_scores', None)
        has_dc = dc_scores is not None and dc_scores.shape[0] == len(chord_ids)

        def _deep_seg(cand_idx: int) -> float:
            """候选和弦在 [fi:fj] 区间的 DeepChroma 平均分数（0-1）。"""
            if not has_dc or fi >= fj:
                return 0.0
            return float(np.clip(np.mean(dc_scores[cand_idx, fi:fj]), 0, 1))

        for chord in detected_chords:
            t0 = chord.get("time", 0)
            t1 = chord.get("end_time", t0 + 1.0)
            cur_cid = chord.get("chord_id", "")

            fi = int(np.searchsorted(frame_times, t0))
            fj = int(np.searchsorted(frame_times, t1)) + 1
            fi = max(0, min(fi, len(frame_times) - 1))
            fj = max(fi + 1, min(fj, len(frame_times)))

            seg_mean = np.mean(scores[:, fi:fj], axis=1)
            top_k = min(3, len(chord_ids))
            top_indices = np.argsort(seg_mean)[-top_k:][::-1]

            top1_cid = chord_ids[top_indices[0]]
            top1_chroma = float(seg_mean[top_indices[0]])

            # 动态权重：chroma 信号弱时 DeepChroma 介入，强时信任主链路
            def _weights(chroma_sc: float):
                if has_dc and chroma_sc < 0.45:
                    return 0.35, 0.25  # w_chroma, w_dc — DeepChroma 升权
                elif has_dc and chroma_sc < 0.65:
                    return 0.50, 0.10  # 混合
                return 0.60, 0.00      # 信任主链路

            best_cid = top1_cid
            best_total = -999.0

            for cand_idx in top_indices:
                cand_cid = chord_ids[cand_idx]
                chroma_sc = float(seg_mean[cand_idx])
                note_ev = _note_evidence(cand_cid, t0, t1)
                prog_b = _progression_bonus(prev_cid, cand_cid)
                w_chroma, w_dc = _weights(chroma_sc)
                dc_sc = _deep_seg(cand_idx) if w_dc > 0 else 0.0
                total = w_chroma * chroma_sc + w_dc * dc_sc + 0.25 * note_ev + 0.15 * prog_b
                if total > best_total:
                    best_total = total
                    best_cid = cand_cid

            # 仅在多源赢家显著优于 chroma top-1 时覆盖（总分差距 > 0.14）
            top1_total = -999.0
            for cand_idx in top_indices:
                if chord_ids[cand_idx] == top1_cid:
                    chroma_sc = float(seg_mean[cand_idx])
                    note_ev = _note_evidence(top1_cid, t0, t1)
                    prog_b = _progression_bonus(prev_cid, top1_cid)
                    w_chroma, w_dc = _weights(chroma_sc)
                    dc_sc = _deep_seg(cand_idx) if w_dc > 0 else 0.0
                    top1_total = w_chroma * chroma_sc + w_dc * dc_sc + 0.25 * note_ev + 0.15 * prog_b
                    break

            if best_cid and best_cid != cur_cid and (best_total - top1_total) > 0.14:
                old_name = chord.get("chord_name", cur_cid)
                new_name = templates.get(best_cid, {}).get("chord_name", best_cid)
                logger.info(f"Chord intent: {old_name}({cur_cid})→{new_name}({best_cid}) "
                            f"@t={t0:.1f}s (margin={best_total - top1_total:.3f})")
                chord = dict(chord)
                chord["chord_id"] = best_cid
                chord["chord_name"] = new_name
                chord["confidence"] = round(max(chord.get("confidence", 0), 0.55), 2)
                chord["_intent_corrected"] = True
            elif best_cid != cur_cid and (best_total - top1_total) > 0.08:
                # DEBUG: log why correction didn't fire
                logger.info(f"Chord intent NO-OP: {cur_cid}→{best_cid} "
                            f"@t={t0:.1f}s margin={best_total - top1_total:.3f} "
                            f"top1={top1_cid}({top1_total:.3f}) best={best_cid}({best_total:.3f})")

            resolved.append(chord)
            prev_cid = chord.get("chord_id", "")

        return resolved

    def _disambiguate_relative_major_minor(self, detected, scores, chord_ids,
                                             chroma, frame_times, templates,
                                             sr, hop_length):
        """消歧相对大小调（如 Am↔C, Em↔G），比较唯一音高类的 chroma 能量。

        相对大小调共享三和弦中的两个音，cosine 相似度无法可靠区分。
        通过比较各和弦独有音高类的 chroma 能量来判定。
        """
        if not detected:
            return detected

        # (major, minor, major_unique_pc, minor_unique_pc)
        RELATIVE_PAIRS = [
            ("C", "Am", 7, 9),    # C 5th=G(7), Am root=A(9)
            ("G", "Em", 2, 4),    # G 5th=D(2), Em root=E(4)
            ("D", "Bm", 9, 11),   # D 5th=A(9), Bm root=B(11)
            ("A", "F#m", 4, 6),   # A 5th=E(4), F#m root=F#(6)
            ("E", "C#m", 11, 1),  # E 5th=B(11), C#m root=C#(1)
            ("F", "Dm", 0, 2),    # F 5th=C(0), Dm root=D(2)
        ]
        pair_map = {}
        for maj, min_c, maj_uq, min_uq in RELATIVE_PAIRS:
            pair_map[maj] = (min_c, maj_uq, min_uq)
            pair_map[min_c] = (maj, min_uq, maj_uq)

        n_frames = len(frame_times)
        fixed = []
        for d in detected:
            cid = d.get("chord_id", "")
            if cid not in pair_map:
                fixed.append(d)
                continue
            alt_cid, my_uq_pc, alt_uq_pc = pair_map[cid]
            if alt_cid not in chord_ids:
                fixed.append(d)
                continue

            t0 = d.get("time", 0)
            t1 = d.get("end_time", t0 + 2.0)
            f0 = max(0, int(t0 * sr / hop_length))
            f1 = min(n_frames, int(t1 * sr / hop_length) + 1)
            if f1 <= f0:
                fixed.append(d)
                continue

            my_idx = chord_ids.index(cid)
            alt_idx = chord_ids.index(alt_cid)
            my_score = float(np.mean(scores[my_idx, f0:f1]))
            alt_score = float(np.mean(scores[alt_idx, f0:f1]))

            # 只处理接近的情况（margin < 0.06）
            if alt_score <= my_score - 0.06:
                fixed.append(d)
                continue

            my_energy = float(np.mean(chroma[my_uq_pc, f0:f1]))
            alt_energy = float(np.mean(chroma[alt_uq_pc, f0:f1]))

            # 对方唯一音能量显著更强 → 翻转
            if alt_energy > my_energy * 1.4:
                info = templates[alt_cid]
                logger.info(f"大小调消歧: {cid}→{alt_cid} @t={t0:.1f}s "
                            f"(my_uq={my_energy:.3f} alt_uq={alt_energy:.3f}, "
                            f"my_score={my_score:.3f} alt_score={alt_score:.3f})")
                d = dict(d)
                d["chord_id"] = alt_cid
                d["chord_name"] = info.get("chord_name", alt_cid)
                d["confidence"] = round(d.get("confidence", 0.5) * 0.95, 2)
            fixed.append(d)

        return fixed

    def _verify_seventh_chords(self, detected, chroma, frame_times, templates, sr, hop_length):
        """两步七和弦验证：以已确定的三和弦为锚点，独立评估7音候选频率是否有证据。

        核心思路（ChordFormer 结构化分解简化版）：
        不靠 chroma 模板匹配的得分差来区分 triad vs 7th（泛音会模糊边界），
        而是在 triad 确定后，单独回答「三和弦之外有没有7音？」。

        证据模型：
        - 7音能量 / 三和弦平均能量 > 阈值 → 有7音
        - 同时检查：7音能量减去预期泛音泄露后的净值 > 0
        - 更高 bins_per_octave(36) 使 CQT 滤波更窄，减少泛音跨 bin 污染
        """
        if not detected or len(templates) < 2:
            return detected

        n_frames = len(frame_times)
        if n_frames == 0:
            return detected

        result = []
        for d in detected:
            cid = d.get("chord_id", "")

            # pcset fallback 已从 original → cid 纠正过。
            # 如果升级候选与原被拒绝的和弦是同一和弦（或其超集），说明 fallback 已经明确否决了这个方向，跳过
            _orig_cid = d.get("_original_template_cid", "")

            info = templates.get(cid, {})
            pcset = info.get("pcset", set())
            if not pcset or len(pcset) < 3:
                result.append(d)
                continue

            # 提取该和弦段的 chroma 均值
            t0 = d.get("time", 0)
            t1 = d.get("end_time", t0 + 1.0)
            f0 = max(0, int(t0 * sr / hop_length))
            f1 = min(n_frames, int(t1 * sr / hop_length) + 1)
            if f1 <= f0:
                result.append(d)
                continue
            seg_c = np.mean(chroma[:, f0:f1], axis=1)  # 12-dim segment chroma

            # 查找以当前和弦 pcset 为子集的所有七和弦候选
            best_upgrade = None
            best_evidence = 0.0
            for cand_cid, cand_info in templates.items():
                cand_pcset = cand_info.get("pcset", set())
                if len(cand_pcset) != len(pcset) + 1:
                    continue
                if not pcset.issubset(cand_pcset):
                    continue

                # maj7 候选：5th 的5次泛音恰好落在 maj7 音位置，
                # 在12-bin chroma 中更难区分 → 提高阈值
                cand_name = cand_info.get("chord_name", cand_cid)
                _is_maj7 = "maj7" in cand_name.lower() or "maj7" in cand_cid.lower()

                # pcset fallback 已拒绝过的基础三和弦方向：不再试探其7和弦变体
                # 用 proper subset (<) 而非 subset (<=)：被拒绝的是三和弦(Dm)、候选是其7和弦(Dm7) → 阻挡
                # 但被拒绝的是7和弦(Bm7)本身、候选相同 → 放行（pcset阈值过严可能是误判）
                if _orig_cid:
                    _orig_pc = templates.get(_orig_cid, {}).get("pcset", set())
                    if _orig_pc and _orig_pc < cand_pcset:
                        # 只阻挡同根音方向：chroma说Am、pcset纠正为F → Fmaj7应放行(不同根音)
                        # 但 chroma说Dm、pcset纠正为F → Dm7应阻挡(同根音Dm方向已被否决)
                        _orig_root = _orig_cid[0].upper()
                        _cand_root = cand_cid[0].upper()
                        if _orig_root == _cand_root:
                            logger.debug(f"7th BLOCKED by orig: {cid}→{cand_cid} @t={d.get('time', 0):.1f}s "
                                        f"orig={_orig_cid} same_root=True")
                            continue

                extra_pc = list(cand_pcset - pcset)[0]

                # 7音能量
                extra_e = float(seg_c[extra_pc])
                # 三和弦各音能量
                triad_es = [float(seg_c[p]) for p in pcset]
                triad_avg = float(np.mean(triad_es)) if triad_es else 0.001

                if triad_avg < 0.005:
                    continue

                ratio = extra_e / triad_avg

                # root-referenced ratio: 7音 vs 根音（根音通常最强，更有区分力）
                _root_pc = list(pcset)[0]  # pcset 首个元素是根音
                _root_e = float(seg_c[_root_pc])
                ratio_vs_root = extra_e / max(_root_e, 0.001)

                # 泛音泄露基线：纯三和弦在7音位置的预期泛音能量
                # root第7泛音≈7音(偏平)，5th第3泛音≈7音(偏高)
                # 36 bins/octave 的 CQT 比 24 窄 33%，泛音泄露更少
                # 保守估计：泄露不超过 triad_avg 的 20%
                overtone_leak = triad_avg * 0.20
                net_evidence = extra_e - overtone_leak

                # chroma rank: 真正的7音应是段内较响的音高类别（前6/12）
                _rank = int(np.sum(seg_c > extra_e)) + 1  # 1=最响
                if _rank > 6:
                    continue

                # 7音至少要比一个三和弦音更响，否则大概率是泛音泄露
                if extra_e <= min(triad_es):
                    continue

                if ratio >= 0.30:  # 仅记录有意义的候选用于调试
                    logger.debug(f"7th CANDIDATE: {cid}→{cand_cid} @t={d.get('time', 0):.1f}s "
                                f"ratio={ratio:.3f} r_vs_root={ratio_vs_root:.3f} net={net_evidence:.4f} rank={_rank}/12 "
                                f"fallback={d.get('_pcset_fallback', False)} maj7={_is_maj7} "
                                f"orig={_orig_cid}")

                # pcset fallback 否决过的段需要更强证据
                _is_fallback = d.get("_pcset_fallback", False)
                if _is_maj7:
                    min_rvr = 0.55  # maj7: 3rd泛音(E→B)泄露严重，需更强7音证据
                elif _is_fallback:
                    min_rvr = 0.38
                else:
                    min_rvr = 0.35
                # pcset fallback 把和弦从 X 降为 Y，7th verify 又想从 Y 升回 X → 循环
                # 需要比普通升级强得多的证据才能打破循环
                if _orig_cid and _orig_cid == cand_cid:
                    min_rvr = max(min_rvr, 0.50)

                if ratio_vs_root >= min_rvr and ratio >= 0.45 and net_evidence > 0.008:
                    evidence = ratio * max(0, net_evidence)
                    if evidence > best_evidence:
                        best_evidence = evidence
                        best_upgrade = (cand_cid, cand_info, extra_pc, ratio, net_evidence)

            if best_upgrade:
                cand_cid, cand_info, extra_pc, ratio, net_e = best_upgrade
                d = dict(d)
                d["_7th_verified"] = True
                d["_pre_7th_chord_id"] = cid
                d["chord_id"] = cand_cid
                d["chord_name"] = cand_info.get("chord_name", cand_cid)
                logger.info(f"7th verify UPGRADE: {cid}→{cand_cid} "
                            f"@t={d.get('time', 0):.1f}s (ratio={ratio:.2f} net={net_e:.4f})")

            result.append(d)

        return result

    def _adjust_chords_for_capo(self, chords: List[Dict], capo: int):
        """根据变调夹调整和弦名标注（chroma 按 capo=0 匹配，需映射回真实指法）"""
        if capo <= 0:
            return
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        for c in chords:
            cid = c.get("chord_id", "")
            # 提取根音
            base = cid[0].upper()
            if len(cid) > 1 and cid[1] in ("#", "b"):
                base = cid[:2]
            if base in note_names:
                idx = note_names.index(base)
                new_idx = (idx - capo) % 12
                new_note = note_names[new_idx]
                c["_capo_chord_id"] = cid
                c["_actual_chord_id"] = cid.replace(base, new_note, 1) if base in cid else cid

    # ==================== 起始检测 & 节拍跟踪 ====================

    def _detect_playing_onset(self, audio: np.ndarray, sr: int = 16000) -> float:
        """检测演奏起始时间（第一个拨弦位置）。

        Returns:
            float: 起始时间（秒），若检测失败返回 0.0
        """
        if not LIBROSA_AVAILABLE or len(audio) < sr * 0.3:
            return 0.0

        try:
            audio_f32 = audio.astype(np.float32)
            onset_env = librosa.onset.onset_strength(y=audio_f32, sr=sr)
            onset_frames = librosa.onset.onset_detect(
                onset_envelope=onset_env, sr=sr,
                backtrack=False, delta=0.08,
            )
            if len(onset_frames) > 0:
                first_onset = librosa.frames_to_time(
                    onset_frames[0], sr=sr, hop_length=512
                )
                if first_onset > 0.3:
                    logger.info(f"演奏起始检测: {first_onset:.2f}s")
                    return round(first_onset, 2)
            return 0.0
        except Exception as e:
            logger.warning(f"起始检测失败: {e}")
            return 0.0

    def _track_beats(self, audio: np.ndarray, sr: int = 16000,
                      onset_time: float = 0.0) -> Dict:
        """节拍跟踪：检测 BPM 和拍点位置。

        Returns:
            {"tempo_bpm": float, "beat_times": [float, ...], "downbeat_times": [float, ...]}
        """
        if not LIBROSA_AVAILABLE or len(audio) < sr * 1.0:
            return {"tempo_bpm": 0.0, "beat_times": [], "downbeat_times": []}

        try:
            audio_f32 = audio.astype(np.float32)
            tempo, beat_frames = librosa.beat.beat_track(y=audio_f32, sr=sr)
            tempo = float(tempo)

            beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)

            if onset_time > 0:
                beat_times = [t for t in beat_times if t >= onset_time - 0.1]

            downbeat_times = beat_times[::4] if len(beat_times) >= 4 else []

            logger.info(f"节拍跟踪: tempo={tempo:.0f} BPM, {len(beat_times)} 拍, "
                        f"{len(downbeat_times)} 小节")
            return {
                "tempo_bpm": round(tempo, 1),
                "beat_times": [round(float(t), 2) for t in beat_times],
                "downbeat_times": [round(float(t), 2) for t in downbeat_times],
            }
        except Exception as e:
            logger.warning(f"节拍跟踪失败: {e}")
            return {"tempo_bpm": 0.0, "beat_times": [], "downbeat_times": []}

    # ==================== 和弦驱动音符过滤 ====================

    def _verify_chords_with_notes(self, detected_chords: List[Dict],
                                   notes: List[Dict], capo: int = 0) -> List[Dict]:
        """用 basic-pitch 音符做安全过滤 + 根音校验。

        chroma 检测在和弦指法空间（已按 capo 移位），basic-pitch 音符在实际音高空间。
        比对时需要把音符按 capo 移回指法空间。

        规则：
        1. 音符太少（<3个）→ 丢弃
        2. 根音缺失 → 丢弃
        3. 和弦音覆盖 < 25% → 丢弃
        """
        if not detected_chords or not notes:
            return detected_chords

        templates = self._load_chord_chroma_templates()
        if len(templates) < 2:
            return detected_chords

        ROOT_MAP = {
            "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
            "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
            "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
        }

        def _get_root_pc(chord_id: str) -> int:
            if not chord_id:
                return None
            cid = chord_id.replace("chord-", "")
            for name in sorted(ROOT_MAP.keys(), key=len, reverse=True):
                if cid.startswith(name):
                    return ROOT_MAP[name]
            return None

        verified = []
        for chord in detected_chords:
            t0 = chord.get("time", 0)
            t1 = chord.get("end_time", t0 + 2.0)
            chroma_cid = chord.get("chord_id", "")

            # 收集音符，按 capo 移位到指法空间（与 chroma 对齐）
            pc_counts = np.zeros(12, dtype=np.float32)
            note_count = 0
            for note in notes:
                nt = note.get("start_time", 0)
                if t0 - 0.3 <= nt <= t1 + 0.3:
                    pc = note.get("pitch", 60) % 12
                    if capo > 0:
                        pc = (pc - capo) % 12  # 实际音高 → 指法空间
                    dur = note.get("duration", 0.2)
                    pc_counts[pc] += min(dur, 0.5)
                    note_count += 1

            total_energy = float(np.sum(pc_counts))
            info = templates.get(chroma_cid, {})
            pcset = info.get("pcset", set())

            if note_count >= 2 and total_energy > 0.01 and pcset:
                chord_energy = sum(pc_counts[pc] for pc in pcset)
                coverage = chord_energy / total_energy
                chord["_note_verified"] = bool(coverage >= 0.15)
                # 仅当 notes 明确反对 chroma（覆盖率极低）时才降置信度
                if coverage < 0.10 and note_count >= 3:
                    chord["confidence"] = round(float(chord.get("confidence", 0.5)) * 0.85, 2)
            else:
                chord["_note_verified"] = False

            verified.append(chord)

        resolved = self._resolve_confusable_chords(verified, notes, capo, templates)
        logger.info(f"和弦验证(capo={capo}): {len(detected_chords)} -> {len(verified)} -> {len(resolved)} "
                    f"(confusable: {sum(1 for c in resolved if c.get('_confusable_flip'))} flipped)")
        return resolved

    def _resolve_confusable_chords(self, chords: List[Dict], notes: List[Dict],
                                    capo: int = 0, templates: Dict = None) -> List[Dict]:
        """用根音能量 + 独特音高能量解决易混淆和弦组。

        支持两类混淆：
        1. 根音不同：用根音能量区分（如 Am/Fmaj7, Am7/C）
        2. 根音相同：用独特音高能量区分（如 Dm7/D7，差 F vs F#）
        """
        if not chords or not notes:
            return list(chords) if chords else []

        if templates is None:
            templates = self._load_chord_chroma_templates()

        ROOT_MAP = {
            "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
            "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
            "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
        }

        def _get_root_pc(chord_id: str):
            if not chord_id:
                return None
            cid = chord_id.replace("chord-", "")
            for name in sorted(ROOT_MAP.keys(), key=len, reverse=True):
                if cid.startswith(name):
                    return ROOT_MAP[name]
            return None

        # 注意：C/Am, G/Em, F/Dm 已由 _disambiguate_relative_major_minor 处理，
        # 不要在此用根音能量比较——大和弦根音在小和弦内是 chord tone，根音能量不可靠
        CONFUSABLE_GROUPS = [
            ("Am/Fmaj7", ["Am", "fmaj7"]),
        ]

        chord_to_group = {}
        for gi, (_, members) in enumerate(CONFUSABLE_GROUPS):
            for m in members:
                chord_to_group.setdefault(m, []).append(gi)

        resolved = []
        for chord in chords:
            cid = chord.get("chord_id", "").replace("chord-", "")
            group_indices = chord_to_group.get(cid, [])

            if not group_indices:
                resolved.append(chord)
                continue

            t0 = chord.get("time", 0)
            t1 = chord.get("end_time", t0 + 2.0)
            pc_now = np.zeros(12, dtype=np.float32)
            for note in notes:
                nt = note.get("start_time", 0)
                if t0 - 0.3 <= nt <= t1 + 0.3:
                    pc = (note.get("pitch", 60) - capo) % 12
                    dur = note.get("duration", 0.2)
                    pc_now[pc] += min(dur, 0.5)

            total_e = float(np.sum(pc_now))
            if total_e < 0.01:
                resolved.append(chord)
                continue

            best_cid = cid
            best_root_e = 0

            # 高置信度 chroma 和弦不参与 note 翻转
            chroma_conf = chord.get("confidence", 0.5)
            if chroma_conf >= 0.75:
                resolved.append(chord)
                continue
            # 意图纠正过的和弦不参与 note 翻转
            if chord.get("_intent_corrected"):
                resolved.append(chord)
                continue

            # Phase 1: 根音能量裁决
            current_bias = 0.02 + max(0, (chroma_conf - 0.65) * 0.25)
            for gi in group_indices:
                for candidate in CONFUSABLE_GROUPS[gi][1]:
                    c_tpl = templates.get(candidate, {})
                    if not c_tpl.get("pcset"):
                        continue
                    root_pc_c = _get_root_pc(candidate)
                    if root_pc_c is None:
                        continue
                    root_e = pc_now[root_pc_c] / total_e
                    bias = current_bias if candidate == cid else 0
                    score = root_e + bias
                    if score > best_root_e + 0.05:
                        best_root_e = score
                        best_cid = candidate

            # Phase 1.5: 不同根音+pcset子集关系时，用独特音高替代根音比较
            # 例: Am/Fmaj7 — Am根音A在Fmaj7内是3音，根音能量不可靠
            if best_cid != cid:
                for gi in group_indices:
                    group_members = CONFUSABLE_GROUPS[gi][1]
                    if cid not in group_members or best_cid not in group_members:
                        continue
                    cur_pcset = templates.get(cid, {}).get("pcset", set())
                    best_pcset = templates.get(best_cid, {}).get("pcset", set())
                    if not cur_pcset or not best_pcset:
                        continue
                    # 一个pcset是另一个的子集 → 比较超集的独特音高
                    if cur_pcset.issubset(best_pcset) and cur_pcset != best_pcset:
                        unique_pcs = best_pcset - cur_pcset
                        unique_e = sum(pc_now[pc] for pc in unique_pcs) / total_e
                        if unique_e < 0.06:
                            # 超集独特音高能量太弱 → 保持子集(更简单的和弦)
                            best_cid = cid
                    elif best_pcset.issubset(cur_pcset) and best_pcset != cur_pcset:
                        unique_pcs = cur_pcset - best_pcset
                        unique_e = sum(pc_now[pc] for pc in unique_pcs) / total_e
                        if unique_e < 0.06:
                            # 当前和弦的额外音高太弱 → 选择更简单的和弦
                            best_cid = best_cid  # already correct

            # Phase 2: 同根音组用独特音高能量进一步裁决
            if best_cid == cid:
                for gi in group_indices:
                    group_members = CONFUSABLE_GROUPS[gi][1]
                    roots = {}
                    for m in group_members:
                        r = _get_root_pc(m)
                        if r is not None:
                            roots.setdefault(r, []).append(m)
                    for _root_pc, same_root_candidates in roots.items():
                        if len(same_root_candidates) < 2 or cid not in same_root_candidates:
                            continue
                        all_pcs = set()
                        for m in same_root_candidates:
                            all_pcs |= templates.get(m, {}).get("pcset", set())
                        # 交集：所有候选和弦共享的音高（如 Cmaj7/C 共享 C,E,G）
                        intersection = all_pcs.copy()
                        for m in same_root_candidates:
                            intersection &= templates.get(m, {}).get("pcset", set())
                        candidate_scores = {}
                        for m in same_root_candidates:
                            m_pcset = templates.get(m, {}).get("pcset", set())
                            unique_pcs = m_pcset - intersection
                            if unique_pcs:
                                unique_e = sum(pc_now[pc] for pc in unique_pcs) / total_e
                            else:
                                unique_e = 0.0
                            # 简约性惩罚：额外音高需要显著能量才值得选更复杂的和弦
                            penalty = len(unique_pcs) * 0.06
                            bias = 0.02 if m == cid else 0
                            candidate_scores[m] = unique_e - penalty + bias
                        if len(candidate_scores) < 2:
                            continue
                        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
                        best_unique_cid, best_unique_score = sorted_candidates[0]
                        second_best_score = sorted_candidates[1][1]
                        if best_unique_cid != cid and best_unique_score > second_best_score + 0.05:
                            best_cid = best_unique_cid

            if best_cid != cid:
                info_new = templates.get(best_cid, {})
                flip_reason = "root_e" if _get_root_pc(best_cid) != _get_root_pc(cid) else "unique_pc"
                logger.info(f"confusable flip: {chord.get('chord_name','?')} → {info_new.get('chord_name', best_cid)} "
                           f"@t={t0:.1f}s ({flip_reason} flip)")
                chord = dict(chord)
                chord["chord_id"] = best_cid
                chord["chord_name"] = info_new.get("chord_name", best_cid + "和弦")
                chord["confidence"] = round(chord.get("confidence", 0.5) * 0.92, 2)
                chord["_confusable_flip"] = True

            resolved.append(chord)

        return resolved

    def _score_chords_against_notes(self, chords: List[Dict], notes: List[Dict],
                                      capo: int = 0) -> float:
        """计算和弦序列与音符的整体匹配度（用于 capo 歧义裁决）。"""
        if not chords or not notes:
            return 0.0
        templates = self._load_chord_chroma_templates()
        total_score = 0.0
        for chord in chords:
            cid = chord.get("chord_id", "")
            info = templates.get(cid, {})
            if not info:
                continue
            t0 = chord.get("time", 0)
            t1 = chord.get("end_time", t0 + 2.0)
            pc_energy = np.zeros(12)
            total_energy = 0.0
            for note in notes:
                nt = note.get("start_time", 0)
                if t0 - 1.0 <= nt <= t1 + 0.5:
                    pc = (note.get("pitch", 60) - capo) % 12
                    vel = note.get("velocity", 50) / 127.0
                    pc_energy[pc] += vel
                    total_energy += vel
            if total_energy < 0.01:
                continue
            pcset = info.get("pcset", set())
            if pcset:
                chord_energy = sum(pc_energy[pc] for pc in pcset)
                total_score += chord_energy / total_energy
        return total_score / max(len(chords), 1)

    def _refine_chords_with_notes(self, detected_chords: List[Dict],
                                    notes: List[Dict], capo: int = 0) -> List[Dict]:
        """用 basic-pitch 音符补充和纠正 chroma 和弦检测。

        1. 纠正：相邻和弦共享2音时 chroma 易混淆（C↔Am, G↔Em, F↔Dm），
           用音符分布纠正为更匹配的和弦
        2. 补充：chroma 漏检的区间，用音符推断和弦填入
        """
        if not notes:
            return detected_chords

        templates = self._load_chord_chroma_templates()
        if len(templates) < 2:
            return detected_chords

        ROOT_MAP = {
            "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
            "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
            "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
        }

        def _get_root_pc(chord_id: str) -> int:
            if not chord_id:
                return None
            cid = chord_id.replace("chord-", "")
            for name in sorted(ROOT_MAP.keys(), key=len, reverse=True):
                if cid.startswith(name):
                    return ROOT_MAP[name]
            return None

        sorted_notes = sorted(notes, key=lambda n: n.get("start_time", 0))

        def _note_score_for_chord(t0, t1, cid, info, tight=False):
            """计算某个和弦在 [t0, t1] 区间的音符匹配分 (0-1)。

            tight=True: 使用紧凑窗口 [t0, t1+0.5]，用于纠正时避免前一个和弦音符渗入
            tight=False: 使用宽窗口 [t0-1.0, t1+0.5]，用于补漏时捕获边界附近的根音
            """
            pcset = info.get("pcset", set())
            if not pcset:
                return 0.0
            pc_energy = np.zeros(12)
            total_energy = 0.0
            pre_margin = 0.0 if tight else 1.0
            for note in sorted_notes:
                nt = note.get("start_time", 0)
                if t0 - pre_margin <= nt <= t1 + 0.5:
                    pc = note.get("pitch", 60) % 12
                    if capo > 0:
                        pc = (pc - capo) % 12
                    vel = note.get("velocity", 50) / 127.0
                    pc_energy[pc] += vel
                    total_energy += vel
            if total_energy < 0.01:
                return 0.0
            chord_energy = sum(pc_energy[pc] for pc in pcset)
            root_pc = _get_root_pc(cid)
            root_score = pc_energy[root_pc] / total_energy if root_pc is not None else 0
            chord_cov = chord_energy / total_energy
            # 音数惩罚：4音和弦（如Am7）天然得分≥3音和弦（Am），轻微惩罚大和弦
            # 让简洁和弦在分数接近时胜出
            n_tones = len(pcset)
            size_penalty = max(0, (n_tones - 3) * 0.04)
            return max(0.0, chord_cov * 0.4 + root_score * 0.6 - size_penalty)

        # === Step 1: 保留 chroma 原始结果（不做纠正） ===
        # basic-pitch 音符转录在复音场景下不可靠（漏根音、泛音干扰），
        # 用它去"纠正" chroma (CQT频谱直接分析) 是本末倒置，会引入更多错误
        refined = list(detected_chords)

        # === Step 2: 补充漏检区间 ===
        # 只在 chroma 结果极少(<3段)时才启用音符填空
        # gap filling 容易引入假阳性，仅作为最后的 fallback
        if not refined:
            gaps = [(0.0, float(sorted_notes[-1].get("start_time", 30)) + 2)]
            fill_threshold = 0.60
        elif len(refined) >= 3:
            logger.info(f"音符填空: chroma 已检测 {len(refined)} 段，跳过补充")
            return refined
        else:
            gaps = []
            prev_end = 0.0
            for chord in sorted(refined, key=lambda c: c.get("time", 0)):
                t0 = chord.get("time", 0)
                if t0 - prev_end >= 3.0:
                    gaps.append((prev_end, t0))
                prev_end = max(prev_end, chord.get("end_time", t0 + 1))
            t_last = sorted_notes[-1].get("start_time", 30) + 2 if sorted_notes else 30
            if t_last - prev_end > 3.0:
                gaps.append((prev_end, t_last))
            fill_threshold = 0.60

        new_chords = []
        raw_names = []

        # 构建 chroma 评分查询（比 note 评分更可靠，因为直接基于频谱）
        _chroma_scores = getattr(self, '_chroma_scores', None)
        _chroma_times = getattr(self, '_chroma_frame_times', None)
        _chroma_ids = getattr(self, '_chroma_chord_ids', None)
        _has_chroma = _chroma_scores is not None and _chroma_times is not None and _chroma_ids is not None

        def _chroma_score_for_chord(t0, t1, cid):
            if not _has_chroma:
                return 0.0
            try:
                row = _chroma_ids.index(cid)
            except ValueError:
                return 0.0
            idx0 = int(np.searchsorted(_chroma_times, t0))
            idx1 = int(np.searchsorted(_chroma_times, t1))
            if idx1 <= idx0:
                return 0.0
            return float(np.mean(_chroma_scores[row, idx0:idx1]))

        for gap_start, gap_end in gaps:
            window = 2.0
            step = 1.0
            pos = gap_start
            while pos + window <= gap_end:
                best_cid = None
                best_score = 0.0
                for cid, info in templates.items():
                    note_s = _note_score_for_chord(pos, pos + window, cid, info)
                    chroma_s = _chroma_score_for_chord(pos, pos + window, cid)
                    # chroma 权重 0.6 > notes 权重 0.4，因为频谱比音符转录更可靠
                    score = note_s * 0.4 + chroma_s * 0.6
                    if score > best_score:
                        best_score = score
                        best_cid = cid
                if best_cid and best_score > fill_threshold:
                    info = templates[best_cid]
                    raw_names.append(f"{info.get('chord_name', best_cid)}({pos:.1f}s,{best_score:.2f})")
                    new_chords.append({
                        "time": round(pos, 2),
                        "end_time": round(pos + window, 2),
                        "chord_id": best_cid,
                        "chord_name": info.get("chord_name", best_cid),
                        "confidence": round(best_score * 0.85, 2),
                        "related_problems": info.get("related_problems", []),
                        "related_techniques": info.get("related_techniques", []),
                        "_source": "notes",
                        "_note_verified": True,
                    })
                    pos += window  # 跳过已找到的和弦，避免滑动窗口重叠
                else:
                    if best_cid and best_score > fill_threshold * 0.7:
                        info = templates[best_cid]
                        logger.debug(f"补漏略过: {info.get('chord_name', best_cid)}@{pos:.1f}s score={best_score:.2f} < {fill_threshold}")
                    pos += step

        # 合并相邻同和弦
        if raw_names:
            logger.info(f"补漏原始窗口: {raw_names}")
        elif gaps:
            logger.info(f"笔记补充: {len(gaps)} 个缺口均未达到阈值({fill_threshold})")
        merged_new = []
        for nc in new_chords:
            if merged_new and merged_new[-1]["chord_id"] == nc["chord_id"] \
                    and nc["time"] - merged_new[-1]["end_time"] < 1.5:
                merged_new[-1]["end_time"] = nc["end_time"]
                merged_new[-1]["confidence"] = max(merged_new[-1]["confidence"], nc["confidence"])
            else:
                merged_new.append(nc)

        # 过滤短段
        merged_new = [c for c in merged_new if c["end_time"] - c["time"] >= 1.5]

        if merged_new:
            names = [f"{c['chord_name']}({c['time']:.1f}s,{c['confidence']:.2f})" for c in merged_new]
            logger.info(f"笔记补充: {len(merged_new)} 个和弦填入 chroma 漏检区间: {names}")

        all_chords = refined + merged_new
        all_chords.sort(key=lambda c: c.get("time", 0))

        # === 最终校验：只对补漏和弦（note-sourced）做音符复核，不碰 chroma 和弦 ===
        corrected_final = []
        for chord in all_chords:
            is_note_sourced = chord.get("_source", "") == "notes"
            if not is_note_sourced:
                corrected_final.append(chord)
                continue
            cid = chord.get("chord_id", "")
            info = templates.get(cid, {})
            t0 = chord.get("time", 0)
            t1 = chord.get("end_time", t0 + 2.0)
            note_s = _note_score_for_chord(t0, t1, cid, info, tight=True)
            chroma_s = _chroma_score_for_chord(t0, t1, cid)
            current_score = note_s * 0.4 + chroma_s * 0.6
            best_cid = cid
            best_score = current_score
            for alt_cid, alt_info in templates.items():
                if alt_cid == cid:
                    continue
                alt_note = _note_score_for_chord(t0, t1, alt_cid, alt_info, tight=True)
                alt_chroma = _chroma_score_for_chord(t0, t1, alt_cid)
                alt_score = alt_note * 0.4 + alt_chroma * 0.6
                if alt_score > best_score + 0.08:
                    best_score = alt_score
                    best_cid = alt_cid
            if best_cid != cid:
                alt_info = templates[best_cid]
                logger.info(f"补漏校验: {info.get('chord_name',cid)} → {alt_info.get('chord_name',best_cid)} "
                           f"@t={t0:.1f}s (score {current_score:.2f}→{best_score:.2f})")
                chord = dict(chord)
                chord["chord_id"] = best_cid
                chord["chord_name"] = alt_info.get("chord_name", best_cid)
                chord["confidence"] = round(best_score * 0.85, 2)
                chord["_note_corrected"] = True
            corrected_final.append(chord)

        # 对补漏和弦也运行混淆组裁决（如 Dm7/D7, Am7/C）
        corrected_final = self._resolve_confusable_chords(corrected_final, notes, capo, templates)
        return corrected_final

    def _correct_chords_during_strumming(self, detected_chords: List[Dict],
                                          strum_segments: List[Dict],
                                          notes: List[Dict]) -> List[Dict]:
        """扫弦段和弦校正：利用扫弦时同时发音的音符簇直接识别和弦。

        扫弦时多弦同时发声，音符簇的 pitch class 集合比 chroma 谱图更可靠。
        对于每个扫弦段内检测到的音符簇，匹配和弦模板，若与 chroma 结果不同则覆写。
        """
        if not strum_segments or not notes:
            return detected_chords

        templates = self._load_chord_chroma_templates()
        if len(templates) < 2:
            return detected_chords

        sorted_notes = sorted(notes, key=lambda n: n.get("start_time", 0))
        STRUM_CLUSTER_MS = 0.05  # 50ms 内视为同一次扫弦

        # 收集扫弦段内的和弦
        strum_chords = []
        for seg in strum_segments:
            t0 = seg.get("start_time", 0)
            t1 = seg.get("end_time", t0 + 0.5)
            # 获取该段内的所有音符
            seg_notes = [n for n in sorted_notes if t0 - 0.15 <= n.get("start_time", 0) <= t1 + 0.15]
            if not seg_notes:
                continue

            # 聚类为音符簇（同一次扫弦）
            clusters = []
            current_cluster = [seg_notes[0]]
            for note in seg_notes[1:]:
                if note.get("start_time", 0) - current_cluster[0].get("start_time", 0) <= STRUM_CLUSTER_MS:
                    current_cluster.append(note)
                else:
                    if len(current_cluster) >= 2:
                        clusters.append(current_cluster)
                    current_cluster = [note]
            if len(current_cluster) >= 2:
                clusters.append(current_cluster)

            for cluster in clusters:
                # 提取 pitch class 集合
                pcs = set()
                for n in cluster:
                    pc = n.get("pitch", 60) % 12
                    pcs.add(pc)

                if len(pcs) < 2:
                    continue

                # 与和弦模板匹配（Jaccard相似度）
                best_cid = None
                best_score = 0.0
                for cid, info in templates.items():
                    tpl_pcs = info.get("pcset", set())
                    if not tpl_pcs:
                        continue
                    intersection = len(pcs & tpl_pcs)
                    union = len(pcs | tpl_pcs)
                    jaccard = intersection / union if union > 0 else 0
                    # 加分：根音匹配
                    info_root = info.get("root_pc")
                    if info_root is None and "pcset" in info:
                        # 根音推断：取 pitch class set 中最低的音（通常是根音）
                        info_root = min(info["pcset"]) if info["pcset"] else None
                    root_bonus = 0.15 if (info_root is not None and info_root in pcs) else 0
                    score = jaccard + root_bonus
                    if score > best_score:
                        best_score = score
                        best_cid = cid

                if best_cid and best_score > 0.3:
                    cluster_time = cluster[0].get("start_time", 0)
                    info = templates[best_cid]
                    strum_chords.append({
                        "time": round(cluster_time - 0.1, 2),
                        "end_time": round(cluster[-1].get("start_time", cluster_time) + 0.3, 2),
                        "chord_id": best_cid,
                        "chord_name": info.get("chord_name", best_cid),
                        "confidence": round(min(0.90, best_score + 0.15), 2),
                        "_source": "strumming_notes",
                        "_strum_score": round(best_score, 3),
                    })

        if not strum_chords:
            return detected_chords

        # 合并相邻同和弦
        merged = []
        for sc in sorted(strum_chords, key=lambda c: c["time"]):
            if merged and merged[-1]["chord_id"] == sc["chord_id"] \
                    and sc["time"] - merged[-1]["end_time"] < 0.8:
                merged[-1]["end_time"] = sc["end_time"]
                merged[-1]["confidence"] = max(merged[-1]["confidence"], sc["confidence"])
            else:
                merged.append(sc)
        strum_chords = merged

        # 用扫弦和弦覆写 chroma 结果中对应时间窗口
        corrected = list(detected_chords)
        override_count = 0
        for sc in strum_chords:
            st = sc["time"]
            se = sc["end_time"]
            # 查找覆盖的 chroma 和弦
            overridden = False
            for i, dc in enumerate(corrected):
                dt = dc.get("time", 0)
                de = dc.get("end_time", dt + 2.0)
                # 时间重叠
                if dt - 0.3 <= st <= de + 0.3 or st - 0.3 <= dt <= se + 0.3:
                    if dc.get("chord_id") != sc["chord_id"]:
                        logger.info(f"扫弦和弦覆写: {dc.get('chord_name','?')} → {sc['chord_name']} "
                                   f"@t={st:.1f}s (strum_score={sc.get('_strum_score','?')})")
                        corrected[i] = sc
                        override_count += 1
                    overridden = True
            if not overridden:
                # 没有覆盖到任何 chroma 和弦 → 作为新和弦插入
                corrected.append(sc)
                logger.info(f"扫弦和弦补充: {sc['chord_name']} @t={st:.1f}s-{se:.1f}s")
                override_count += 1

        if override_count > 0:
            corrected.sort(key=lambda c: c.get("time", 0))
            logger.info(f"扫弦和弦校正: {override_count} 处覆写/补充")

        return corrected

    def _filter_notes_by_chords(self, notes: List[Dict],
                                  detected_chords: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """用和弦检测结果过滤 basic-pitch 音符。

        对每个和弦时间段内的音符：
        - 音高类别在期望集合内 → 保留
        - 音高类别不在期望集合内 → 标记为疑似幻音
        - 期望音高缺失 → 标记为可能闷音/漏弹

        Returns:
            (filtered_notes, phantom_notes)
        """
        if not notes or not detected_chords:
            return notes, []

        templates = self._load_chord_chroma_templates()
        if not templates:
            return notes, []

        phantom_notes = []
        filtered = []

        for note in notes:
            t = note.get("start_time", 0)
            pitch = note.get("pitch", 60)
            pc = pitch % 12

            covering_chord = None
            for chord in detected_chords:
                ct = chord.get("time", 0)
                c_end = chord.get("end_time", ct + 2.0)
                if ct - 0.3 <= t <= c_end + 0.3:
                    if covering_chord is None or abs(t - ct) < abs(t - covering_chord.get("time", 0)):
                        covering_chord = chord

            if covering_chord:
                cid = covering_chord.get("chord_id", "")
                info = templates.get(cid, {})
                expected_pcset = info.get("pcset", set())

                if expected_pcset and pc in expected_pcset:
                    note["_chord_verified"] = True
                    note["_chord_id"] = cid
                elif expected_pcset:
                    phantom_notes.append({
                        "time": t,
                        "pitch": pitch,
                        "note_name": note.get("note_name", ""),
                        "chord_id": cid,
                        "chord_name": covering_chord.get("chord_name", ""),
                    })
                    note["_phantom"] = True
                    note["_chord_id"] = cid

            filtered.append(note)

        if phantom_notes:
            logger.info(
                f"和弦驱动过滤: {len(phantom_notes)} 个幻音标记 "
                + ", ".join(
                    f"{p['note_name']}({p['time']:.1f}s, 期望{p['chord_name']})"
                    for p in phantom_notes[:5]
                )
            )

        return filtered, phantom_notes

    @staticmethod
    def _detect_capo_from_pitches(notes: List[Dict]) -> int:
        """用音符音高直接检测变调夹位置（不依赖弦位分配）。

        原理：标准调弦下，开放弦音高类别为 {D, E, G, A, B} = {2,4,7,9,11}。
        变调夹使这些整体上移N个半音。对每种 capo 假设，将音符移回指法空间，
        统计落在开放弦音高类别的比例。最高分的 capo 即为检测结果。

        这比 _detect_capo 更可靠，因为不需要先做弦位分配。
        """
        if len(notes) < 5:
            return 0

        import numpy as np
        OPEN_STRING_PCS = {2, 4, 7, 9, 11}  # D E G A B
        pitch_classes = np.array([n.get("pitch", 60) % 12 for n in notes], dtype=np.int32)
        velocities = np.array([min(n.get("velocity", 50), 100) / 100.0 for n in notes])

        best_capo = 0
        best_score = -1.0
        for capo_hyp in range(6):
            shifted = (pitch_classes - capo_hyp) % 12
            open_match = sum(velocities[i] for i, pc in enumerate(shifted) if pc in OPEN_STRING_PCS)
            total = velocities.sum()
            score = open_match / total if total > 0 else 0
            if score > best_score:
                best_score = score
                best_capo = capo_hyp

        logger.info(f"Pitch capo 检测: capo={best_capo} (score={best_score:.2f})")
        return best_capo

    @staticmethod
    def _detect_capo(notes: List[Dict]) -> int:
        """自动检测变调夹位置。

        原理：变调夹使所有弦的空弦音升高N个半音，basic-pitch检测到后，
        tab assigner会把所有本该空弦的音映射到「X弦N品」。
        如果多根弦的最低品位都是同一个正整数N，则推断 capo = N。

        Returns: 检测到的变调夹品位 (0=无变调夹)
        """
        if len(notes) < 3:
            return 0

        # 统计每根弦的最低品位
        min_fret_per_string = {}
        for n in notes:
            s = n.get("string")
            f = n.get("fret")
            if s is not None and f is not None:
                if s not in min_fret_per_string or f < min_fret_per_string[s]:
                    min_fret_per_string[s] = f

        if len(min_fret_per_string) < 2:
            return 0

        # 统计每个最低品位的弦数
        from collections import Counter
        fret_counts = Counter(min_fret_per_string.values())

        # 如果某个正值N出现在至少3根弦的最低品位中，判定为 capo=N
        for fret, count in fret_counts.most_common():
            if fret >= 1 and count >= 3:
                return int(fret)

        # 如果2根弦同样的正值，且没有0品弦，也可能是capo
        if 0 not in fret_counts:
            for fret, count in fret_counts.most_common(1):
                if fret >= 1 and count >= 2:
                    return int(fret)

        return 0

    def _midi_to_note(self, midi_pitch: int) -> str:
        """MIDI 音高转音符名称（已弃用，保留兼容）"""
        return _midi_to_note_name(midi_pitch)

    @staticmethod
    def _detect_phrase_gaps(notes: List[Dict]) -> Dict:
        """检测音符之间的乐句间隔，使用 IOI (Inter-Onset Interval) 离群值检测。

        原理：
        - 吉他延音导致音符高度重叠（80%+ end-to-start 为负值），
          传统「无声间隙」检测对吉他完全失效
        - 改用 onset-to-onset 间隔 (IOI)：当演奏者停顿换和弦时，
          相邻音符的起音间隔会显著大于典型 IOI
        - 用全量 IOI 分布计算中位数（含扫弦簇），确保基准不被拉高
        - IOI > 3x 全量中位数 且 > 0.5s → 标记为可能的换和弦停顿
        - 同时报告 top-5 最大 IOI，供 AI 辅助判断（即使未达阈值）

        Returns: {
            "gap_summary": "人类可读的间隔分析",
            "has_large_gaps": bool,
            "ioi_median": float,
            "ioi_p90": float,
            "gaps_detail": [...],
            "top_ioi_slowdowns": [...]
        }
        """
        if len(notes) < 4:
            return {"gap_summary": "", "has_large_gaps": False, "ioi_median": 0, "ioi_p90": 0,
                    "gaps_detail": [], "top_ioi_slowdowns": []}

        sorted_notes = sorted(notes, key=lambda n: n.get("start_time", 0))
        # 过滤幻音（被和弦标记为 _phantom 的音符）用于 IOI 分析
        clean_notes = [n for n in sorted_notes if not n.get("_phantom")]

        # 全量 IOI（用于计算分布基准，使用干净音符）
        all_gaps = []
        for i in range(1, len(clean_notes)):
            g = clean_notes[i]["start_time"] - clean_notes[i - 1]["start_time"]
            all_gaps.append(g)

        full_median = float(np.median(all_gaps))
        full_p90 = float(np.percentile(all_gaps, 90))

        # 非扫弦 IOI（用于找离群值，过滤 <50ms 的同时发音，使用干净音符）
        ioi_entries = []
        for i in range(1, len(clean_notes)):
            gap = clean_notes[i]["start_time"] - clean_notes[i - 1]["start_time"]
            if gap >= 0.05:
                ioi_entries.append({
                    "time": round(clean_notes[i - 1]["end_time"], 1),
                    "gap": round(gap, 2),
                })

        if len(ioi_entries) < 3:
            return {"gap_summary": "", "has_large_gaps": False, "ioi_median": round(full_median, 2),
                    "ioi_p90": round(full_p90, 2), "gaps_detail": [], "top_ioi_slowdowns": []}

        # 离群值检测：使用更保守的阈值区分延音和真正的停顿
        # 延音是正常的音乐表达，只有超长间隔才可能是换和弦不流畅
        # 阈值 = max(5x中位数, 2xP90, 1.2s) — 多重保护避免误报
        outlier_threshold = max(full_median * 5.0, full_p90 * 2.0, 1.2)
        large_gaps = []
        for x in ioi_entries:
            if x["gap"] > outlier_threshold:
                large_gaps.append({
                    "time": x["time"],
                    "duration": x["gap"],
                    "context": f"t={x['time']:.1f}s处起音间隔{x['gap']:.1f}s",
                })

        # 为每个大间隔计算局部 IOI 上下文（周围的节奏均匀度）
        for g in large_gaps:
            gap_time = g["time"]
            nearby_iois = [x["gap"] for x in ioi_entries
                           if abs(x["time"] - gap_time) < 3.0]  # ±3秒窗口
            if len(nearby_iois) >= 4:
                local_median = float(np.median(nearby_iois))
                local_cv = round(float(np.std(nearby_iois) / local_median), 3) if local_median > 0 else 0
                g["local_ioi_cv"] = local_cv
                g["local_ioi_median"] = round(local_median, 2)
                g["context"] += f"（周围3秒节奏{'均匀' if local_cv < 0.15 else '一般' if local_cv < 0.25 else '不太稳'}）"
            else:
                g["local_ioi_cv"] = None
                g["local_ioi_median"] = None

        # Top-5 最大 IOI（即使不达阈值，也供 AI 参考）
        sorted_ioi = sorted(ioi_entries, key=lambda x: x["gap"], reverse=True)
        top_slowdowns = []
        for x in sorted_ioi[:5]:
            top_slowdowns.append(
                f"⏱ {x['time']:.1f}s处起音间隔{x['gap']:.2f}s"
            )

        total_notes = len(clean_notes)
        gap_summary = (
            f"共{total_notes}个音符，全量IOI中位数{full_median:.2f}s，P90={full_p90:.2f}s。"
            f"离群阈值={outlier_threshold:.2f}s。"
        )
        if large_gaps:
            gap_summary += (
                f"发现{len(large_gaps)}处IOI间隔拉长（>{outlier_threshold:.2f}s）："
                + "；".join(g["context"] for g in large_gaps)
                + "。这些位置的起音间隔显著大于典型值。"
                  "注意：大间隔也可能是乐句之间的自然呼吸或段落切换，不一定是技巧问题。"
                  "请结合整体IOI变异系数(CV)来判断——CV低表示整体节奏均匀，大间隔更可能是刻意留白。"
            )
        else:
            gap_summary += "未发现超过阈值的IOI间隔拉长。"
        if top_slowdowns:
            gap_summary += (
                " 全曲最慢的5处起音间隔：" + "；".join(top_slowdowns)
                + "。请结合这些位置判断是否存在换和弦不流畅的情况。"
            )

        return {
            "gap_summary": gap_summary,
            "has_large_gaps": len(large_gaps) > 0,
            "ioi_median": round(full_median, 2),
            "ioi_p90": round(full_p90, 2),
            "gaps_detail": large_gaps,
            "top_ioi_slowdowns": top_slowdowns,
        }

    @staticmethod
    def _gaps_to_errors(gap_analysis: Dict) -> List[Dict]:
        """将确定性间隔检测结果转换为 transition_gap 错误。

        绕过 LLM，确保每次运行都能稳定检测到换和弦停顿。
        关键区分：延音（正常音乐表达）vs 真正换和弦卡顿。
        - 间隔在其局部上下文中是离群值（>2.5x局部中位数）→ 可能是停顿
        - 间隔与局部上下文一致（周围也都是长间隔）→ 延音，不报
        """
        errors = []
        gaps_detail = gap_analysis.get("gaps_detail", [])
        if not gaps_detail:
            return errors

        for g in gaps_detail:
            duration = g.get("duration", 0)
            if duration < 0.8:
                continue

            local_cv = g.get("local_ioi_cv")
            local_median = g.get("local_ioi_median")
            context = g.get("context", "")

            # 间隔是否在局部上下文中显著偏大？（延音 vs 卡顿的核心判断）
            is_local_outlier = (
                local_median is not None and local_median > 0
                and duration > local_median * 2.5
            )

            if not is_local_outlier:
                continue  # 周围也都是长间隔 → 延音，不报

            # 局部离群值 + 周围节奏不规则 → 真实卡顿（高置信度）
            if local_cv is not None and local_cv > 0.30:
                confidence = 0.80
                if duration > 2.5:
                    severity = "high"
                elif duration > 1.2:
                    severity = "medium"
                else:
                    severity = "low"
            elif local_cv is not None:
                # 周围节奏均匀但此处间隔偏大 → 可能是乐句边界或偶发停顿
                confidence = 0.60
                if duration > 3.0:
                    severity = "medium"
                else:
                    severity = "low"
            else:
                confidence = 0.55
                severity = "low"

            detail = (
                f"乐句间停顿约{duration:.1f}秒，疑似换和弦/换把位不够流畅。"
                f"{context}"
            )

            errors.append({
                "type": "transition_gap",
                "time": round(g.get("time", 0), 1),
                "detail": detail,
                "severity": severity,
                "confidence": confidence,
                "_source": "deterministic_gap",
            })

        return errors

    def _build_note_text(self, notes: List[Dict]) -> str:
        """将音符转化为可读的文字描述（吉他使用品位表示）。

        自动检测扫弦/和弦：50ms 内同时出现的多个音符合并为和弦块，
        避免 AI 把扫弦误判为分解和弦逐音分析。
        """
        if not notes:
            return "（无音符数据）"

        STRUM_WINDOW = 0.05  # 50ms 内视为同时发音
        sorted_notes = sorted(notes, key=lambda n: n["start_time"])
        clusters = []
        i = 0
        while i < len(sorted_notes):
            cluster = [sorted_notes[i]]
            t0 = sorted_notes[i]["start_time"]
            j = i + 1
            while j < len(sorted_notes) and sorted_notes[j]["start_time"] - t0 < STRUM_WINDOW:
                cluster.append(sorted_notes[j])
                j += 1
            clusters.append(cluster)
            i = j

        lines = []
        for cluster in clusters:
            if len(cluster) == 1:
                n = cluster[0]
                lines.append(
                    f"[{n['start_time']:.1f}s-{n['end_time']:.1f}s] "
                    f"note={n.get('note_name','?')} (vel={n['velocity']})"
                )
            else:
                # 和弦块：多个音符同时发音 → 这是扫弦或同时弹响的和弦
                t0 = cluster[0]["start_time"]
                t1 = max(n["end_time"] for n in cluster)
                avg_vel = int(sum(n.get('velocity', 0) for n in cluster) / len(cluster))
                note_names = [n.get('note_name', '?') for n in cluster]
                lines.append(
                    f"[{t0:.1f}s-{t1:.1f}s] 🎸 和弦块: {' + '.join(note_names)} "
                    f"(共{len(cluster)}音, avg_vel={avg_vel})"
                )

        return "\n".join(lines[:200])

    # ==================== DeepSeek 智能错误分析 ====================

    def _deepseek_analyze_errors(self, note_text: str, notes: List[Dict],
                                  instrument: str, features: Optional[AudioFeatures] = None,
                                  gap_analysis: Optional[Dict] = None) -> Tuple[List[Dict], Dict]:
        """用 DeepSeek 智能分析演奏错误。

        现在接收 AudioFeatures（信号分析层量化指标），LLM 在已知客观数据的基础上
        做解读，而非凭空猜测错误。
        """
        default_errors = []
        default_stats = {
            "total_notes": len(notes),
            "pitch_accuracy": 0,
            "rhythm_accuracy": 0,
            "overall_score": 70,
        }

        instrument_label = "吉他" if instrument.strip().lower() == "guitar" else "钢琴"

        # 构建特征摘要部分
        feature_section = ""
        if features and features.feature_summary:
            feature_section = f"""
## 信号分析层量化结果（仅供你内部参考，禁止在报告中直接引用）

{features.feature_summary}

**重要规则**:
1. 以上数据是代码计算的内部指标，仅供你判断演奏水平时参考
2. **严禁在报告/detail/任何输出中直接引用这些数字**——禁止出现dB、velocity、Hz、ms、频谱质心、非谐波能量比等词汇
3. 你需要做的是把这些数字「翻译」成大白话，例如：
   - 「动态范围5.9dB」→ 说「你的力度变化还可以更丰富一些」
   - 「velocity P50=62」→ 说「整体弹得偏轻」或「力度适中」
   - 「频谱质心=2800Hz」→ 说「音色偏亮」
4. 用老师对学生的自然语言，不要写实验报告

"""

        # 构建乐句间隔分析部分
        gap_section = "（无间隔数据）"
        if gap_analysis and gap_analysis.get("gap_summary"):
            gap_section = gap_analysis["gap_summary"]

        prompt = f"""你是一位资深吉他教师，正在审听一段吉他演奏的 MIDI 转录数据。

**⚠️ 分析模式：自由演奏（无参考谱面）**
你没有乐谱对照，不知道演奏者打算弹什么。你只能通过音频信号本身判断问题。

{feature_section}

## 🎸 起音间隔分析 — IOI 离群值检测（系统自动检测，供你参考）

{gap_section}

**IOI (Inter-Onset Interval) = 相邻音符起音之间的时间间隔**

**解读规则**：
- 吉他演奏中音符高度重叠（延音），所以不能靠「无声间隙」检测停顿。
  系统改用起音间隔 (IOI) 离群值——如果某个 IOI 远超典型值（>3倍中位数），说明这个位置的音符间距显著大于平均值
- IOI 中位数代表演奏者典型的音符间距，P90 代表偏慢但仍正常的间距
- **关键判断：IOI间隔拉长 ≠ 一定是技巧问题！** 请按以下三步判断：
  1. 先看 feature_summary 中的 **CV(核心)**——这是排除乐句呼吸等离群值后的真实节奏稳定性。吉他 CV(核心) 参考：<0.25 优秀，0.25-0.45 正常，>0.45 偏不稳。CV(核心)<0.45 表示演奏者基本节奏感没问题——此时的大间隔更可能是**音乐处理（乐句呼吸/段落切换）**而非技巧失误，不应扣分
  2. 看大间隔周围的相邻 IOI 是否接近中位数。如果停顿前后节奏控制精准，说明演奏者有能力均匀弹奏——这是刻意留白
  3. 只有同时满足以下条件才报 transition_gap：(a)CV(核心)>0.45（节奏本身不够稳）且 (b)大间隔处前后音符的把位/品位有明显变化（疑似换和弦卡顿）
- 如果无法确定是音乐处理还是技巧问题：confidence 0.4-0.55 并在 detail 中说明「不确定是音乐处理还是技巧停顿」
- 如果系统报告「未发现阈值以上的IOI间隔拉长」但你自己听出节奏不均匀：报 rhythm_fault
- 如果系统同时报告了节奏不稳（CV值偏高）和 IOI 间隔拉长：两者都报告，transition_gap 标记局部停顿，rhythm_fault 标记不均匀

## ⚠️ 第一步：判断整体水平（必须！）
**先回答自己：这段演奏听起来是干净流畅的吗？**
- 如果答案是"干净流畅无明显瑕疵" → 整体给高分（rhythm_accuracy 80-92），errors 可以只报 0-1 个，也可以为空
- 如果答案是"基本流畅但个别地方不太对" → 只报真正听到的问题（0-2个），不要为了凑数编造问题
- 如果答案是"有明显问题" → 如实挑2-4个最明显的问题
- basic-pitch 转录时会产生微小抖动——揉弦看起来像 pitch 波动，rubato 看起来像节奏不均，滑音看起来像连续音符。**这些全都不算错误！不要报！**
- 真正的演奏瑕疵：可察觉的抢拍拖拍（节奏明显不均匀）、出现不该有的杂音/打品声、力度明显忽大忽小、明显的哑音/死音
- **确定性问题必须报**：transition_gap（换和弦停顿）已由系统确定性检测，你只需关注 rhythm_fault（节奏不均匀、抢拍拖拍）和 extra_note（杂音/打品）
- **关键判断**：一段演奏可以有 90 分的整体感，但中间某两三个音衔接不够流畅——这是真实的瑕疵，应该报 rhythm_fault (confidence 0.5-0.65)

**🎯 pitch_accuracy 评分规则（自由演奏模式 — 基于音质而非音准！）**
你没有参考乐谱，无法判断演奏者弹的音"对不对"。但你可以从音频信号中判断每个音的**发声质量**：
- **音符清晰度**：velocity 反映拨弦力度和音头清晰度。velocity<35 的音往往是哑音/死音（没弹响）。统计低力度音的比例。
- **音色干净度**：非谐波能量比高 → 杂音多；高频噪声多 → 可能有打品/刮擦。
- **力度一致性**：如果大部分音 velocity 50-80 但有几个音 velocity<25 → 那几个音没弹响。

**pitch_accuracy 评分标准（自由演奏）**：
- 95%+ 音符清晰有力（velocity>30），音色干净 → pitch 82-92
- 有少量弱音/哑音（5-15%），整体仍然清晰 → pitch 72-84
- 多处弱音/哑音（15-30%），或音色偏杂 → pitch 60-75
- 大量哑音/死音（>30%），或音色非常嘈杂 → pitch 40-62

**重要**：如果你看到 velocity P5 很低（<20），说明存在明显的哑音/死音——必须报 dead_note 错误并据此降低 pitch 分！

**🚨 无参考谱面约束（最重要！）**
- **禁止报告 wrong_note 和 missed_note**——你没有乐谱，无法判断演奏者「应该弹什么」
- **禁止根据转录音符猜测演奏者的意图**——比如不能猜「三弦五品应该换成三弦七品」
- 你可以报告的错误类型：
  - rhythm_fault: 音符节奏不均匀、抢拍、拖拍
  - transition_gap: **局部**的乐句间长停顿（>0.5s），其余部分节奏正常 → 说明换和弦/换把位不流畅
  - extra_note: 杂音/打品/刮擦声
  - dead_note: 明显没弹响的音/哑音/死音（velocity<35，attack_time>60ms，音头模糊）
- extra_note 只能用于明显的非演奏杂音（如手指擦弦的刮擦声、碰撞声），不能用于多弹的音
- dead_note 必须引用具体的时间位置（如"约3秒处有个音没弹响"），不要引用具体的几弦几品（guitar_position 是系统估算的，不保证准确）
- **判断 transition_gap vs rhythm_fault 的关键**：如果只有个别地方停顿长、其余节奏均匀 → transition_gap；如果整体忽快忽慢 → rhythm_fault

## 第二步：给分
- 专业/录音室级别的演奏：overall_score 88-98
- 熟练的业余演奏：overall_score 78-90
- 有明显问题但仍能听出演奏意图：overall_score 62-78
- 严重问题/几乎听不出旋律：overall_score 40-62

**新手演奏的善意分原则**：初学者能把旋律弹下来就应该鼓励。节奏大致稳定 → rhythm 应 >= 78。但 pitch 必须如实反映音符发声质量——有哑音就要扣分，不能因为是新手就给虚高的音质分。不要因为某些音之间的微小不均匀就大幅扣 rhythm 分——报出来让学生知道就行。

**🎸 吉他 IOI 变异系数 (CV) 特殊说明（重要！）：**
- 吉他独奏中，扫弦(极短IOI)和旋律音符(中等IOI)天然混合，CV天生就比其他乐器高
- **吉他 CV(核心) 参考范围**：<0.25 优秀，0.25-0.45 正常（扫弦+旋律的正常混合），0.45-0.55 偏不稳，>0.55 明显不稳
- CV(核心) 在 0.25-0.45 之间时，不应因 CV 值扣节奏分——这是吉他演奏的常态
- **评分时以这两个指标为准（优先级高于 CV）**：
  1. 过滤后的加速/拖拍段落数（已自动排除扫弦→旋律的正常交替）
  2. IOI 离群值的数量和局部上下文
- 如果过滤后的加速+拖拍总数少，且仅有1-2处 IOI 离群值 → 节奏是干净的，rhythm_accuracy >= 82

**transition_gap 与 rhythm_fault 的评分权重区别（重要！）：**
- transition_gap 是局部问题（换和弦时手指跟不上），不代表整体节奏感差
- rhythm_fault 是系统性问题（多处抢拍/拖拍/不均匀），说明节奏基本功需要加强
- **CV(核心) 与 rhythm_accuracy 的锚定（必须遵守！）**：
  - CV(核心) 标签为「良好/正常」→ rhythm_accuracy 必须在 82-92 之间
  - CV(核心) 标签为「偏不稳」→ rhythm_accuracy 必须在 75-82 之间
  - CV(核心) 标签为「不稳定」→ rhythm_accuracy 必须在 65-78 之间
- **评分规则**：
  - 只有1处 transition_gap，其余节奏干净 → rhythm_accuracy 82-90（不是70！）
  - 有2+处 rhythm_fault（抢拍拖拍） → rhythm_accuracy 68-82
  - 既有 transition_gap 又有 rhythm_fault → rhythm_accuracy 65-78
  - 一个 transition_gap 不应该把节奏分从90拉到70，这是局部停顿不是节奏崩了

**铁律：仔细听每一个音符之间的衔接。如实报告你听到的。如果演奏干净流畅，errors 可以为空，stats 给高分。不要为了找问题而编造问题。不确定的不要报。最多报5个问题。**
**置信度：每个error必须带confidence字段(0.0-1.0)，<0.5的不要报。**
**定量标准：相邻音差距超过半音(>1 semitone)且持续>0.15s才可能是错音。单个音的微小pitch波动(<0.5半音)是揉弦/颤音，不算错误。**
**节奏瑕疵：只有明确听到节奏不均匀、抢拍、拖拍时才报告。如果整体节奏感好，不要为了凑数报低置信度的节奏问题。**

**⚠️ 自检（输出 JSON 之前必须做！）：**
1. errors 应该驱动 scores，而非反过来。先判断有哪些问题，再根据问题给分。不要先打分再凑错误。
2. 如果你给了高分但 errors 不为空 → 检查是否误报了，把不确定的去掉
3. 如果你给了低分但 errors 为空 → 检查是否漏报了真正的问题
4. 不确定的问题不报，宁可漏报也不要误报

## 关于乐器技巧的重要提醒（你必须知道！）
乐器是：{instrument_label}

如果乐器是"吉他"，你必须认识以下吉他技巧，**不能把它们当成错误**：
- 滑音（slide）：音符连续滑动，midi音高会连续变化
- 击弦（hammer-on）：第二个音比第一个音高，没有重新拨弦
- 勾弦（pull-off）：第二个音比第一个音低，没有重新拨弦
- 扫弦（strum）：多个音符在50ms内同时出现——注意 note_text 中标记为「🎸 和弦块」的行是同时弹响的和弦，不要把它们当成逐个弹奏的分解和弦来分析
- 切音（chop）：音符很短促（dur<0.1s），有顿音感
- 泛音（harmonic）：音色特殊的清脆高音
- 揉弦（vibrato）：音高有轻微波动
- 只有真正的音高不对、节奏不稳、多弹了不该弹的音，才标记为错误

## ⚠️ 吉他品位描述要求（铁律！）
**系统无法精确定位弦和品位。不要在任何错误描述中写出「几弦几品」！**
- ✅ 用时间+大致区域："开头低把位附近有个音..."、"中段有个音节奏偏快"、"约14秒处有个音没弹响"
- ✅ 描述 transition_gap 时说清停顿前后的大致区域即可："低把位换到中把位时停顿了0.8s"
- ❌ 严禁写具体品位："三弦空弦"、"二弦一品"、"五弦三品"等具体弦+品组合一律禁止
- ❌ 编造数据中没有的音："三弦五品应该改成三弦八品"（你不知道他应该弹什么！）
- detail 字段：用模糊但有用的语言描述。如："中高把位附近有个音偏高了约半音"、"低把位部分节奏不均匀，像在往前赶"

## 位置描述要求（非常重要！）
描述错误位置时，**绝对不要只说"16.7秒处"**，要说：
- "开头部分"、"中段"、"接近结尾"
- 或者大致描述："你弹的第三个音"、"副歌开始前"
- 也可以用秒数辅助，但必须加上文字描述："开头几秒（约第3秒）..."

## 输出格式（严格 JSON）
```json
{{
  "errors": [
    {{
      "type": "rhythm_fault | transition_gap | extra_note",
      "time": 错误时间（秒）,
      "detail": "口语描述。节奏错误必须说清楚：哪里不稳、偏快还是偏慢、是否不在拍子上（如'中段节奏偏快，像在往前赶''开头部分几个音拖拍了，比正常节拍慢了半拍'）。杂音错误描述杂音来源和位置。",
      "severity": "high | medium | low",
      "confidence": 0.85
    }}
  ],
  "stats": {{
    "pitch_accuracy": 音准得分(0-100),
    "rhythm_accuracy": 节奏得分(0-100),
    "overall_score": 综合得分(0-100)
  }}
}}
```

## 注意事项
- rhythm_fault：节奏不对（抢拍、拖拍、不均匀）
- transition_gap：乐句间的**局部**长停顿（>0.5s），其余部分节奏正常 → 换和弦/换把位不流畅，detail要描述停顿前后的品位变化
- extra_note：非演奏产生的杂音（手指擦弦的刮擦声、碰撞声），不能用于多弹的音
- **仔细听每一个音符**：只要听到任何不流畅、不均匀、有瑕疵的地方，都要标记。宁愿多报一个低置信度的瑕疵(confidence 0.5-0.6)，也不要漏掉真实的问题。好的演奏通常是 85 分以上，非常好的 90 分以上。
- **无参考谱面时，pitch_accuracy 只能基于整体音色判断，不要高置信度地打低分**
- 如果是吉他，注意区分"技巧"和"错误"
- **detail 必须包含品位描述！**
- **严禁在任何输出中提及音名+八度（如G3、A#3、D4、E2等），禁止出现「在低把位和弦（G3+A#3）之后」这类表述。遇到需要描述音高时，用自然语言（如「第3品」「空弦音」「根音」「五度音」）替代。**

演奏数据：
{note_text}
"""

        try:
            response = self.ds_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=800,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content

            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                parsed = json.loads(json_match.group(0))
                errors = parsed.get("errors", [])
                stats = parsed.get("stats", default_stats)
                # 置信度过滤 + 最多3个错误
                # 0.55 阈值：允许 AI 以低置信度标记的疑似问题通过，避免漏报
                errors = [e for e in errors
                          if e.get("confidence", 0.0) >= 0.55][:3]
                # 归一化：LLM可能输出0-10分制
                for sk in ['pitch_accuracy', 'rhythm_accuracy', 'overall_score']:
                    val = stats.get(sk)
                    if val is not None and isinstance(val, (int, float)) and 0 < val < 10:
                        stats[sk] = val * 10
                # 无错误 + 信号指标优秀 → 适当提升总分（但不过度，避免与有瑕疵视频拉开过大差距）
                if len(errors) == 0:
                    pitch_ok = stats.get("pitch_accuracy", 90) >= 88
                    rhythm_ok = stats.get("rhythm_accuracy", 90) >= 88
                    if pitch_ok and rhythm_ok:
                        stats["overall_score"] = max(stats.get("overall_score", 85), 85)
                return errors, stats

        except Exception as e:
            logger.warning(f"DeepSeek 分析失败: {e}")

        return default_errors, default_stats

    # ==================== 规则备选（无 DeepSeek 时用） ====================

    def _rule_based_errors(self, notes: List[Dict], instrument: str) -> Tuple[List[Dict], List[float]]:
        """本地规则引擎分析错误（备选方案）"""
        errors = []
        timestamps = []

        if len(notes) < 3:
            return [], []

        # 节奏分析
        intervals = []
        for i in range(1, len(notes)):
            gap = notes[i]["start_time"] - notes[i-1]["end_time"]
            intervals.append(gap)

        if intervals:
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)

            for i, gap in enumerate(intervals):
                idx = i + 1
                if gap < -0.05:
                    errors.append({
                        "type": "overlap",
                        "time": notes[idx]["start_time"],
                        "detail": f"约{notes[idx-1]['start_time']:.1f}s和{notes[idx]['start_time']:.1f}s两个音重叠",
                        "severity": "medium",
                    })
                    timestamps.append(notes[idx]["start_time"])
                elif gap > mean_interval + 2 * std_interval and gap > 0.3:
                    t_prev = notes[idx-1].get('start_time', 0)
                    t_next = notes[idx].get('start_time', 0)
                    errors.append({
                        "type": "transition_gap",
                        "time": notes[idx]["start_time"],
                        "detail": f"约{t_prev:.1f}s到约{t_next:.1f}s之间停顿{gap:.2f}s，换和弦或换把位时手指跟不上",
                        "severity": "medium",
                    })
                    timestamps.append(notes[idx]["start_time"])

        # 音域检查
        pitches = [n["pitch"] for n in notes]
        if instrument == "piano":
            if max(pitches) > 108 or min(pitches) < 21:
                errors.append({
                    "type": "out_of_range",
                    "time": notes[0]["start_time"],
                    "detail": "超出钢琴标准音域",
                    "severity": "info",
                })

        return errors, timestamps

    def _detect_dead_notes(self, notes: List[Dict], audio_features) -> List[Dict]:
        """检测闷音/哑音/死音：扫描时间窗口内的低力度音符集中区域。

        当某段时间内多个音符 velocity 偏低（<25），说明演奏者手指没按实，
        产生闷音/断音。返回 dead_note 错误条目。
        """
        errors = []
        if len(notes) < 4:
            return errors

        # 找出低力度音符（velocity<35 视为可能闷音/哑音，排除幻音）
        low_vel_notes = [n for n in notes if 0 < n.get("velocity", 0) < 35 and not n.get("_phantom")]
        if not low_vel_notes:
            return errors

        # 按时间聚类：相邻低力度音符间隔<1.5s归为一组
        low_vel_notes.sort(key=lambda n: n.get("start_time", 0))
        clusters = []
        current = [low_vel_notes[0]]
        for n in low_vel_notes[1:]:
            if n.get("start_time", 0) - current[-1].get("start_time", 0) < 1.5:
                current.append(n)
            else:
                clusters.append(current)
                current = [n]
        clusters.append(current)

        # 计算整体低力度音符占比，用于判断严重程度
        total_notes = len(notes)
        low_ratio = len(low_vel_notes) / total_notes

        for cluster in clusters:
            if len(cluster) < 2:
                continue
            start_time = cluster[0].get("start_time", 0)
            end_time = cluster[-1].get("start_time", 0)
            avg_vel = sum(n.get("velocity", 0) for n in cluster) / len(cluster)
            positions = [n.get("guitar_position", "未知") for n in cluster[:5]]

            # 严重程度判断：avg_vel越低越严重，连续音符越多越严重
            if avg_vel < 12 or len(cluster) >= 5:
                severity = "high"
            elif avg_vel < 22 or len(cluster) >= 3:
                severity = "medium"
            else:
                severity = "low"

            errors.append({
                "type": "dead_note",
                "time": round(start_time, 1),
                "end_time": round(end_time, 1),
                "detail": (
                    f"闷音/哑音：约{round(start_time,1)}-{round(end_time,1)}s处{len(cluster)}个音符没弹响"
                    f"（平均力度{avg_vel:.0f}）。手指没按实或换和弦时没跟上。"
                ),
                "severity": severity,
                "confidence": 0.82,
            })

        logger.info(
            f"闷音检测: {len(low_vel_notes)}/{total_notes} 低力度音符 ({low_ratio:.1%}), "
            f"聚类为 {len(errors)} 个闷音区域, velocity P5={audio_features.velocity_p5:.0f}"
        )
        return errors

    def _calc_pitch_accuracy(self, errors: List[Dict]) -> float:
        """自由演奏模式下规则引擎无法精确判断音准。
        回退时返回中性偏高值 88，不给虚假低分。"""
        return 88.0

    def _calc_rhythm_accuracy(self, errors: List[Dict]) -> float:
        rhythm_weighted = sum(
            3 if e.get("severity") == "high" else 2 if e.get("severity") == "medium" else 1
            for e in errors if e["type"] in ("overlap", "pause", "rhythm_fault", "transition_gap")
        )
        total_weighted = max(sum(
            3 if e.get("severity") == "high" else 2 if e.get("severity") == "medium" else 1
            for e in errors
        ), 1)
        # 按严重程度加权，最低 50（规则引擎的节奏判断精度有限）
        return round(min(100, max(65, 100 - (rhythm_weighted / total_weighted * 30))), 1)

    def _calc_overall_score(self, errors: List[Dict], total_notes: int) -> float:
        if total_notes == 0:
            return 0
        penalty = sum(4 if e["severity"] == "high" else 2 if e["severity"] == "medium" else 1 for e in errors)
        # 最低 50（规则引擎回退时的保守估计）
        return round(min(100, max(65, 100 - (penalty * 100 / total_notes))), 1)



