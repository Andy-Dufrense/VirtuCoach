"""
VirtuCoach - 和弦手型检查模块
逐弦音频清晰度分析 + 和弦知识库匹配 + 手型检测
"""

import os
import json
import struct
import subprocess
import tempfile
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class StringResult:
    string_num: int       # 1-6 (①弦最细=1, ⑥弦最粗=6)
    string_name: str      # "⑥弦" "⑤弦" etc
    note: str             # 期望音名
    onset_time: float     # 起音时间(秒)
    rms_db: float         # 响度(RMS转dB)
    spectral_centroid: float  # 频谱重心(Hz)
    clarity_score: float  # 0-1 清晰度评分
    decay_time: float     # 衰减到-20dB的时间(秒)
    has_signal: bool      # 是否有可检测信号
    ok: bool              # 综合判定
    flatness: float = 0.0  # 频谱平坦度（>0.30 = 闷音/未按实）


@dataclass
class ChordAudioResult:
    string_results: List[Dict]
    onset_count: int
    signal_present: bool
    noise_floor_db: float
    overall_clarity: float
    audio_quality: str = "unknown"  # "good" / "weak" / "poor"
    peak_rms_db: float = -60.0
    snr_db: float = 0.0


class ChordAnalyzer:
    """和弦手型检查分析器：逐弦音频检测 + 和弦知识检索"""

    # 标准调弦下的弦→音名映射（开放弦）
    STRING_NAMES = {
        6: ("⑥弦", "E2"),
        5: ("⑤弦", "A2"),
        4: ("④弦", "D3"),
        3: ("③弦", "G3"),
        2: ("②弦", "B3"),
        1: ("①弦", "E4"),
    }

    # 吉他开放弦标准频率 (Hz) — 用于基本音高检测
    OPEN_STRING_HZ = {
        6: 82.41,   # E2
        5: 110.00,  # A2
        4: 146.83,  # D3
        3: 196.00,  # G3
        2: 246.94,  # B3
        1: 329.63,  # E4
    }

    # 半音频率比
    SEMITONE_RATIO = 2 ** (1/12)

    def __init__(self):
        pass

    def extract_audio(self, video_path: str) -> Optional[str]:
        """从视频提取音频为 WAV（16kHz mono）"""
        try:
            from config import FFMPEG_PATH
            wav_path = video_path + ".chord.wav"
            cmd = [
                FFMPEG_PATH, "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                wav_path
            ]
            subprocess.run(cmd, capture_output=True, timeout=30)
            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000:
                return wav_path
            return None
        except Exception as e:
            logger.warning(f"音频提取失败: {e}")
            return None

    def analyze_per_string(self, video_path: str, chord_data: dict) -> ChordAudioResult:
        """逐弦分析音频清晰度

        Args:
            video_path: 和弦检查视频
            chord_data: 和弦知识库数据 (含 strings 字段如 [0, 2, 2, 0, 0, 0])
        """
        wav_path = self.extract_audio(video_path)
        if not wav_path:
            return ChordAudioResult(
                string_results=[],
                onset_count=0,
                signal_present=False,
                noise_floor_db=-60.0,
                overall_clarity=0.0,
            )

        try:
            # 读取音频
            with open(wav_path, "rb") as f:
                # 跳过 WAV 头
                f.seek(44)
                raw = f.read()

            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            sr = 16000

            # === 音频质量前置评估 ===
            # 计算噪声基底（取前100ms作为静音参考）
            noise_samples = samples[:min(1600, len(samples))]
            noise_rms = np.sqrt(np.mean(noise_samples ** 2)) if len(noise_samples) > 0 else 1e-6
            noise_floor_db = 20 * np.log10(max(noise_rms, 1e-10))

            # 计算整体信号峰值（取 RMS 最高的 200ms 窗口）
            hop = 256
            if len(samples) >= hop * 2:
                num_frames = (len(samples) - hop) // hop
                frame_rms = np.zeros(num_frames)
                for i in range(num_frames):
                    frame = samples[i * hop:(i + 1) * hop]
                    frame_rms[i] = np.sqrt(np.mean(frame ** 2))
                # 取前 5 帧的平均作为静音参考，其余帧的 90 分位数作为信号峰值
                silence_rms = np.mean(frame_rms[:min(5, num_frames)])
                signal_rms = np.percentile(frame_rms, 90)
                peak_rms_db = 20 * np.log10(max(signal_rms, 1e-10))
                silence_db = 20 * np.log10(max(silence_rms, 1e-10))
                snr_db = peak_rms_db - silence_db
            else:
                peak_rms_db = noise_floor_db
                snr_db = 0.0

            # 判定音频质量等级
            if peak_rms_db > -22 and snr_db > 12:
                audio_quality = "good"
            elif peak_rms_db > -32 and snr_db > 6:
                audio_quality = "weak"
            else:
                audio_quality = "poor"

            logger.info(f"Audio quality: {audio_quality} (peak={peak_rms_db:.1f}dBFS, SNR={snr_db:.1f}dB)")

            # 质量太差 → 跳过逐弦分析，全部标记为不确定
            if audio_quality == "poor":
                try:
                    os.remove(wav_path)
                except:
                    pass
                expected_strings = chord_data.get("strings", [0, 0, 0, 0, 0, 0])
                uncertain_results = []
                for string_num in [6, 5, 4, 3, 2, 1]:
                    s_name, _open_note = self.STRING_NAMES[string_num]
                    expected_hz = self.OPEN_STRING_HZ[string_num]
                    fret_raw = expected_strings[6 - string_num] if (6 - string_num) < len(expected_strings) else 0
                    try:
                        fret = int(fret_raw)
                    except (ValueError, TypeError):
                        fret = 0
                    note_name = self._freq_to_note(expected_hz * (self.SEMITONE_RATIO ** fret))
                    uncertain_results.append({
                        "string_num": string_num, "string_name": s_name,
                        "note": note_name, "onset_time": 0, "rms_db": -60,
                        "spectral_centroid": 0, "clarity_score": 0,
                        "decay_time": 0, "has_signal": False, "ok": False,
                        "status_text": "信号不足以判断",
                    })
                return ChordAudioResult(
                    string_results=uncertain_results,
                    onset_count=0, signal_present=False,
                    noise_floor_db=round(noise_floor_db, 1),
                    overall_clarity=0.0,
                    audio_quality="poor",
                    peak_rms_db=round(peak_rms_db, 1),
                    snr_db=round(snr_db, 1),
                )

            # 起音检测（能量包络 + 峰值检测）
            # 根据音频质量调整检测阈值：弱信号需要更低阈值才能捕捉到拨弦瞬态
            if audio_quality == "good":
                onset_threshold = 10.0
                retry_threshold = 8.0
            elif audio_quality == "weak":
                onset_threshold = 8.0
                retry_threshold = 6.0
            else:
                onset_threshold = 12.0
                retry_threshold = None

            onsets = self._detect_onsets(samples, sr, noise_floor_db, threshold_db=onset_threshold)
            if len(onsets) < 2 and retry_threshold is not None:
                onsets = self._detect_onsets(samples, sr, noise_floor_db, threshold_db=retry_threshold)

            # 期望拨弦数（非空弦的弦数 + 空弦也应弹响来验证）
            expected_strings = chord_data.get("strings", [0, 0, 0, 0, 0, 0])
            # strings 从⑥到①: 0=开放弦, >0=品位
            active_strings = list(range(6, 0, -1))  # 所有6根弦都应弹

            # 将检测到的起音分配到各弦
            string_results = self._assign_onsets_to_strings(
                samples, sr, onsets, active_strings, expected_strings, noise_floor_db
            )

            # 清理临时文件
            try:
                os.remove(wav_path)
            except:
                pass

            overall_clarity = (
                np.mean([s.clarity_score for s in string_results])
                if string_results else 0.0
            )

            # weak 质量时降低所有结果的置信度标记
            string_dicts = [self._sr_to_dict(s) for s in string_results]
            if audio_quality == "weak":
                for sd in string_dicts:
                    if sd["status_text"] == "清晰 ✓":
                        sd["status_text"] = "清晰(信号偏弱)"
                    elif sd["status_text"] == "信号弱":
                        sd["status_text"] = "信号弱(录音偏轻)"

            return ChordAudioResult(
                string_results=string_dicts,
                onset_count=len(onsets),
                signal_present=len(onsets) >= 1,
                noise_floor_db=round(noise_floor_db, 1),
                overall_clarity=round(overall_clarity, 2),
                audio_quality=audio_quality,
                peak_rms_db=round(peak_rms_db, 1),
                snr_db=round(snr_db, 1),
            )

        except Exception as e:
            logger.warning(f"分析失败: {e}")
            import traceback
            traceback.print_exc()
            return ChordAudioResult(
                string_results=[],
                onset_count=0,
                signal_present=False,
                noise_floor_db=-60.0,
                overall_clarity=0.0,
                audio_quality="poor",
                peak_rms_db=-60.0,
                snr_db=0.0,
            )

    def _detect_onsets(self, samples: np.ndarray, sr: int, noise_floor_db: float,
                       threshold_db: float = 10.0) -> List[float]:
        """检测音频起音时间点（能量上升沿）

        Args:
            threshold_db: 高于噪声基底的阈值(dB)
        Returns:
            起音时间列表(秒)
        """
        hop = 256  # ~16ms at 16kHz
        if len(samples) < hop * 2:
            return []

        # 计算短时能量
        num_frames = (len(samples) - hop) // hop
        if num_frames < 2:
            return []

        energy = np.zeros(num_frames)
        for i in range(num_frames):
            frame = samples[i * hop:(i + 1) * hop]
            energy[i] = np.sqrt(np.mean(frame ** 2))

        energy_db = 20 * np.log10(np.maximum(energy, 1e-10))

        # 检测能量上升沿
        threshold = noise_floor_db + threshold_db
        onsets = []
        min_interval = int(0.3 * sr / hop)  # 最小间隔300ms（防止同一拨弦多次触发）

        for i in range(1, num_frames):
            if energy_db[i] > threshold and energy_db[i - 1] <= threshold:
                # 与前一个起音至少间隔 min_interval 帧
                onset_time = i * hop / sr
                if not onsets or (onset_time - onsets[-1]) > 0.3:
                    onsets.append(onset_time)

        return onsets

    def _assign_onsets_to_strings(
        self, samples: np.ndarray, sr: int,
        onsets: List[float],
        active_strings: List[int],
        expected_frets: List[int],
        noise_floor_db: float,
    ) -> List[StringResult]:
        """将检测到的起音按音高匹配分配到各弦。

        用 FFT 检测每个起音段的主频，与 6 根弦的期望频率做最近匹配，
        不再依赖顺序假设。同一根弦有多个起音时取信噪比最高的。
        """

        # Build string → expected frequency map
        expected_freqs = {}  # string_num → (expected_hz, note_name)
        for string_num in [6, 5, 4, 3, 2, 1]:
            fret_raw = expected_frets[6 - string_num] if (6 - string_num) < len(expected_frets) else 0
            try:
                fret = int(fret_raw)
            except (ValueError, TypeError):
                fret = 0
            expected_hz = self.OPEN_STRING_HZ[string_num] * (self.SEMITONE_RATIO ** fret)
            note_name = self._freq_to_note(expected_hz)
            expected_freqs[string_num] = (expected_hz, note_name)

        # For each onset, find the dominant frequency and match to closest string
        string_best = {}  # string_num → (onset_t, rms_db, clarity, decay_t, spec_cent, flatness)
        used_onsets = set()

        for onset_t in onsets:
            start_sample = int(onset_t * sr)
            end_sample = min(start_sample + int(0.8 * sr), len(samples))
            segment = samples[start_sample:end_sample]
            if len(segment) < 100:
                continue

            # Find dominant frequency in this onset segment
            dom_freq = self._find_dominant_freq(segment, sr)
            if dom_freq is None:
                continue

            # Match to closest expected string pitch (within 2 semitones)
            best_string = None
            best_dist = float('inf')
            for string_num, (expected_hz, _note_name) in expected_freqs.items():
                if expected_hz <= 0:
                    continue
                # Log distance in semitones
                dist = abs(np.log2(dom_freq / expected_hz)) * 12.0
                if dist < best_dist and dist < 2.0:  # within 2 semitones
                    best_dist = dist
                    best_string = string_num

            if best_string is None:
                continue

            expected_hz, note_name = expected_freqs[best_string]

            # Analyze segment quality against the matched string's expected pitch
            rms_db, clarity, decay_t, spec_cent, flatness = self._analyze_segment(
                segment, sr, noise_floor_db, expected_hz
            )

            # If this string already has a better onset, keep the better one
            if best_string in string_best:
                existing_clarity = string_best[best_string][2]
                if clarity <= existing_clarity:
                    continue

            string_best[best_string] = (onset_t, rms_db, clarity, decay_t, spec_cent, flatness)
            used_onsets.add(onset_t)

        # Build results for all 6 strings (⑥→①)
        results = []
        strings_order = [6, 5, 4, 3, 2, 1]
        for string_num in strings_order:
            s_name, _open_note = self.STRING_NAMES[string_num]
            _expected_hz, note_name = expected_freqs[string_num]

            if string_num in string_best:
                onset_t, rms_db, clarity, decay_t, spec_cent, flatness = string_best[string_num]
                results.append(StringResult(
                    string_num=string_num,
                    string_name=s_name,
                    note=note_name,
                    onset_time=round(onset_t, 2),
                    rms_db=round(rms_db, 1),
                    spectral_centroid=round(spec_cent, 0),
                    clarity_score=round(clarity, 2),
                    decay_time=round(decay_t, 2),
                    has_signal=True,
                    ok=bool(clarity >= 0.5),
                    flatness=round(flatness, 3),
                ))
            else:
                # No onset matched to this string — check if there's energy at the expected freq anyway
                # (sustained notes might not have a clear onset but still have audible energy)
                mid_sample = len(samples) // 2
                start_s = max(0, mid_sample - int(0.4 * sr))
                end_s = min(len(samples), mid_sample + int(0.4 * sr))
                mid_segment = samples[start_s:end_s]
                if len(mid_segment) > 100:
                    _r, clarity, _d, _s, _f = self._analyze_segment(
                        mid_segment, sr, noise_floor_db, _expected_hz
                    )
                    if clarity >= 0.3:
                        results.append(StringResult(
                            string_num=string_num, string_name=s_name,
                            note=note_name, onset_time=0,
                            rms_db=round(_r, 1), spectral_centroid=round(_s, 0),
                            clarity_score=round(clarity, 2), decay_time=round(_d, 2),
                            has_signal=True, ok=bool(clarity >= 0.5),
                            flatness=round(_f, 3),
                        ))
                        continue

                results.append(StringResult(
                    string_num=string_num, string_name=s_name,
                    note="", onset_time=0, rms_db=-60,
                    spectral_centroid=0, clarity_score=0,
                    decay_time=0, has_signal=False, ok=False,
                ))

        return results

    def _find_dominant_freq(self, segment: np.ndarray, sr: int) -> Optional[float]:
        """Find the dominant pitch frequency in an audio segment using HPS (Harmonic Product Spectrum)."""
        n_fft = min(2048, len(segment))
        if n_fft < 64:
            return None
        window = np.hanning(n_fft)
        spec = np.abs(np.fft.rfft(segment[:n_fft] * window))
        freqs = np.fft.rfftfreq(n_fft, 1 / sr)

        # Constrain to guitar range: 80Hz (E2) to 660Hz (E5)
        lo_idx = np.searchsorted(freqs, 75)
        hi_idx = np.searchsorted(freqs, 700)

        if hi_idx - lo_idx < 3:
            return None

        # Harmonic Product Spectrum: multiply downsampled spectra to find fundamental
        hps = spec[lo_idx:hi_idx].copy()
        for h in [2, 3]:
            downsampled = spec[lo_idx * h:hi_idx * h:h][:len(hps)]
            hps[:len(downsampled)] *= downsampled

        if len(hps) == 0 or np.max(hps) <= 0:
            return None

        peak_idx = np.argmax(hps)
        return float(freqs[lo_idx + peak_idx])

    @staticmethod
    def _spectral_flatness(seg: np.ndarray) -> float:
        """频谱平坦度 = 几何平均/算术平均 of FFT（0=纯音谐波，1=白噪声/闷音）。

        与 audio_analyzer._spectral_flatness 同一算法，阈值 flat>0.30=闷音。
        """
        if len(seg) < 128:
            return -1.0
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        spec = spec[spec > 1e-10]
        if len(spec) == 0:
            return 1.0
        return float(np.exp(np.mean(np.log(spec))) / np.mean(spec))

    def _analyze_segment(
        self, segment: np.ndarray, sr: int,
        noise_floor_db: float, expected_hz: float,
    ) -> Tuple[float, float, float, float, float]:
        """分析单段音频：RMS、清晰度、衰减时间、频谱重心、频谱平坦度

        Returns: (rms_db, clarity_score, decay_time, spectral_centroid, flatness)
        """
        # RMS 响度
        rms = np.sqrt(np.mean(segment ** 2))
        rms_db = 20 * np.log10(max(rms, 1e-10))

        # 频谱分析
        n_fft = min(1024, len(segment))
        if n_fft < 64:
            return rms_db, 0.0, 0.0, 0.0, 0.0

        # 加窗
        window = np.hanning(n_fft)
        spec = np.abs(np.fft.rfft(segment[:n_fft] * window))
        freqs = np.fft.rfftfreq(n_fft, 1 / sr)

        # 频谱重心
        if np.sum(spec) > 0:
            spectral_centroid = np.sum(freqs * spec) / np.sum(spec)
        else:
            spectral_centroid = 0.0

        # 信号清晰度：检查基频能量集中度
        # 找到基频附近的能量占比
        half_step_hz = expected_hz * (self.SEMITONE_RATIO - 1)  # 半音宽度
        fund_lo = expected_hz - half_step_hz * 0.5
        fund_hi = expected_hz + half_step_hz * 0.5

        fund_energy = 0.0
        total_energy = np.sum(spec ** 2)
        for i, f in enumerate(freqs):
            if fund_lo <= f <= fund_hi:
                fund_energy += spec[i] ** 2
            # 也检查泛音（2倍、3倍频）
            for harmonic in [2, 3, 4]:
                if abs(f - expected_hz * harmonic) < half_step_hz * harmonic * 0.3:
                    fund_energy += spec[i] ** 2 * 0.5  # 泛音权重减半

        harmonic_ratio = fund_energy / max(total_energy, 1e-10)

        # 衰减时间：RMS包络从峰值降到 -20dB 的时间
        env_hop = 128
        env_len = (len(segment) - env_hop) // env_hop
        if env_len > 3:
            rms_env = np.zeros(env_len)
            for i in range(env_len):
                frm = segment[i * env_hop:(i + 1) * env_hop]
                rms_env[i] = np.sqrt(np.mean(frm ** 2))
            rms_env_db = 20 * np.log10(np.maximum(rms_env, 1e-10))
            peak_db = np.max(rms_env_db)
            peak_idx = np.argmax(rms_env_db)

            # 找到衰减到 peak-20dB 的帧
            decay_idx = peak_idx
            target_db = peak_db - 20
            for i in range(peak_idx, len(rms_env_db)):
                if rms_env_db[i] <= target_db:
                    decay_idx = i
                    break
            decay_time = (decay_idx - peak_idx) * env_hop / sr
        else:
            decay_time = 0.0

        # 综合清晰度评分
        # 1. 响度分 (够响就行, >-25dBFS 满分)
        loudness_score = min(1.0, max(0.0, (rms_db + 40) / 15))

        # 2. 谐波集中度分
        harmonic_score = min(1.0, harmonic_ratio * 3)

        # 3. 衰减分 (自然衰减 0.3-1.5s 正常, 太快说明闷住了)
        if decay_time < 0.1:
            decay_score = 0.2  # 太短，弦被闷住了
        elif decay_time < 0.3:
            decay_score = 0.6
        elif decay_time < 2.0:
            decay_score = 1.0
        else:
            decay_score = 0.9

        clarity = 0.4 * loudness_score + 0.35 * harmonic_score + 0.25 * decay_score
        clarity = max(0.0, min(1.0, clarity))

        # 4. 频谱平坦度（闷音/未按实）：在能量峰值附近取 ~100ms 短窗计算，
        # 避免整段 0.8s 的衰减尾噪声抬高 flatness。
        if env_len > 3:
            peak_sample = int(peak_idx * env_hop)
            flat_win = segment[peak_sample:peak_sample + int(0.1 * sr)]
        else:
            flat_win = segment[:int(0.1 * sr)]
        flatness = self._spectral_flatness(flat_win) if len(flat_win) >= 128 else -1.0

        # 闷音惩罚：flatness > 0.30（与视频分析同一阈值）→ 手指没按实，弦闷住。
        # 只压明确闷音，不动 0.22-0.30 的模糊带（干净音上的换弦/碰弦假阳性区）。
        if flatness > 0.30:
            clarity = min(clarity, 0.45)

        # 响度安全网：信号明显可闻时，即使谐波集中度异常也不应判为闷音
        # 初学者可能按弦力度大但手指略微塌陷闷住邻弦——弦本身是响的，不应标记为"未弹响"。
        # 但响且闷（flatness 高）的音确实是闷音，安全网不能救回，故加 flatness<=0.30 门槛。
        if loudness_score > 0.60 and clarity < 0.50 and flatness <= 0.30:
            clarity = 0.50

        return rms_db, clarity, decay_time, spectral_centroid, flatness

    def _freq_to_note(self, freq: float) -> str:
        """频率转音名"""
        if freq < 20:
            return "?"
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        midi = 69 + 12 * np.log2(freq / 440.0)
        midi_int = int(round(midi))
        if midi_int < 0 or midi_int > 127:
            return "?"
        octave = (midi_int // 12) - 1
        return f"{names[midi_int % 12]}{octave}"

    @staticmethod
    def _sr_to_dict(sr: StringResult) -> dict:
        muted = (not sr.ok) and bool(sr.has_signal) and (sr.flatness > 0.30)
        if sr.ok:
            status_text = "清晰 ✓"
        elif muted:
            status_text = "闷音（未按实）"
        elif sr.has_signal:
            status_text = "信号弱"
        else:
            status_text = "未检测到拨弦"
        return {
            "string_num": int(sr.string_num),
            "string_name": str(sr.string_name),
            "note": str(sr.note),
            "onset_time": float(sr.onset_time) if sr.onset_time else 0.0,
            "rms_db": float(sr.rms_db) if sr.rms_db is not None else -60.0,
            "spectral_centroid": float(sr.spectral_centroid) if sr.spectral_centroid is not None else 0.0,
            "clarity_score": float(sr.clarity_score) if sr.clarity_score is not None else 0.0,
            "decay_time": float(sr.decay_time) if sr.decay_time is not None else 0.0,
            "flatness": float(sr.flatness) if sr.flatness is not None else 0.0,
            "has_signal": bool(sr.has_signal),
            "ok": bool(sr.ok),
            "status_text": status_text,
        }

    # ====== 和弦知识库匹配 ======

    @staticmethod
    def _parse_inline_dict(text: str) -> dict:
        """Parse {key: val, key2: "val2", key3: 123} inline dict format."""
        result = {}
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            text = text[1:-1]
        # Split by comma, respecting quoted strings
        parts = []
        current = ""
        in_quotes = False
        quote_char = None
        for ch in text:
            if ch in ('"', "'") and not in_quotes:
                in_quotes = True
                quote_char = ch
                current += ch
            elif ch == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
                current += ch
            elif ch == ',' and not in_quotes:
                parts.append(current.strip())
                current = ""
            else:
                current += ch
        if current.strip():
            parts.append(current.strip())

        for part in parts:
            if ':' in part:
                k, v = part.split(':', 1)
                k = k.strip().strip('"').strip("'")
                v = v.strip()
                # Remove surrounding quotes
                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    v = v[1:-1]
                elif v.replace('.', '').replace('-', '').isdigit():
                    v = float(v) if '.' in v else int(v)
                result[k] = v
        return result

    @staticmethod
    def _parse_frontmatter(fm_text: str) -> dict:
        """Parse YAML-like frontmatter with one level of nesting."""
        fm = {}
        lines = fm_text.strip().split("\n")
        parent_key = None
        parent_dict = {}

        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue

            # Check indentation
            stripped = line.strip()
            if not stripped or not (':' in stripped):
                continue

            is_indented = line.startswith("  ") or line.startswith("\t")

            if is_indented and parent_key:
                # This is a nested entry under parent_key
                k, v = stripped.split(':', 1)
                k = k.strip()
                v = v.strip()
                # Parse inline dict or simple value
                if v.startswith("{") and v.endswith("}"):
                    parent_dict[k] = ChordAnalyzer._parse_inline_dict(v)
                else:
                    parent_dict[k] = v
            else:
                # End of nested block: save parent
                if parent_key and parent_dict:
                    fm[parent_key] = parent_dict
                    parent_dict = {}
                    parent_key = None

                # New top-level key
                k, v = stripped.split(':', 1)
                k = k.strip()
                v = v.strip()

                if v == "":
                    # Start of a nested block
                    parent_key = k
                    parent_dict = {}
                elif v.startswith("[") and v.endswith("]"):
                    # List value
                    inner = v[1:-1]
                    fm[k] = [x.strip().strip('"').strip("'") for x in inner.split(",") if x.strip()]
                elif v.startswith("{") and v.endswith("}"):
                    # Inline dict at top level (e.g., thumb_position)
                    fm[k] = ChordAnalyzer._parse_inline_dict(v)
                elif v.isdigit():
                    fm[k] = int(v)
                else:
                    fm[k] = v

        # Don't forget last parent
        if parent_key and parent_dict:
            fm[parent_key] = parent_dict

        return fm

    @staticmethod
    def load_chord_knowledge(chord_id: str) -> Optional[dict]:
        """从 knowledge/chords/ 加载和弦数据"""
        knowledge_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "knowledge", "chords"
        )
        # 尝试多种文件名匹配
        candidates = [
            f"{chord_id}.md",
            f"chord-{chord_id}.md",
            f"{chord_id.replace('-', ' ')}.md",
            f"{chord_id}-major.md",
            f"{chord_id}-minor.md",
        ]
        for fname in candidates:
            fpath = os.path.join(knowledge_dir, fname)
            if os.path.exists(fpath):
                break
        else:
            # 遍历搜索：用词边界匹配避免 "F" 误匹配 "D-over-Fsharp"
            import re
            found = None
            # 第一轮：文件名以 chord_id 开头或以 -chord_id 接续
            for fname in sorted(os.listdir(knowledge_dir)):
                if not fname.endswith(".md"):
                    continue
                base = fname[:-3]  # remove .md
                # 分词匹配：chord_id 必须作为完整单词出现（不被字母包围）
                if re.search(rf'(?<![a-zA-Z]){re.escape(chord_id)}(?![a-zA-Z])', base, re.IGNORECASE):
                    found = fname
                    break
            if not found:
                # 第二轮：退化为子串匹配（兼容旧行为）
                for fname in sorted(os.listdir(knowledge_dir)):
                    if not fname.endswith(".md"):
                        continue
                    if chord_id.lower() in fname.lower():
                        found = fname
                        break
            if not found:
                # 在文件内容中搜索 chord_name 显示名
                for fname in sorted(os.listdir(knowledge_dir)):
                    if not fname.endswith(".md"):
                        continue
                    try:
                        fpath_test = os.path.join(knowledge_dir, fname)
                        with open(fpath_test, "r", encoding="utf-8") as ft:
                            content_test = ft.read()
                        if content_test.startswith("---"):
                            parts = content_test.split("---", 2)
                            if len(parts) >= 3:
                                for line in parts[1].strip().split("\n"):
                                    if line.startswith("chord_name:") and chord_id in line:
                                        found = fname
                                        break
                        if found:
                            break
                    except:
                        pass
            if not found:
                return None
            fpath = os.path.join(knowledge_dir, found)

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()

            fm = {}
            body = content
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    fm = ChordAnalyzer._parse_frontmatter(parts[1])
                    body = parts[2]

            # 提取标题
            title = fm.get("chord_name", chord_id)
            for line in body.split("\n"):
                if line.startswith("# "):
                    title = line[2:].strip()
                    break

            # 解析 fingers（可能嵌套 YAML）
            fingers = fm.get("fingers", {})
            if isinstance(fingers, str):
                fingers = {}

            return {
                "id": fm.get("id", chord_id),
                "chord_name": title,
                "difficulty": fm.get("difficulty", ""),
                "strings": fm.get("strings", [0, 0, 0, 0, 0, 0]),
                "fingers": fingers,
                "thumb_position": fm.get("thumb_position", {}),
                "related_problems": fm.get("related_problems", []),
                "related_techniques": fm.get("related_techniques", []),
                "tags": fm.get("tags", []),
                "body": body,
                "has_knowledge": True,
            }
        except Exception as e:
            logger.warning(f"和弦数据加载失败: {e}")
            return None

    @staticmethod
    def load_technique_knowledge(technique_id: str) -> Optional[dict]:
        """从 knowledge/ 目录加载技巧数据（搜索 techniques/子目录 + 根目录）"""
        knowledge_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "knowledge"
        )
        knowledge_base = os.path.join(knowledge_root, "techniques")
        if not os.path.exists(knowledge_base):
            return {"id": technique_id, "name": technique_id, "has_knowledge": False, "body": ""}

        # 搜索范围：techniques/根目录 + 子目录 (left-hand, right-hand) + knowledge/根目录
        search_dirs = [knowledge_base, knowledge_root]
        for sub in ["left-hand", "right-hand"]:
            sub_path = os.path.join(knowledge_base, sub)
            if os.path.isdir(sub_path):
                search_dirs.append(sub_path)

        # 直接匹配文件名
        fpath = None
        for d in search_dirs:
            candidate = os.path.join(d, f"{technique_id}.md")
            if os.path.exists(candidate):
                fpath = candidate
                break

        if not fpath:
            # 模糊搜索
            for d in search_dirs:
                for fname in sorted(os.listdir(d)):
                    if not fname.endswith(".md"):
                        continue
                    if technique_id.lower() in fname.lower():
                        fpath = os.path.join(d, fname)
                        break
                if fpath:
                    break

        if not fpath:
            return {"id": technique_id, "name": technique_id, "has_knowledge": False, "body": ""}

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()

            fm = {}
            body = content
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    fm = ChordAnalyzer._parse_frontmatter(parts[1])
                    body = parts[2]

            title = fm.get("id", technique_id)
            for line in body.split("\n"):
                if line.startswith("# "):
                    title = line[2:].strip()
                    break

            return {
                "id": fm.get("id", technique_id),
                "name": title,
                "difficulty": fm.get("difficulty", ""),
                "related_problems": fm.get("related_problems", []),
                "related_techniques": fm.get("related_techniques", []),
                "tags": fm.get("tags", []),
                "body": body,
                "has_knowledge": True,
            }
        except Exception as e:
            logger.warning(f"技巧数据加载失败: {e}")
            return {"id": technique_id, "name": technique_id, "has_knowledge": False, "body": ""}

    @staticmethod
    def make_generic_chord_data(chord_name: str) -> dict:
        """为未知和弦生成通用数据"""
        return {
            "id": "",
            "chord_name": chord_name,
            "difficulty": "unknown",
            "strings": [0, 0, 0, 0, 0, 0],
            "fingers": {},
            "thumb_position": {},
            "related_problems": ["flat-fingers", "excessive-pressure", "thumb-no-support"],
            "related_techniques": ["left-hand-arching", "wrist-posture"],
            "tags": ["自定义和弦"],
            "body": "",
            "has_knowledge": False,
        }

    # 标准调弦：弦号(6-1) → MIDI音高
    _OPEN_STRING_MIDI = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}

    # 和弦类型 → 音程集合（半音）
    _CHORD_TYPE_INTERVALS = {
        "major": [0, 4, 7],
        "minor": [0, 3, 7],
        "7": [0, 4, 7, 10],
        "m7": [0, 3, 7, 10],
        "maj7": [0, 4, 7, 11],
        "dim": [0, 3, 6],
    }

    @staticmethod
    def _pcset_from_strings(strings: list, capo: int = 0) -> set:
        """从 strings 数组计算实际音高集合。strings: [6弦,5弦,4弦,3弦,2弦,1弦], x=muted, 0=open, N=fret"""
        midi_set = set()
        for i, s in enumerate(strings):
            string_num = 6 - i
            if s == "x" or s == "X":
                continue
            fret = int(s) if str(s).isdigit() else 0
            midi = ChordAnalyzer._OPEN_STRING_MIDI.get(string_num, 40) + fret + capo
            midi_set.add(midi % 12)
        return frozenset(midi_set)

    # 常见开放和弦(不需横按) — 用于capo推断
    _OPEN_SHAPE_CHORDS = {
        "C", "C7", "Cmaj7", "D", "D7", "Dm", "Dm7", "D-over-Fsharp",
        "E", "E7", "Em", "G", "G7", "A", "A7", "Am", "Am7",
    }

    @staticmethod
    def infer_capo_from_chords(notes: list, detected_chords: list) -> int:
        """从检测到的和弦名推断变调夹位置。

        原理: 尝试不同capo值(0-5)，看哪个能让和弦名全部映射到开放手型。
        选择使开放手型比例最高的capo值。
        """
        if len(detected_chords) < 2:
            return 0

        chord_names = [c.get("chord_name", "").replace("和弦", "") for c in detected_chords]

        # 音名半音映射
        note_to_semi = {"C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,"F#":6,"Gb":6,"G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11}
        semi_to_note_sharp = {0:"C",1:"C#",2:"D",3:"D#",4:"E",5:"F",6:"F#",7:"G",8:"G#",9:"A",10:"A#",11:"B"}

        def chord_root(cn):
            if not cn: return None
            if len(cn) >= 2 and cn[1] in "#b": root = cn[:2]
            else: root = cn[0]
            return root

        def count_open(names):
            c = 0
            for cn in names:
                if cn in ChordAnalyzer._OPEN_SHAPE_CHORDS:
                    c += 1
                    continue
                root = chord_root(cn)
                if root and root in {"C","D","E","G","A"} and len(cn) <= 4:
                    c += 1
            return c

        # 基准: capo=0 的开放手型比例
        baseline = count_open(chord_names)

        # 尝试 capo 1-5，找最佳
        best_capo, best_score = 0, baseline
        for capo_try in range(1, 6):
            transposed = []
            for cn in chord_names:
                root = chord_root(cn)
                if not root or root not in note_to_semi:
                    transposed.append(cn)
                    continue
                semi = note_to_semi[root]
                new_semi = (semi - capo_try) % 12
                new_root = semi_to_note_sharp[new_semi]
                new_cn = cn.replace(root, new_root, 1) if root in cn else new_root
                transposed.append(new_cn)
            score = count_open(transposed)
            if score > best_score:
                best_score = score
                best_capo = capo_try

        # 只有非零capo明显优于基准(≥15%提升或绝对优势≥2)才采用
        if best_capo > 0 and (best_score - baseline >= 2 or best_score >= len(chord_names) * 0.75):
            return best_capo
        return 0

    @staticmethod
    def detect_chords_from_notes(notes: List[dict], capo: int = 0,
                                 min_confidence: float = 0.85) -> List[dict]:
        """从 basic-pitch 转录音符中检测和弦。

        策略：
        1. 将音符按起始时间分组（50ms 窗口内 = 同时弹奏）
        2. 每组提取音高类别 → 与所有已知和弦的弦位音高匹配
        3. 返回检测到的和弦列表 [{time, chord_id, chord_name, confidence}, ...]
        """
        if not notes or len(notes) < 2:
            return []

        # 加载所有和弦知识的音高集合
        knowledge_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "knowledge", "chords"
        )
        chord_pcsets = {}  # chord_id → {pcset, chord_name}
        if os.path.isdir(knowledge_dir):
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
                        pcset = ChordAnalyzer._pcset_from_strings(strings, capo)
                        chord_name = fm.get("chord_name", chord_id)
                        chord_pcsets[chord_id] = {
                            "pcset": pcset,
                            "chord_name": chord_name,
                            "difficulty": fm.get("difficulty", ""),
                            "related_problems": fm.get("related_problems", []),
                            "related_techniques": fm.get("related_techniques", []),
                        }
                except Exception:
                    continue

        if not chord_pcsets:
            return []

        # 按起始时间分组（50ms 窗口）
        sorted_notes = sorted(notes, key=lambda n: n.get("start_time", 0))
        clusters = []
        current_cluster = [sorted_notes[0]]
        cluster_start = sorted_notes[0].get("start_time", 0)
        for note in sorted_notes[1:]:
            t = note.get("start_time", 0)
            if t - cluster_start < 0.05:
                current_cluster.append(note)
            else:
                clusters.append(current_cluster)
                current_cluster = [note]
                cluster_start = t
        clusters.append(current_cluster)

        # 匹配每组音符到和弦
        detected = []
        for cluster in clusters:
            if len(cluster) < 3:
                continue
            t = cluster[0].get("start_time", 0)
            pc_set = frozenset(n.get("pitch", 0) % 12 for n in cluster)
            if len(pc_set) < 3:
                continue

            best_id, best_info, best_score = None, None, 0
            for cid, info in chord_pcsets.items():
                if len(info["pcset"]) == 0:
                    continue
                overlap = len(pc_set & info["pcset"])
                total = len(pc_set | info["pcset"])
                jaccard = overlap / total if total > 0 else 0
                coverage = overlap / len(info["pcset"]) if len(info["pcset"]) > 0 else 0
                # 基础分: Jaccard, 不加 size_bonus (避免手指数相同但音不匹配的虚假加分)
                score = jaccard
                # 覆盖率奖励: 和弦音全部出现时才加分
                if info["pcset"].issubset(pc_set):
                    score += 0.10
                elif coverage >= 0.75:
                    score += 0.05
                if score > best_score and score >= 0.55:
                    best_score = score
                    best_id = cid
                    best_info = info

            if best_id and best_score >= min_confidence:
                detected.append({
                    "time": round(t, 2),
                    "chord_id": best_id,
                    "chord_name": best_info["chord_name"],
                    "confidence": round(min(1.0, best_score), 2),
                    "related_problems": best_info.get("related_problems", []),
                    "related_techniques": best_info.get("related_techniques", []),
                })
            elif best_id and best_score >= 0.55:
                logger.info(f"和弦候选(未达阈值): {best_info['chord_name']} score={best_score:.2f} at {t:.1f}s")

        return detected

    # ====== 视频手型和弦检测 ======

    # 指尖 landmark 索引: 拇指=4, 食指=8, 中指=12, 无名指=16, 小指=20
    # MCP 指根索引: 拇指=2, 食指=5, 中指=9, 无名指=13, 小指=17
    _FINGER_TIP_MCP = {
        "食指": (8, 5), "中指": (12, 9), "无名指": (16, 13), "小指": (20, 17),
    }

    @staticmethod
    def _extract_finger_positions(landmarks):
        """从一帧的 hand landmarks 中提取各手指的指尖位置和按压状态。

        按压判定: 指尖 z 明显在 MCP 前方(弯曲按弦)。
        小指阈值更高——小指天然容易悬浮在指板上方,需要更强的弯曲信号才判定为按压。
        """
        result = {}
        for finger_name, (tip_idx, mcp_idx) in ChordAnalyzer._FINGER_TIP_MCP.items():
            if tip_idx >= len(landmarks) or mcp_idx >= len(landmarks):
                continue
            tip = landmarks[tip_idx]
            mcp = landmarks[mcp_idx]
            z_tip = tip[2] if isinstance(tip, (list, tuple)) else tip.z
            z_mcp = mcp[2] if isinstance(mcp, (list, tuple)) else mcp.z
            z_depth = z_mcp - z_tip
            # 小指需要更强的弯曲信号(>0.012), 其他手指 >0.008
            threshold = 0.012 if finger_name == "小指" else 0.008
            is_pressing = z_depth > threshold
            result[finger_name] = {
                "tip": (tip[0], tip[1], tip[2]) if isinstance(tip, (list, tuple)) else (tip.x, tip.y, tip.z),
                "mcp": (mcp[0], mcp[1], mcp[2]) if isinstance(mcp, (list, tuple)) else (mcp.x, mcp.y, mcp.z),
                "pressing": is_pressing,
                "z_depth": round(z_depth, 4),
            }
        return result

    @staticmethod
    def _load_chord_templates_with_finger_data():
        """加载所有和弦模板，包含 finger→(string, fret) 映射和预计算特征。"""
        knowledge_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "knowledge", "chords"
        )
        templates = {}
        if not os.path.isdir(knowledge_dir):
            return templates

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
                chord_name = fm.get("chord_name", chord_id)
                strings_raw = fm.get("strings", [])
                fingers_raw = fm.get("fingers", {})

                if not strings_raw or len(strings_raw) != 6:
                    continue

                # Parse finger→(string, fret, pip_ideal, mcp_ideal)
                finger_map = {}
                finger_angles = {}  # 逐指理想角度，供 KB 驱动检测使用
                if isinstance(fingers_raw, str):
                    fingers_raw = ChordAnalyzer._parse_inline_dict(fingers_raw)
                if isinstance(fingers_raw, dict):
                    for finger_name, finger_data in fingers_raw.items():
                        if isinstance(finger_data, str):
                            finger_data = ChordAnalyzer._parse_inline_dict(finger_data)
                        if isinstance(finger_data, dict):
                            s = finger_data.get("string", 0)
                            f_val = finger_data.get("fret", 0)
                            pip = finger_data.get("pip_ideal")
                            mcp = finger_data.get("mcp_ideal")
                            # 横按指标记：横按手指（食指）正确姿势就是伸直侧面压弦，
                            # 常规角度阈值会误报「太竖直/太弯」。来源：note/desc/tip_contact/barre_note
                            # 含「横按」，或 string 为范围 "1-5"（横按覆盖多弦）。
                            _barre_markers = " ".join(
                                str(finger_data.get(k, ""))
                                for k in ("note", "desc", "tip_contact", "barre_note")
                            )
                            _is_barre = ("横按" in _barre_markers) or (
                                isinstance(s, str) and "-" in s
                            )
                            if s and f_val is not None:
                                if isinstance(s, str) and "-" in s:
                                    # 横按范围 "1-5"/"5-1"/"1-2" → 取 min 弦作代表
                                    s = min(int(x) for x in s.split("-"))
                                finger_map[finger_name] = (int(s), int(f_val))
                            if pip is not None or mcp is not None:
                                finger_angles[finger_name] = {
                                    "pip_ideal": pip,
                                    "mcp_ideal": mcp,
                                }
                                if _is_barre:
                                    finger_angles[finger_name]["barre"] = True

                if not finger_map:
                    continue

                # 预计算和弦特征指纹 (用于相对模式匹配)
                finger_set = frozenset(finger_map.keys())
                # 按弦号排序的手指列表 [(finger, string, fret), ...] ⑥弦(大号)→①弦(小号)
                sorted_by_string = sorted(
                    [(name, s, f) for name, (s, f) in finger_map.items()],
                    key=lambda x: x[1], reverse=True  # 大弦号→小弦号
                )
                # 弦间距模式: [gap1, gap2, ...] 相邻按压弦之间的弦数差
                string_gaps = tuple(
                    sorted_by_string[i][1] - sorted_by_string[i+1][1]
                    for i in range(len(sorted_by_string) - 1)
                )
                # y 排序: 弦号大=⑥弦(画面中偏上/y偏小), 弦号小=①弦(画面中偏下/y偏大)
                y_order = tuple(name for name, s, f in sorted_by_string)  # 从⑥弦→①弦的y顺序
                # 品位分组: 相同品位的归为一组
                fret_groups = {}
                for name, s, f in sorted_by_string:
                    fret_groups.setdefault(f, []).append(name)
                fret_pattern = tuple(sorted(
                    tuple(sorted(g)) for g in fret_groups.values()
                ))  # e.g., (("食指","中指"), ("无名指",)) for D chord

                # Extract KB rule sections for AI-aware filtering
                body = parts[2] if len(parts) >= 3 else ""
                normal_section = ""
                error_section = ""
                if "正常表现" in body or "不应算问题" in body:
                    markers = ["正常表现", "不应算问题"]
                    for marker in markers:
                        if marker in body:
                            idx = body.index(marker)
                            chunk = body[idx:idx + 600]
                            end_markers = ["实际错误信号", "## 验证", "## 练习", "---"]
                            for em in end_markers:
                                if em in chunk:
                                    chunk = chunk[:chunk.index(em)]
                                    break
                            normal_section = chunk[:400]
                            break
                if "实际错误信号" in body:
                    idx = body.index("实际错误信号")
                    chunk = body[idx:idx + 600]
                    for em in ["## 验证", "## 练习", "---"]:
                        if em in chunk:
                            chunk = chunk[:chunk.index(em)]
                            break
                    error_section = chunk[:400]

                templates[chord_id] = {
                    "chord_id": chord_id,
                    "chord_name": chord_name,
                    "strings": strings_raw,
                    "fingers": finger_map,
                    "finger_angles": finger_angles,
                    "finger_count": len(finger_map),
                    "finger_set": finger_set,
                    "string_gaps": string_gaps,
                    "y_order": y_order,
                    "fret_pattern": fret_pattern,
                    "difficulty": fm.get("difficulty", ""),
                    "related_problems": fm.get("related_problems", []),
                    "related_techniques": fm.get("related_techniques", []),
                    "_normal_section": normal_section,
                    "_error_section": error_section,
                }
            except Exception:
                continue
        return templates

    @staticmethod
    def _match_finger_pattern(finger_positions: dict, templates: dict,
                               mirror_detected: bool = False, hand_zone: float = None) -> dict:
        """将检测到的手指按压模式与和弦模板匹配(相对模式匹配, 不依赖绝对坐标映射)。

        核心思路: 比较检测到的手指位置关系(finger_set, y_order, fret_groups)
        与每个和弦模板的期望关系, 而非尝试映射 (x,y) → (string, fret)。
        """
        pressing_fingers = {name: data for name, data in finger_positions.items() if data["pressing"]}
        n_pressing = len(pressing_fingers)

        if n_pressing == 0:
            return None

        # ── 计算检测到的特征 ──
        # y排序(从小到大=⑥弦→①弦)
        sorted_by_y = sorted(pressing_fingers.items(), key=lambda x: x[1]["tip"][1])
        detected_y_order = tuple(name for name, _ in sorted_by_y)
        detected_finger_set = frozenset(pressing_fingers.keys())

        # y间距模式(归一化)
        y_vals = [d["tip"][1] for _, d in sorted_by_y]
        if len(y_vals) >= 2:
            y_range = y_vals[-1] - y_vals[0]
            if y_range > 0.001:
                y_gaps = tuple(round((y_vals[i+1] - y_vals[i]) / y_range, 2) for i in range(len(y_vals)-1))
            else:
                y_gaps = tuple([0.5] * (len(y_vals)-1))
        else:
            y_gaps = ()

        # x聚类(品位分组): 相邻手指x差 < 0.03 → 同品
        x_vals = {name: d["tip"][0] for name, d in pressing_fingers.items()}
        # 简单聚类: x差<0.03的归为一组
        fingers_by_x = sorted(pressing_fingers.items(), key=lambda x: x[1]["tip"][0])
        detected_fret_groups = []
        current_group = [fingers_by_x[0][0]]
        for i in range(1, len(fingers_by_x)):
            if fingers_by_x[i][1]["tip"][0] - fingers_by_x[i-1][1]["tip"][0] < 0.03:
                current_group.append(fingers_by_x[i][0])
            else:
                detected_fret_groups.append(tuple(sorted(current_group)))
                current_group = [fingers_by_x[i][0]]
        detected_fret_groups.append(tuple(sorted(current_group)))
        detected_fret_pattern = tuple(sorted(detected_fret_groups))

        # ── 匹配每个模板 ──
        best_match = None
        best_score = 0.0

        for chord_id, tmpl in templates.items():
            tmpl_finger_set = tmpl.get("finger_set", frozenset())
            tmpl_y_order = tmpl.get("y_order", ())
            tmpl_string_gaps = tmpl.get("string_gaps", ())
            tmpl_fret_pattern = tmpl.get("fret_pattern", ())
            tmpl_n = tmpl.get("finger_count", 0)

            score = 0.0

            # 1. 手指集合匹配 (Jaccard) — 权重 0.30
            if len(tmpl_finger_set | detected_finger_set) > 0:
                jaccard = len(tmpl_finger_set & detected_finger_set) / len(tmpl_finger_set | detected_finger_set)
            else:
                jaccard = 0
            score += 0.30 * jaccard

            # 如果手指集合完全不同, 跳过
            if jaccard < 0.4:
                continue

            # 2. y排序匹配 — 权重 0.35
            if len(tmpl_y_order) == len(detected_y_order) and len(tmpl_y_order) >= 2:
                # Kendall tau 风格的排序比较
                # tmpl_y_order 和 detected_y_order 的归一化后应一致
                matches = 0
                for i, tf in enumerate(tmpl_y_order):
                    if i < len(detected_y_order) and detected_y_order[i] == tf:
                        matches += 1
                order_score = matches / len(tmpl_y_order)

                # 间隙模式比较: 模板弦间距 vs 检测y间距
                if len(tmpl_string_gaps) == len(y_gaps) and len(y_gaps) >= 1:
                    # 归一化模板弦间距
                    tmpl_gap_sum = sum(tmpl_string_gaps)
                    if tmpl_gap_sum > 0:
                        tmpl_gaps_norm = tuple(g / tmpl_gap_sum for g in tmpl_string_gaps)
                        gap_corr = sum(abs(tmpl_gaps_norm[i] - y_gaps[i]) for i in range(len(y_gaps)))
                        gap_score = max(0, 1.0 - gap_corr)
                    else:
                        gap_score = 0.5
                else:
                    gap_score = 0.5

                score += 0.35 * (0.5 * order_score + 0.5 * gap_score)
            elif len(tmpl_y_order) == len(detected_y_order) == 1:
                score += 0.35  # 单指无法比排序, 给满分
            else:
                score += 0.10  # 手指数不同, 给低分

            # 3. 品位分组匹配 — 权重 0.25
            if detected_fret_pattern and tmpl_fret_pattern:
                # 比较分组结构
                det_struct = tuple(sorted(len(g) for g in detected_fret_pattern))
                tmpl_struct = tuple(sorted(len(g) for g in tmpl_fret_pattern))
                if det_struct == tmpl_struct:
                    fret_score = 1.0
                elif len(detected_fret_pattern) == len(tmpl_fret_pattern):
                    fret_score = 0.6
                else:
                    fret_score = 0.3
                score += 0.25 * fret_score

            # 4. 手区位奖励 — 权重 0.10
            if hand_zone is not None:
                has_open = any(s in (0, "0") for s in tmpl.get("strings", []))
                is_low = hand_zone <= 5.0
                if (is_low and has_open) or (not is_low and not has_open):
                    score += 0.10
                elif has_open and 5.0 < hand_zone <= 8.0:
                    score += 0.05  # 勉强可能是开放和弦

            if score > best_score:
                best_score = score
                best_match = {
                    "chord_id": chord_id,
                    "chord_name": tmpl["chord_name"],
                    "confidence": round(min(0.95, score), 2),
                    "finger_count": tmpl.get("finger_count", 0),
                }

        if best_match and best_match["confidence"] >= 0.60:
            return best_match
        return None

    @staticmethod
    def _cluster_chord_frames(frame_chords: list) -> list:
        """将逐帧检测结果聚合为和弦时间段。

        Args:
            frame_chords: [(time, chord_id, chord_name, confidence), ...]

        Returns:
            [{"time": float, "chord_id": str, "chord_name": str, "confidence": float}, ...]
        """
        if not frame_chords:
            return []

        # 按时间排序
        frame_chords.sort(key=lambda x: x[0])

        # 合并相邻同和弦帧
        clusters = []
        current = None
        for t, cid, cname, conf, fcount in frame_chords:
            if current is None:
                current = {"chord_id": cid, "chord_name": cname,
                           "times": [t], "confs": [conf], "finger_counts": [fcount]}
            elif cid == current["chord_id"] and t - current["times"][-1] < 1.5:
                current["times"].append(t)
                current["confs"].append(conf)
                current["finger_counts"].append(fcount)
            else:
                clusters.append(current)
                current = {"chord_id": cid, "chord_name": cname,
                           "times": [t], "confs": [conf], "finger_counts": [fcount]}
        if current:
            clusters.append(current)

        # 过滤: 至少2帧(约80ms)以上才认为有效
        detected = []
        for cl in clusters:
            if len(cl["times"]) < 2:
                continue
            avg_time = sum(cl["times"]) / len(cl["times"])
            avg_conf = sum(cl["confs"]) / len(cl["confs"])
            # 取众数作为该和弦段的手指数
            from collections import Counter
            fcount_counter = Counter(cl["finger_counts"])
            dominant_fcount = fcount_counter.most_common(1)[0][0]
            detected.append({
                "time": round(avg_time, 1),
                "chord_id": cl["chord_id"],
                "chord_name": cl["chord_name"],
                "confidence": round(avg_conf, 2),
                "finger_count": dominant_fcount,
                "_source": "video",
            })

        return sorted(detected, key=lambda d: d["time"])

    @staticmethod
    def detect_chords_from_landmarks(hand_landmarks: list, mirror_detected: bool = False,
                                      hand_zone: float = None) -> list:
        """从视频手部关键点数据检测和弦。

        Args:
            hand_landmarks: [{"time": float, "hand": "Left"/"Right", "landmarks": [(x,y,z)*21]}, ...]
            mirror_detected: 是否为镜像画面(前置摄像头)
            hand_zone: 估算的手区品数

        Returns:
            [{"time": float, "chord_id": str, "chord_name": str, "confidence": float, "_source": "video"}, ...]
        """
        if not hand_landmarks:
            logger.info("视频和弦检测: 无手部关键点数据")
            return []

        templates = ChordAnalyzer._load_chord_templates_with_finger_data()
        if not templates:
            logger.info("视频和弦检测: 无和弦模板")
            return []

        # 只处理左手帧
        left_frames = [f for f in hand_landmarks if f.get("hand") == "Left"]
        if not left_frames:
            logger.info("视频和弦检测: 无左手帧")
            return []

        frame_chords = []
        finger_history = []  # 用于平滑: 最近N帧的按压状态

        for frame_data in left_frames:
            landmarks = frame_data.get("landmarks", [])
            if len(landmarks) < 21:
                continue
            t = frame_data.get("time", 0)

            finger_positions = ChordAnalyzer._extract_finger_positions(landmarks)
            # 使用多帧平滑: 记录最近3帧的按压手指集合
            pressing_now = frozenset(name for name, d in finger_positions.items() if d["pressing"])
            finger_history.append(pressing_now)
            if len(finger_history) > 3:
                finger_history.pop(0)

            # 只在连续2帧以上检测到相同按压模式时才匹配
            if len(finger_history) >= 2:
                if pressing_now != finger_history[-2] if len(finger_history) >= 2 else False:
                    continue
                # 更新 finger_positions 中的 pressing 状态为稳定的按压手指
                stable_pressing = pressing_now
                for name in list(finger_positions.keys()):
                    finger_positions[name]["pressing"] = name in stable_pressing

            best = ChordAnalyzer._match_finger_pattern(
                finger_positions, templates, mirror_detected, hand_zone
            )
            if best:
                frame_chords.append((t, best["chord_id"], best["chord_name"],
                                     best["confidence"], best.get("finger_count", 0)))

        detected = ChordAnalyzer._cluster_chord_frames(frame_chords)
        if detected:
            names = [f"{c['chord_name']}(vid:{c['confidence']:.2f})" for c in detected[:6]]
            logger.info(f"视频和弦检测: {len(left_frames)}帧 -> {len(detected)}个和弦: {names}")
        else:
            logger.info(f"视频和弦检测: {len(left_frames)}帧 -> 未检测到和弦")

        return detected

    @staticmethod
    def finger_positions_to_chord_id(finger_positions: dict) -> Optional[dict]:
        """将 Qwen VL 检测到的绝对弦/品位置映射到 chord_id。

        Args:
            finger_positions: {"食指": {"string": 2, "fret": 1}, "中指": {"string": 4, "fret": 2}, ...}

        Returns:
            {"chord_id": "C", "chord_name": "C和弦", "confidence": 0.95, "finger_count": 2} or None
        """
        if not finger_positions:
            return None

        templates = ChordAnalyzer._load_chord_templates_with_finger_data()
        if not templates:
            return None

        # Convert VL format to (string, fret) tuples
        vl_fingers = {}
        for fname, fdata in finger_positions.items():
            if isinstance(fdata, dict) and "string" in fdata and "fret" in fdata:
                vl_fingers[fname] = (int(fdata["string"]), int(fdata["fret"]))

        if not vl_fingers:
            return None

        best_match = None
        best_score = 0.0

        for chord_id, tmpl in templates.items():
            tmpl_fingers = tmpl.get("fingers", {})
            if not tmpl_fingers:
                continue

            # Position-based matching: compare (string, fret) pairs regardless of finger name.
            # Different players use different fingers for the same position
            # (e.g. Em: 食指+中指 vs 中指+无名指), so matching by finger name is unreliable.
            vl_positions = [(s, f) for s, f in vl_fingers.values()]
            tmpl_positions = [(s, f) for s, f in tmpl_fingers.values()]
            total_vl = len(vl_positions)
            total_tmpl = len(tmpl_positions)

            # Greedy assignment: for each VL position, find best-matching template position
            unmatched = list(tmpl_positions)
            matches = 0.0
            for vl_s, vl_f in vl_positions:
                best_idx = -1
                best_dist = float("inf")
                for i, (t_s, t_f) in enumerate(unmatched):
                    if vl_s == t_s and vl_f == t_f:
                        best_dist = 0.0  # perfect match
                        best_idx = i
                        break
                    elif vl_s == t_s:
                        dist = abs(vl_f - t_f) * 0.5  # same string, different fret
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = i
                if best_idx >= 0:
                    matches += 1.0 if best_dist == 0.0 else 0.5
                    unmatched.pop(best_idx)

            if matches == 0:
                continue

            # Jaccard-like score
            union_size = total_vl + total_tmpl - matches
            jaccard = matches / union_size if union_size > 0 else 0

            # Coverage: how many template fingers are matched
            coverage = matches / total_tmpl if total_tmpl > 0 else 0

            # Combined score: prefer exact matches but allow missing fingers
            score = jaccard * 1.0 + coverage * 0.5

            # Finger count penalty: VL seeing N fingers but template expects M → penalty
            # Helps distinguish E (3 fingers) vs Em (2), Em (2) vs Am (3)
            finger_diff = abs(total_vl - total_tmpl)
            if finger_diff > 0:
                finger_penalty = 1.0 / (1.0 + finger_diff * 0.50)
                score *= finger_penalty

            if score > best_score:
                best_score = score
                best_match = {
                    "chord_id": chord_id,
                    "chord_name": tmpl.get("chord_name", chord_id),
                    "confidence": round(min(0.95, score / 1.5), 2),
                    "finger_count": total_tmpl,
                    "_matched_fingers": matches,
                    "_total_vl_fingers": total_vl,
                }

        if best_match and best_match["confidence"] >= 0.40:
            logger.info(f"VL和弦映射: {vl_fingers} → {best_match['chord_name']} "
                        f"(conf={best_match['confidence']:.2f}, matched={best_match['_matched_fingers']})")
            return best_match

        return None
