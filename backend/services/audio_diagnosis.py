"""
音频诊断引擎：将 AudioFeatures 原始数据转换为结构化诊断结果。

三层架构中的 Layer 2 — 信号 → 诊断的翻译层。
绿区(正常) / 黄区(边缘) / 红区(明确问题) 三级分类。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class Finding:
    """单条诊断发现"""
    category: str          # rhythm / dynamics / tone / noise / technique
    zone: str              # green / yellow / red
    label: str             # 人类可读标签，如 "节奏不稳定"
    severity: str          # 无 / 轻微 / 中等 / 明显 / 严重
    evidence: str          # 证据描述，如 "连续8个小节内节拍偏差超过40ms"
    metrics: Dict = field(default_factory=dict)  # 相关原始数据
    suggestion: str = ""   # 简短建议


@dataclass
class AudioDiagnosis:
    """结构化音频诊断结果"""
    findings: List[Finding] = field(default_factory=list)
    green_count: int = 0
    yellow_count: int = 0
    red_count: int = 0
    summary: str = ""


def diagnose(audio_result: Dict[str, Any]) -> AudioDiagnosis:
    """从音频分析结果中生成结构化诊断。

    Args:
        audio_result: audio_analyzer.analyze() 的返回字典

    Returns:
        AudioDiagnosis 包含所有诊断发现
    """
    diag = AudioDiagnosis()
    findings = []

    af = audio_result.get("audio_features", {})
    errors = audio_result.get("errors", [])
    stats = audio_result.get("stats", {})
    notes = audio_result.get("notes", [])

    # ── 1. 节拍稳定性 ──
    findings.extend(_diagnose_rhythm(af, errors))

    # ── 1.5. 逐音符节奏问题定位 ──
    tempo_bpm = af.get("tempo_bpm", 0)
    if tempo_bpm > 0 and notes:
        findings.extend(_diagnose_per_note_rhythm(notes, tempo_bpm))

    # ── 2. 动态/力度 ──
    findings.extend(_diagnose_dynamics(af))

    # ── 4. 音色 ──
    findings.extend(_diagnose_tone(af))

    # ── 5. 杂音 ──
    findings.extend(_diagnose_noise(af))

    # ── 6. 音准/错音 ──
    findings.extend(_diagnose_pitch_errors(errors, stats))

    # ── 7. 扫弦质量 ──
    technique_segs = audio_result.get("technique_segments", [])
    findings.extend(_diagnose_strumming(af, technique_segs))

    # ── 统计 ──
    diag.findings = findings
    diag.red_count = sum(1 for f in findings if f.zone == "red")
    diag.yellow_count = sum(1 for f in findings if f.zone == "yellow")
    diag.green_count = sum(1 for f in findings if f.zone == "green")

    # ── 摘要 ──
    diag.summary = _build_summary(findings)

    return diag


def _diagnose_rhythm(af: Dict, errors: List[Dict]) -> List[Finding]:
    findings = []
    ioi_cv = af.get("ioi_cv", 0)
    ioi_cv_core = af.get("ioi_cv_core", 0)
    rush_regions = af.get("local_rush_regions", [])
    drag_regions = af.get("local_drag_regions", [])

    # 优先使用核心 CV（排除 P90 以上离群值，如乐句呼吸、扫弦交替）
    # 吉他演奏中扫弦(极短IOI)和旋律(中等IOI)天然混合，CV 天生偏高
    cv = ioi_cv_core if ioi_cv_core > 0 else ioi_cv

    if cv <= 0:
        return findings

    if cv < 0.20:
        findings.append(Finding(
            category="rhythm", zone="green", label="节拍稳定",
            severity="无",
            evidence=f"节拍稳定性良好，各音符时值均匀",
            metrics={"ioi_cv": ioi_cv, "ioi_cv_core": ioi_cv_core},
        ))
    elif cv < 0.35:
        findings.append(Finding(
            category="rhythm", zone="green", label="节拍大体稳定",
            severity="无",
            evidence=f"节拍稳定性良好，扫弦和旋律的交替在正常范围内",
            metrics={"ioi_cv": ioi_cv, "ioi_cv_core": ioi_cv_core},
        ))
    elif cv < 0.50:
        findings.append(Finding(
            category="rhythm", zone="yellow", label="节拍欠稳",
            severity="轻微",
            evidence=f"节拍稳定性一般，部分音符时值有偏差",
            metrics={"ioi_cv": ioi_cv, "ioi_cv_core": ioi_cv_core},
            suggestion="用节拍器慢练，确保每个音符的时值均匀",
        ))
    else:
        findings.append(Finding(
            category="rhythm", zone="red", label="节拍不稳定",
            severity="中等",
            evidence=f"节拍稳定性偏差明显，多处时值不均匀",
            metrics={"ioi_cv": ioi_cv, "ioi_cv_core": ioi_cv_core},
            suggestion="先用节拍器在50%原速下练习，重点感受每个拍的落点",
        ))

    # 加速区域
    if rush_regions:
        regions_desc = "、".join(
            f"{r['start']:.1f}s-{r['end']:.1f}s" for r in rush_regions[:3]
        )
        findings.append(Finding(
            category="rhythm", zone="red", label="存在明显加速",
            severity="明显",
            evidence=f"在 {regions_desc} 处存在加速（速度突增 {1-rush_regions[0]['ratio']:.0%}）",
            metrics={"rush_regions": rush_regions},
            suggestion="加速段落单独练习，刻意控制速度不要变快",
        ))

    # 拖拍区域
    if drag_regions:
        regions_desc = "、".join(
            f"{r['start']:.1f}s-{r['end']:.1f}s" for r in drag_regions[:3]
        )
        findings.append(Finding(
            category="rhythm", zone="red", label="存在明显拖拍",
            severity="明显",
            evidence=f"在 {regions_desc} 处存在拖拍（速度下降 {drag_regions[0]['ratio']-1:.0%}）",
            metrics={"drag_regions": drag_regions},
            suggestion="拖拍段落通常是技术难点，单独慢练确保手指能跟上",
        ))

    return findings


def _diagnose_per_note_rhythm(notes: List[Dict], tempo_bpm: float) -> List[Finding]:
    """逐音符节奏诊断：找出明显偏离拍点的个别音符。

    排除停顿后紧跟的音符（gap > 1s），避免停顿/换和弦导致的连锁误报。
    """
    findings = []
    if not notes or len(notes) < 3 or tempo_bpm <= 0:
        return findings

    beat_interval = 60.0 / tempo_bpm
    notes_sorted = sorted(notes, key=lambda n: n.get("start_time", 0))

    deviations = []
    prev_end = None
    for n in notes_sorted:
        t = n.get("start_time", 0)
        # 跳过停顿后的音符：前一个音符结束距今 > 1s，说明是换和弦/乐句分界
        if prev_end is not None and t - prev_end > 1.0:
            prev_end = max(prev_end, n.get("end_time", t))
            continue
        prev_end = max(prev_end or 0, n.get("end_time", t))

        nearest_beat = round(t / beat_interval) * beat_interval
        deviation = t - nearest_beat
        pos = n.get("guitar_position", "") or ""
        deviations.append((t, deviation, pos))

    threshold = beat_interval * 0.15
    bad_notes = [(t, dev, pos) for t, dev, pos in deviations if abs(dev) > threshold]
    if not bad_notes:
        return findings

    rush_notes = [(t, dev, pos) for t, dev, pos in bad_notes if dev < 0][:5]
    drag_notes = [(t, dev, pos) for t, dev, pos in bad_notes if dev > 0][:5]

    parts = []
    if rush_notes:
        descs = [f"第{t:.1f}秒 偏快了约{abs(dev)*1000:.0f}毫秒" for t, dev, pos in rush_notes]
        if len([x for x in bad_notes if x[1] < 0]) > len(rush_notes):
            descs.append("等")
        parts.append("偏快: " + "；".join(descs))
    if drag_notes:
        descs = [f"第{t:.1f}秒 偏慢了约{dev*1000:.0f}毫秒" for t, dev, pos in drag_notes]
        if len([x for x in bad_notes if x[1] > 0]) > len(drag_notes):
            descs.append("等")
        parts.append("偏慢: " + "；".join(descs))

    if parts:
        # 不传精确数字给 LLM，防止报告里生成 "56处偏差" 这种误导性数字
        bad_count = len(bad_notes)
        if bad_count <= 3:
            qualifier = "个别"
        elif bad_count <= 10:
            qualifier = "多处"
        else:
            qualifier = "部分段落"

        findings.append(Finding(
            category="rhythm", zone="yellow",
            label=f"节奏偏差（{qualifier}）",
            severity="轻微" if bad_count <= 3 else "中等",
            evidence="；".join(parts),
            metrics={"bad_note_count": bad_count, "rush_count": len(rush_notes), "drag_count": len(drag_notes)},
            suggestion="以上位置单独放慢练习，用节拍器确保每个音符都卡在拍点上",
        ))

    return findings


def _diagnose_dynamics(af: Dict) -> List[Finding]:
    findings = []
    dynamic_range = af.get("dynamic_range_db", 0)

    if dynamic_range <= 0:
        return findings

    if dynamic_range > 15:
        findings.append(Finding(
            category="dynamics", zone="green", label="动态表现丰富",
            severity="无",
            evidence=f"动态范围 {dynamic_range:.1f}dB，力度有层次变化",
            metrics={"dynamic_range_db": dynamic_range},
        ))
    elif dynamic_range > 8:
        findings.append(Finding(
            category="dynamics", zone="green", label="动态变化适中",
            severity="无",
            evidence=f"动态范围 {dynamic_range:.1f}dB，有一定的力度变化",
            metrics={"dynamic_range_db": dynamic_range},
        ))
    elif dynamic_range > 4:
        findings.append(Finding(
            category="dynamics", zone="yellow", label="力度变化偏小",
            severity="轻微",
            evidence=f"动态范围仅 {dynamic_range:.1f}dB，演奏力度较单一",
            metrics={"dynamic_range_db": dynamic_range},
            suggestion="尝试在乐句中加入强弱对比，让音乐更有呼吸感",
        ))
    else:
        findings.append(Finding(
            category="dynamics", zone="red", label="力度过于均匀",
            severity="中等",
            evidence=f"动态范围仅 {dynamic_range:.1f}dB，几乎无力度变化",
            metrics={"dynamic_range_db": dynamic_range},
            suggestion="刻意练习：同一个乐句用pp→mf→ff三种力度各弹一遍",
        ))

    return findings


def _diagnose_tone(af: Dict) -> List[Finding]:
    findings = []
    centroid = af.get("spectral_centroid_avg", 0)
    attack = af.get("attack_time_avg_ms", 0)

    if centroid > 0:
        if centroid > 3000:
            findings.append(Finding(
                category="tone", zone="yellow", label="音色偏亮/偏尖",
                severity="轻微",
                evidence=f"频谱质心 {centroid:.0f}Hz，高频成分偏多",
                metrics={"spectral_centroid_avg": centroid},
                suggestion="尝试调整拨弦角度或力度，让音色更圆润",
            ))
        elif centroid > 2000:
            findings.append(Finding(
                category="tone", zone="green", label="音色明亮",
                severity="无",
                evidence=f"频谱质心 {centroid:.0f}Hz，音色清晰明亮",
                metrics={"spectral_centroid_avg": centroid},
            ))
        elif centroid > 1000:
            findings.append(Finding(
                category="tone", zone="green", label="音色适中",
                severity="无",
                evidence=f"频谱质心 {centroid:.0f}Hz，音色温暖均衡",
                metrics={"spectral_centroid_avg": centroid},
            ))
        else:
            findings.append(Finding(
                category="tone", zone="yellow", label="音色偏暗",
                severity="轻微",
                evidence=f"频谱质心 {centroid:.0f}Hz，高频成分偏少",
                metrics={"spectral_centroid_avg": centroid},
                suggestion="检查指甲长度或拨片角度，确保拨弦清晰",
            ))

    if attack > 0:
        if attack < 20:
            findings.append(Finding(
                category="tone", zone="green", label="起音干净利落",
                severity="无",
                evidence=f"起音时间 {attack:.0f}ms，拨弦动作清晰",
                metrics={"attack_time_avg_ms": attack},
            ))
        elif attack < 50:
            findings.append(Finding(
                category="tone", zone="green", label="起音正常",
                severity="无",
                evidence=f"起音时间 {attack:.0f}ms",
                metrics={"attack_time_avg_ms": attack},
            ))
        elif attack < 80:
            findings.append(Finding(
                category="tone", zone="yellow", label="起音偏慢",
                severity="轻微",
                evidence=f"起音时间 {attack:.0f}ms，拨弦不够干脆",
                metrics={"attack_time_avg_ms": attack},
                suggestion="注意拨弦时手指/拨片的动作要更快更果断",
            ))
        else:
            findings.append(Finding(
                category="tone", zone="red", label="起音过慢",
                severity="明显",
                evidence=f"起音时间 {attack:.0f}ms，音头模糊不清",
                metrics={"attack_time_avg_ms": attack},
                suggestion="这是基础问题：确保手指/拨片先接触琴弦再发力，而非划过去",
            ))

    return findings


def _diagnose_noise(af: Dict) -> List[Finding]:
    findings = []
    inharm = af.get("inharmonicity_ratio", 0)
    hf_noise = af.get("high_freq_noise_ratio", 0)

    if inharm > 0:
        if inharm < 0.10:
            findings.append(Finding(
                category="noise", zone="green", label="音色干净",
                severity="无",
                evidence=f"非谐波能量比 {inharm:.3f}，杂音极低",
                metrics={"inharmonicity_ratio": inharm},
            ))
        elif inharm < 0.20:
            findings.append(Finding(
                category="noise", zone="green", label="音色较干净",
                severity="无",
                evidence=f"非谐波能量比 {inharm:.3f}，杂音很少",
                metrics={"inharmonicity_ratio": inharm},
            ))
        elif inharm < 0.30:
            findings.append(Finding(
                category="noise", zone="yellow", label="存在一些杂音",
                severity="轻微",
                evidence=f"非谐波能量比 {inharm:.3f}，有一定杂音成分",
                metrics={"inharmonicity_ratio": inharm},
                suggestion="检查按弦手指是否完全按住品格，拨弦后手指是否碰到相邻弦",
            ))
        elif inharm < 0.45:
            findings.append(Finding(
                category="noise", zone="red", label="杂音较多",
                severity="中等",
                evidence=f"非谐波能量比 {inharm:.3f}，杂音占比偏高",
                metrics={"inharmonicity_ratio": inharm},
                suggestion="重点排查：按弦力度不足、指甲过长导致打品、左手离弦太慢",
            ))
        else:
            findings.append(Finding(
                category="noise", zone="red", label="杂音严重",
                severity="严重",
                evidence=f"非谐波能量比 {inharm:.3f}，杂音占主导",
                metrics={"inharmonicity_ratio": inharm},
                suggestion="从单音开始排查：每个音单独弹奏，确认按弦位置和力度都正确",
            ))

    if hf_noise > 0:
        if hf_noise > 0.12:
            findings.append(Finding(
                category="noise", zone="red", label="高频噪声过多",
                severity="明显",
                evidence=f"8kHz以上高频能量占比 {hf_noise:.1%}",
                metrics={"high_freq_noise_ratio": hf_noise},
                suggestion="高频噪声通常来自：指甲刮弦、拨片过硬、或录音环境问题",
            ))
        elif hf_noise > 0.07:
            findings.append(Finding(
                category="noise", zone="yellow", label="高频噪声偏高",
                severity="轻微",
                evidence=f"8kHz以上高频能量占比 {hf_noise:.1%}",
                metrics={"high_freq_noise_ratio": hf_noise},
            ))

    return findings


def _diagnose_pitch_errors(errors: List[Dict], stats: Dict) -> List[Finding]:
    findings = []
    if not errors:
        return findings

    # 分类统计
    pitch_errs = [e for e in errors if e.get("type") in ("pitch", "wrong_note", "extra_note", "missing_note", "dead_note")]
    rhythm_errs = [e for e in errors if e.get("type") in ("overlap", "pause", "rhythm_fault", "transition_gap")]
    severe = [e for e in errors if e.get("severity") == "high"]
    medium = [e for e in errors if e.get("severity") == "medium"]

    total_notes = stats.get("total_notes", len(errors) or 1)

    if pitch_errs:
        sev_count = sum(1 for e in pitch_errs if e.get("severity") == "high")
        med_count = sum(1 for e in pitch_errs if e.get("severity") == "medium")
        err_rate = len(pitch_errs) / max(total_notes, 1)

        if err_rate > 0.2 or sev_count >= 5:
            findings.append(Finding(
                category="pitch", zone="red", label="音准问题较多",
                severity="严重" if err_rate > 0.3 else "明显",
                evidence=f"{len(pitch_errs)}个音准错误（严重{sev_count}处、中等{med_count}处），错误率{err_rate:.0%}",
                metrics={"error_count": len(pitch_errs), "severe": sev_count, "medium": med_count, "error_rate": round(err_rate, 2)},
                suggestion="逐小节检查：找到具体错音位置，反复练习直到正确",
            ))
        elif err_rate > 0.1 or sev_count >= 2:
            findings.append(Finding(
                category="pitch", zone="yellow", label="存在音准错误",
                severity="中等" if sev_count >= 2 else "轻微",
                evidence=f"{len(pitch_errs)}个音准错误（严重{sev_count}处、中等{med_count}处）",
                metrics={"error_count": len(pitch_errs), "severe": sev_count, "medium": med_count, "error_rate": round(err_rate, 2)},
                suggestion="重点关注错误音符，放慢速度确保每个音都按对",
            ))

    if rhythm_errs:
        findings.append(Finding(
            category="rhythm", zone="yellow",
            label="存在节奏错误",
            severity="轻微" if len(rhythm_errs) < 3 else "中等",
            evidence=f"{len(rhythm_errs)}处节奏问题（时值不准或音符重叠/缺失）",
            metrics={"rhythm_error_count": len(rhythm_errs)},
            suggestion="用节拍器逐小节检查节奏型",
        ))

    # 如果没有任何错误
    if not pitch_errs and not rhythm_errs:
        findings.append(Finding(
            category="pitch", zone="green", label="未检测到明显音准/节奏错误",
            severity="无",
            evidence=f"basic-pitch 检测到 {total_notes} 个音符，未发现显著偏差",
            metrics={"total_notes": total_notes},
        ))

    return findings


def _diagnose_strumming(af: Dict, technique_segs: List[Dict]) -> List[Finding]:
    """扫弦质量诊断：结合扫弦检测时间窗和音频特征评估扫弦质量"""
    findings = []
    strum_segs = [s for s in technique_segs
                  if s.get("technique_id") == "strumming" and s.get("confidence", 0) >= 0.5]
    if not strum_segs:
        return findings

    inharm = af.get("inharmonicity_ratio", 0)
    hf_noise = af.get("high_freq_noise_ratio", 0)
    ioi_cv = af.get("ioi_cv", 0)

    # 扫弦时允许较高的非谐波能量（多弦同时发声天然有更多泛音交互）
    STRUM_INHARM_THRESHOLD = 0.35  # 扫弦容忍度高于单音(0.20)
    if inharm > 0 and inharm < STRUM_INHARM_THRESHOLD:
        findings.append(Finding(
            category="strumming", zone="green", label="扫弦音色干净",
            severity="无",
            evidence=f"扫弦中杂音水平正常，扫弦清晰度好",
            metrics={"inharmonicity_ratio": inharm},
        ))
    elif inharm >= STRUM_INHARM_THRESHOLD and inharm < 0.50:
        findings.append(Finding(
            category="strumming", zone="yellow", label="扫弦有杂音",
            severity="轻微",
            evidence="扫弦时杂音偏多，可能是触弦太深或指甲角度不对",
            suggestion="扫弦只需指甲轻轻划过弦面（约一半指甲），不要'砍'进琴弦。上扫时大拇指稍微侧一点减小刮擦声",
        ))
    elif inharm >= 0.50:
        findings.append(Finding(
            category="strumming", zone="red", label="扫弦杂音严重",
            severity="中等",
            evidence="扫弦杂音占比过高，音色不够干净",
            suggestion="放慢速度，先确保每次扫弦指甲接触深度一致。可以用闷音扫弦练习先找到'擦过'的感觉",
        ))

    # 扫弦节奏评估
    if ioi_cv > 0:
        if ioi_cv < 0.10:
            findings.append(Finding(
                category="strumming", zone="green", label="扫弦节奏均匀",
                severity="无",
                evidence="扫弦节奏稳定，上下摆动均匀",
                metrics={"ioi_cv": ioi_cv},
            ))
        elif ioi_cv < 0.18:
            findings.append(Finding(
                category="strumming", zone="yellow", label="扫弦节奏欠稳",
                severity="轻微",
                evidence="扫弦节奏偶尔不均匀，可能在换和弦或换扫弦型时手停了一下",
                suggestion="右手像节拍器一样持续摆动，即使不触弦手也在动。用节拍器辅助，先从单一扫弦型开始",
            ))
        else:
            findings.append(Finding(
                category="strumming", zone="red", label="扫弦节奏不稳",
                severity="中等",
                evidence="扫弦节奏忽快忽慢，节拍变异大",
                suggestion="降低速度（50-60BPM），只练右手持续上下摆动，左手不按弦",
            ))

    # 高频刮擦声（扫弦过深时指甲侧面刮弦产生）
    if hf_noise > 0.10:
        findings.append(Finding(
            category="strumming", zone="yellow", label="扫弦有指甲刮擦声",
            severity="轻微",
            evidence=f"高频噪声偏高，可能是扫弦角度偏平（指甲侧面刮弦）",
            suggestion="调整扫弦角度：下扫时食指甲稍微倾斜（约45°），不要让指甲侧面平行于琴弦",
        ))

    return findings


def _build_summary(findings: List[Finding]) -> str:
    """生成一段简洁的诊断摘要，供 AI prompt 使用。"""
    reds = [f for f in findings if f.zone == "red"]
    yellows = [f for f in findings if f.zone == "yellow"]
    greens = [f for f in findings if f.zone == "green"]

    lines = []

    if reds:
        lines.append(f"⚠️ 明确问题（{len(reds)}项）：{'; '.join(f.label for f in reds)}")
    if yellows:
        lines.append(f"⚡ 需关注（{len(yellows)}项）：{'; '.join(f.label for f in yellows)}")
    if not reds and not yellows:
        lines.append("✅ 未检测到明确的技术问题")
    if greens:
        lines.append(f"✓ 表现良好（{len(greens)}项）：{', '.join(f.label for f in greens)}")

    return "\n".join(lines)


def format_for_ai(diag: AudioDiagnosis) -> str:
    """将诊断结果格式化为 AI prompt 用的结构化文本。"""
    sections = []

    # 摘要区
    sections.append("## 音频诊断结果（Python 规则引擎自动分析）\n")
    sections.append(diag.summary)
    sections.append("")

    # 红区详情
    reds = [f for f in diag.findings if f.zone == "red"]
    if reds:
        sections.append("### 🔴 必须处理的问题\n")
        for f in reds:
            sections.append(f"**{f.label}**（严重程度：{f.severity}）")
            sections.append(f"- 证据：{f.evidence}")
            if f.suggestion:
                sections.append(f"- 建议方向：{f.suggestion}")
            sections.append("")

    # 黄区详情
    yellows = [f for f in diag.findings if f.zone == "yellow"]
    if yellows:
        sections.append("### 🟡 需要关注的问题\n")
        for f in yellows:
            sections.append(f"**{f.label}**（严重程度：{f.severity}）")
            sections.append(f"- 证据：{f.evidence}")
            if f.suggestion:
                sections.append(f"- 建议方向：{f.suggestion}")
            sections.append("")

    # 给 AI 的指令
    sections.append("### 给 AI 老师的指令")
    sections.append("1. 以上诊断由规则引擎自动生成，你应该在报告中逐条回应红区和黄区的问题")
    sections.append("2. 绿区的项目说明学员在这方面表现良好，可以给予肯定")
    sections.append("3. 评分时请参考诊断结果的严重程度和数量")
    sections.append("4. 你的核心任务是：用教学语言解释「为什么这是问题」和「怎么改」")
    sections.append("5. 报告中不要引用原始数据（dB、Hz、CV值），用学员能理解的描述替代")

    return "\n".join(sections)
