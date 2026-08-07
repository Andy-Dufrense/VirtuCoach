"""
VirtuCoach - 吉他智能品位分配模块
DP/Viterbi 算法：给定 MIDI 音符序列，计算最优弦+品位分配

基于 gtrsnipe 的评分算法思路，自实现轻量版。
- O(N * S^2) 复杂度，S=6 弦，极快
- 时间感知：间隔 >2s 的音符间不施加移动代价
- 和弦约束：重叠音符的品位跨度 ≤ 4 品
- 奖励空弦、惩罚高把位
"""

from typing import List, Dict, Tuple

GUITAR_STRINGS = [
    (1, "E4", 64),   # 1弦 (最细)
    (2, "B3", 59),   # 2弦
    (3, "G3", 55),   # 3弦
    (4, "D3", 50),   # 4弦
    (5, "A2", 45),   # 5弦
    (6, "E2", 40),   # 6弦 (最粗)
]

CHINESE_NUM = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六"}
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAX_FRET = 22
MAX_STRETCH = 4          # 同一手位内最大品位跨度
TIME_GAP_THRESHOLD = 2.0  # 超过2秒不计移动代价

# DP 权重
FRET_COST = 0.1          # 每品基础代价
OPEN_BONUS = -0.8        # 空弦奖励
MOVE_COST = 1.2          # 手移动每品
STRING_COST = 0.5        # 跨弦每根
STRETCH_PENALTY = 50.0   # 和弦跨度过大（硬约束，远大于其他代价）
SWEET_MAX = 12           # 甜区上限
HIGH_BASS_PENALTY = 3.0  # 粗弦高把位惩罚（5弦>15品, 6弦>12品不现实）
HAND_ZONE_WEIGHT = 2.0   # 手区偏置权重（每偏离1品增加的代价）
HAND_ZONE_HARD_LIMIT = 5  # 手区<4品时禁止分配的品位上限（除非该音高在≤5品无解）


def _note_name(midi: int) -> str:
    o = (midi // 12) - 1
    return f"{_NOTE_NAMES[midi % 12]}{o}"


def _guitar_desc(s_num: int, fret: int, midi: int, capo: int = 0) -> str:
    cn = CHINESE_NUM.get(s_num, str(s_num))
    if capo > 0 and fret >= capo:
        rel_fret = fret - capo
        if rel_fret == 0:
            return f"{cn}弦空弦"
        return f"{cn}弦{rel_fret}品"
    return f"{cn}弦{fret}品"


def assign_guitar_positions(notes: List[Dict], capo: int = 0,
                            hand_zone_center: float = None) -> List[Dict]:
    """
    Viterbi DP: 为音符序列分配最优弦+品位。

    输入 notes 需包含: pitch (int), start_time (float), end_time (float)
    hand_zone_center: 可选，MediaPipe 检测到的左手大致所在品数（如 5.0 表示第5品附近），
                      用于软偏置 DP 选择。None 表示无手部数据。
    原地修改并返回, 添加: string, fret, guitar_position
    """
    if not notes:
        return notes

    use_hand_zone = hand_zone_center is not None and hand_zone_center > 0
    S = len(GUITAR_STRINGS)
    N = len(notes)

    # 预计算每个音符的所有有效 (弦索引, 品位)
    valid = []
    for n in notes:
        pitch = int(n.get("pitch", 0))
        opts = []
        for si, (_, _, smidi) in enumerate(GUITAR_STRINGS):
            f = pitch - smidi
            if 0 <= f <= MAX_FRET:
                opts.append((si, f))
        if not opts:
            best = min(range(S), key=lambda si: abs(pitch - GUITAR_STRINGS[si][2]))
            f = pitch - GUITAR_STRINGS[best][2]
            opts = [(best, max(0, min(MAX_FRET, f)))]
        valid.append(opts)

    # DP: dp[i][k] = (total_cost, prev_k)
    dp = []
    for i in range(N):
        opts = valid[i]
        row = []
        t_i = notes[i].get("start_time", 0)
        prev_end = notes[i-1].get("end_time", 0) if i > 0 else 0
        gap = t_i - prev_end  # >0 = 有空隙, <0 = 重叠

        for k, (si, fret) in enumerate(opts):
            # 基础代价
            base = FRET_COST * fret
            if fret == 0:
                base += OPEN_BONUS
            if fret <= SWEET_MAX:
                base -= 0.2  # 甜区微奖励
            # 粗弦高把位惩罚：6弦>12品、5弦>15品 极少使用，加惩罚避免误分配
            if (si + 1 == 6 and fret > 12) or (si + 1 == 5 and fret > 15):
                base += HIGH_BASS_PENALTY
            # 手区偏置：偏离 MediaPipe 检测到的手实际位置越远，代价越高
            if use_hand_zone:
                zone_dist = abs(fret - hand_zone_center)
                base += HAND_ZONE_WEIGHT * zone_dist
                # 硬约束：手区<4品（低把位）时，禁止分配>5品的品位
                if hand_zone_center < 4.0 and fret > HAND_ZONE_HARD_LIMIT:
                    # 检查该音高是否在≤5品有解
                    low_fret_possible = False
                    for check_si, (_, _, open_midi) in enumerate(GUITAR_STRINGS):
                        check_fret = notes[i].get("pitch", 0) - open_midi
                        if 0 <= check_fret <= HAND_ZONE_HARD_LIMIT:
                            low_fret_possible = True
                            break
                    if low_fret_possible:
                        base += 20.0  # 强惩罚：低把位不该用高品位

            if i == 0:
                row.append((base, -1))
                continue

            best = (float("inf"), -1)
            for pk, (psi, pfret) in enumerate(valid[i-1]):
                prev = dp[i-1][pk][0]

                if gap > TIME_GAP_THRESHOLD:
                    # 间隔大，不计移动代价
                    trans = 0
                elif gap < 0:
                    # 音符重叠 → 和弦约束
                    span = abs(fret - pfret)
                    if span > MAX_STRETCH:
                        trans = STRETCH_PENALTY
                    else:
                        trans = MOVE_COST * abs(fret - pfret) + STRING_COST * abs(si - psi)
                else:
                    # 正常间隔
                    trans = MOVE_COST * abs(fret - pfret) + STRING_COST * abs(si - psi)

                c = prev + base + trans
                if c < best[0]:
                    best = (c, pk)

            row.append(best)
        dp.append(row)

    # 回溯
    best_k = min(range(len(dp[-1])), key=lambda k: dp[-1][k][0])
    path = []
    k = best_k
    for i in range(N - 1, -1, -1):
        si, fret = valid[i][k]
        path.append((si, fret))
        k = dp[i][k][1] if i > 0 else -1
    path.reverse()

    for i, (si, fret) in enumerate(path):
        snum = GUITAR_STRINGS[si][0]
        notes[i]["string"] = snum
        notes[i]["fret"] = fret
        notes[i]["guitar_position"] = _guitar_desc(snum, fret, int(notes[i]["pitch"]), capo)

    return notes


def batch_assign(notes: List[Dict], window: int = 150, capo: int = 0,
                 hand_zone_center: float = None) -> List[Dict]:
    """
    分段 DP，避免超长序列尾部累积代价畸变。
    每段独立计算，段边界重置手位置。
    """
    if len(notes) <= window:
        return assign_guitar_positions(notes, capo=capo, hand_zone_center=hand_zone_center)

    for start in range(0, len(notes), window):
        assign_guitar_positions(notes[start:start + window], capo=capo,
                               hand_zone_center=hand_zone_center)

    return notes
