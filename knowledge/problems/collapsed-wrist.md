---
id: collapsed-wrist
type: problem
severity: moderate
difficulty: beginner
related_techniques: [wrist-posture, left-hand-arching]
related_problems: [flat-fingers, excessive-pressure, thumb-too-high]
tags: [手腕, 尺偏, 塌腕, 屈伸]
---

# 手腕塌陷或弯曲不当

## 表现
手腕角度偏离自然中立位。两种典型表现：
1. **手腕下沉（塌腕）**: 手腕过度弯曲 <100°，手掌贴近琴颈侧面，手指被迫放平
2. **手腕过度伸直**: 手腕 >170°，几乎完全伸直，缺乏灵活性

## AI检测/铁律

**触发条件**（满足任一条即标记为 collapsed-wrist）:
- 手腕弯曲角度 < 100°（手腕显著向内弯折，手背与前臂形成明显V形）
- 手腕过度伸直 > 170°（手腕僵直，丧失弹性活动空间）

**排除条件**（以下情况不应标记为 collapsed-wrist）:
- 正在演奏高把位（5品以上）时手腕自然弯曲稍大是正常的
- 正在使用拇指扣弦技巧时手腕位置会自然调整
- 手腕弯曲在 100-130° 之间属于正常调整范围，不需要标记

**与持琴姿势的关联**:
- 如果持续检测到 collapsed-wrist，优先检查坐姿和吉他角度，而非单纯纠正手腕

## 原因
1. 持琴姿势不正确 — 琴身角度、高度不合适
2. 坐姿不正 — 驼背、耸肩导致手臂位置偏差
3. 吉他背带调节不当（站立演奏时）

## 因果链
坐姿不正/吉他太低 → 手腕被迫弯曲去够琴颈 → 手腕塌陷（collapsed-wrist）→ 手指被迫放平（flat-fingers）→ 按弦费力/杂音
- **上游根因**: 坐姿、吉他高度/角度 — collapsed-wrist 几乎总是全身姿势问题的局部表现
- **下游后果**: flat-fingers, thumb-too-high — 手腕塌了以后，手指和拇指都会自动调整到错误位置来补偿

## 正确做法
- 手腕保持自然弧度，前臂、手腕、手背成近似直线
- 尺偏约 15°（小指侧略低于拇指侧）
- 手掌与琴颈保持适当距离（约可塞入一个乒乓球的空间）
- 手腕既不过度弯曲也不完全僵直，保持弹性

## 练习方法
1. **手腕位置检查**: 每次练习前，检查手腕弧度——应能自然地在手腕处放一支笔而不掉落
2. **镜前自检**: 侧对镜子，观察前臂→手腕→手背的连线是否平滑
3. **滑动练习**: 在同一弦上做品位滑动，保持手腕弧度不变
4. **吉他位置调整**: 尝试将琴头抬高 5-10°，通常能改善手腕角度

## 预期改善周期
调整持琴姿势后通常即刻见效。形成习惯需要 1-2 周。

## 参考来源
- Larsen, C. J. (2016). "Ergonomic Considerations for the Classical Guitarist" — 手腕 15° 尺偏最优
- Iznaola, R. *Summa Kitharologica* — 吉他生物力学
- This is Classical Guitar — Proper Left Hand Position
