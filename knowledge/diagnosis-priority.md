---
id: diagnosis-priority
type: system
description: 手型问题诊断优先级、因果合并规则和分层反馈策略——AI检测到多个问题时如何排序、合并和输出
related_techniques: [basic-fretting, left-hand-arching, wrist-posture, anchor-finger]
tags: [诊断, 优先级, 因果链, 问题合并, 反馈策略, AI引擎]
---

# 诊断优先级与因果合并规则

## 概述

当 AI 同时检测到多个手型问题时（例如 collapsed-wrist + flat-fingers + string-buzzing），不应该把所有问题都甩给学员。这会让初学者不知所措，也不符合专业教学逻辑——因为 flat-fingers 本身就是 collapsed-wrist 的下游后果，单独纠正 flat-fingers 不解决根因，问题会反复出现。

本文档定义了问题之间的因果依赖关系、合并规则和分层反馈策略，是 AI 诊断引擎的核心规则。

## 因果依赖图

### 左手因果链

```
sitting-posture (身体层面, 根因中的根因)
  │
  └─→ collapsed-wrist (手腕层面)
        │
        ├─→ thumb-too-high ────→ flat-fingers ──→ excessive-pressure
        │     (拇指包覆)            (手指放平)        (力度代偿)
        │                              │
        ├─→ thumb-gripping-too-tight ─┤──→ string-buzzing (闷音/打品)
        │     (拇指捏紧)                │
        ├─→ thumb-no-support ────────→│──→ pinky-flying (小指飞离)
        │     (拇指无支撑)              │
        └─→ fingers-bunched-together ─┘──→ palm-perpendicular
              (手指挤在一起)                  (手掌垂直)
```

**关键依赖关系:**
- `collapsed-wrist` 是 7 个问题的上游根因——如果检测到手腕塌陷，下面的大多数手指问题是连锁反应而非独立问题
- `thumb-gripping-too-tight` 是唯一的 severe 级问题——出现时必须作为最高优先级对待
- `flat-fingers` 是最常见的"汇总症状"——它几乎可以追溯到任何上游问题
- `string-buzzing` 和 `excessive-pressure` 是"终末症状"——它们永远不是根因，总是由手指姿态或力度问题引起

### 右手因果链

```
right-hand-off-soundhole (手臂位置错误, 独立根因)
  │
  └─→ 音色发闷/发薄

right-hand-tension (右手整体紧张, 根因)
  │
  ├─→ picking-uneven (力度不均)
  ├─→ strumming-rhythm (扫弦节奏不稳)
  └─→ right-hand-fingers-not-curved (手指不弯曲)
```

**关键依赖关系:**
- 左手问题和右手问题**不应合并**——它们是独立的系统，分开报告
- `right-hand-tension` 是右手问题的上游根因
- `right-hand-off-soundhole` 是独立问题，不与其他右手问题合并

### 交叉影响（左手→右手）

```
左手问题（注意力被占用、手部整体紧张）
  │
  └─→ 右手也跟着紧张 → right-hand-tension 的间接诱因
```

但这种交叉影响较弱——在诊断报告中除非有明确证据（左手高度紧张 + 右手也检测到紧张），否则不合并。

## 优先级分级

### Tier 0 — 致命（Fatal）
演奏无法正常进行，必须立即纠正。

| 问题 | 触发条件 |
|------|---------|
| `thumb-gripping-too-tight` (重度) | 手掌锁死 + 所有 PIP < 100° |
| `string-buzzing` (重度，且 >50% 音符受影响) | 大面积杂音/打品 |

### Tier 1 — 根因（Root Cause）
修复它能连锁解决多个下游问题，是最高诊断杠杆。

| 问题 | 判断依据 |
|------|---------|
| `sitting-posture` | 全身姿势问题是所有手型问题的起点 |
| `collapsed-wrist` | 手腕塌陷是 7 个手指问题的上游 |

### Tier 2 — 结构（Structural）
手型的基础支撑问题，不直接导致杂音但从根本上影响手型。

| 问题 | 判断依据 |
|------|---------|
| `thumb-too-high` | 拇指位置错误导致手指被迫放平 |
| `thumb-no-support` | 拇指未提供支撑，手指代偿 |
| `right-hand-tension` | 右手紧张连锁影响拨弦和扫弦 |
| `right-hand-off-soundhole` | 右手位置偏离共振区 |

### Tier 3 — 表现（Symptom）
这些"问题"通常是 Tier 1-2 的下游表现，直接纠正它们而不解决根因会反复。

| 问题 | 常见上游根因 |
|------|------------|
| `flat-fingers` | collapsed-wrist, thumb-too-high, thumb-gripping-too-tight |
| `too-vertical-fingers` | collapsed-wrist, thumb-no-support |
| `fingers-bunched-together` | collapsed-wrist, thumb-gripping-too-tight |
| `palm-perpendicular` | collapsed-wrist |
| `pinky-flying` | 手指独立性不足, flat-fingers |
| `right-hand-fingers-not-curved` | right-hand-tension |

### Tier 4 — 补偿（Compensation）
身体的代偿反应，不是问题本身而是问题导致的结果。

| 问题 | 常见上游根因 |
|------|------------|
| `excessive-pressure` | flat-fingers, thumb-gripping-too-tight |
| `picking-uneven` | right-hand-tension |
| `strumming-rhythm` | right-hand-tension |

## 合并规则

### 规则 1: 根因吸纳症状
如果同时检测到根因（Tier 1-2）和其下游症状（Tier 3-4），则**只报告根因**，症状作为根因的"表现"字段列出。

**示例:**
- 检测到 `collapsed-wrist` + `flat-fingers` + `string-buzzing`
- → 只报告 `collapsed-wrist`，附加描述: "表现: 手指放平、按弦有杂音"
- → 纠正建议聚焦于手腕调整，附带"手腕调好后手指放平的问题自然改善"

### 规则 2: 多症状同根合并
如果检测到多个 Tier 3-4 症状但它们的共同根因未被检测到（可能因为根因检测不够灵敏），将症状合并为一个条目，按最可能的根因给出建议。

**示例:**
- 检测到 `flat-fingers` + `fingers-bunched-together` + `excessive-pressure`
- 但 `collapsed-wrist` 未达到检测阈值
- → 合并为一条: "左手多个手指姿态问题"，建议检查手腕是否塌陷

### 规则 3: 拇指问题取最严重
如果同时检测到多个拇指相关的问题（thumb-too-high, thumb-gripping-too-tight, thumb-no-support），只报告最严重的一个（按严重程度: gripping-too-tight > no-support > too-high）。

### 规则 4: 左右手分离
左手问题和右手问题不互相合并——分别报告，左手优先（因为初学者左手问题更普遍也更影响演奏）。

### 规则 5: 最大输出数
- **初学者**: 最多 1 个问题（根因优先）
- **中级**: 最多 2 个问题（1 根因 + 1 独立问题）
- **高级**: 最多 3 个问题
- 如果检测到致命问题（Tier 0），跳过数量限制直接报告

### 规则 6: 致命问题独占
如果检测到致命问题（Tier 0），所有其他非致命问题降级为"次要问题"备注，不独立报告。

## 问题ID映射

从 `video_processor.py` 输出的中文描述文本到 `problems/*.md` 知识条目 id 的映射：

| 中文描述关键特征 | 问题 ID | Tier |
|-----------------|---------|------|
| 拇指在琴颈后方 + 虎口闭合/捏紧/力度过大 | `thumb-gripping-too-tight` | 0 |
| 杂音/闷音/打品（大面积、严重影响） | `string-buzzing` | 0 |
| 坐姿/持琴姿势/身体弯曲 | `sitting-posture` | 1 |
| 手腕塌陷/内扣/角度<140° | `collapsed-wrist` | 1 |
| 拇指过高/包覆琴颈/从上方伸出 | `thumb-too-high` | 2 |
| 拇指未接触琴颈/悬空/无支撑 | `thumb-no-support` | 2 |
| 右手整体紧张/僵硬/手指锁死 | `right-hand-tension` | 2 |
| 右手不在音孔上方/偏离共振区 | `right-hand-off-soundhole` | 2 |
| 手指放平/指腹按弦/PIP>155° | `flat-fingers` | 3 |
| 手指竖直/MCP<50°/手指像筷子 | `too-vertical-fingers` | 3 |
| 手指挤在一起/间距<2.5cm/缩 | `fingers-bunched-together` | 3 |
| 手掌贴琴颈/手掌垂直 | `palm-perpendicular` | 3 |
| 小指飞离/翘起/PIP>160° | `pinky-flying` | 3 |
| 右手手指太直/PIP>160°/无弯曲 | `right-hand-fingers-not-curved` | 3 |
| 按弦力度过大/过度用力 | `excessive-pressure` | 4 |
| 右手拨弦力度不均 | `picking-uneven` | 4 |
| 扫弦节奏不稳 | `strumming-rhythm` | 4 |

## 分层反馈策略

### 初学者反馈模板
```
检测到 1 个核心问题: [根因问题]
表现: [1-2 个下游症状]
建议: [根因纠正方法 + 1 个具体练习]
时间: [问题出现的时间点]
```

**原则**: 
- 永远只给 1 个反馈
- 用生活化比喻（"你的手腕像在捏尖叫鸡"）
- 给出 1 个可立即执行的练习

### 中级反馈模板
```
检测到 2 个问题:
1. [根因问题] — 这导致了 [症状表现]
2. [独立问题]（与问题 1 不相关）
建议: [根因纠正] + [独立问题纠正]
```

**原则**:
- 最多 2 个反馈
- 说明根因和症状的关系
- 给出技术性但可理解的建议

### 高级反馈模板
```
检测到以下问题:
1. [最高优先级问题] (Tier X) — [角度/数据]
2. [次要问题] (Tier Y) — [角度/数据]
3. [独立问题] (Tier Z) — [角度/数据]

诊断分析: [因果链分析]
建议: [具体技术建议 + 练习组合]
```

**原则**:
- 可以给 2-3 个反馈
- 包含具体角度数据和阈值信息
- 给出练习组合而非单一练习

## 特殊情况处理

### 情况 1: 正在演奏大横按
横按期间拇指需要提供对向夹力——`thumb-gripping-too-tight` 的检测阈值放宽。如果正在横按且检测到的拇指力度在正常横按范围内 → 不合并也不标记。

### 情况 2: 正在演奏推弦/揉弦
推弦和揉弦会临时改变手指角度和力度——所有角度阈值放宽 15°，力度阈值放宽 20%。

### 情况 3: 高把位演奏（12品以上）
品距缩小导致手指间距自然变小——手指聚集（fingers-bunched-together）的阈值放宽 30%。

### 情况 4: 非常短的问题片段（<0.5秒）
如果一个问题只出现在不到 0.5 秒的时间窗口内（可能是瞬间的调整），不参与合并，直接丢弃。

## AI 实现要点

1. **映射优先**: 先将所有 issue 的中文描述映射为问题 ID，映射失败的保持原样（不参与合并但不丢弃）
2. **因果合并**: 按因果依赖图自底向上合并——从 Tier 4 向 Tier 3 合并，Tier 3 向 Tier 1-2 合并
3. **去重**: 同手同问题 ID 只保留置信度最高的一个
4. **排序**: 合并后按 Tier 升序排序（Tier 0 最先），同 Tier 按置信度降序
5. **截断**: 按学员等级的 max 限制截取，Tier 0 不参与截断
6. **保留原始**: 合并后的条目保留原始 issue 的时间戳和截图引用，确保前端仍能展示对应画面

## 参考来源
- 现有 17 个 problem 条目的因果链分析
- 吉他教学诊断方法论
- JustinGuitar — Problem Solving Approach
- Pumping Nylon (Scott Tennant) — Troubleshooting Left Hand Issues
