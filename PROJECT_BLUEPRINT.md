# VirtuCoach 项目蓝图

> 编写日期: 2026-07-13
> 基于: 竞品调研(国内外15+产品)、代码审查(21个Gap发现)、3轮技术方案论证
> 定位: AI 吉他教师 — 用户上传演奏视频，AI 独立给出专业评价和改进建议
> 目标: 最专业、最精准的 AI 吉他教师。不需要真人兜底，AI 本身就是权威。

---

## 一、这个项目能做到什么

### 1.1 核心能力

用户上传一段吉他演奏视频 → 系统自动完成：

| 能力 | 说明 | 精度目标 |
|------|------|----------|
| **音频分析** | 节拍稳定性、动态范围、音色清晰度、起音质量、泛音丰富度 | 定量指标，可验证 |
| **3D 手型分析** | 21点真实3D坐标 → 每指MCP+PIP+DIP关节角度 → 指尖垂直度 → 手腕姿态 | 与量角器误差<7° |
| **VL 视觉验证** | Qwen-VL 针对问题帧做 AI 视觉确认+详细文字描述 | 上下文注入，减少幻觉 |
| **参考图对比** | 检索相似手型的正确参考图 → 并排对比 → AI 指出差异 | 多图 VL 对比分析 |
| **RAG 知识检索** | 根据检测到的问题，从知识库检索教学方案 → 注入报告 | 专业度对标真人教师 |
| **AI 专业报告** | DeepSeek 生成结构化报告：总评→音频→手型→练习建议 | 对话式、可追问 |
| **持续对话** | 用户可以追问任何报告中的问题，AI 结合 RAG 知识库回答 | 不限轮次 |

### 1.2 不做的事

- ~~跟弹练习~~ — 已移除
- ~~实时音高反馈~~ — 不是实时引擎
- 报告里不说「建议找老师确认」— AI 自己就是老师，必须给出肯定结论
- 不确定的结论要说「这部分信心不足」，但不说「建议咨询真人」

### 1.3 与竞品的差异

| | 小叶子 | ROLI | Yousician | **VirtuCoach** |
|---|---|---|---|---|
| 音频分析 | 99%音准 | ✅ | ✅ | 多维度特征提取 |
| 手型分析 | ❌ 纯音频 | 红外硬件 | ❌ 纯音频 | **3D 视觉（纯软件）** |
| 参考图对比 | ❌ | ❌ | ❌ | **✅ 独有** |
| 教学知识库 | 曲库为主 | ChatGPT 通用 | 课程体系 | **✅ RAG 专项知识库** |
| 定位 | 辅助练习 | 陪练伴侣 | 自学工具 | **AI 吉他教师** |
| 硬件要求 | 无 | $299+ | 无 | 无 |

---

## 二、怎么实现的

### 2.1 完整技术架构

```
用户上传视频 (POST /api/analyze)
    │
    ├──→ [1] 音频分离 (demucs，可选)
    │       └→ 去人声 → 纯乐器音轨
    │
    ├──→ [2] 信号分析层 (Python, 确定性计算)
    │    │
    │    ├── 音频特征提取 (audio_analyzer.py)
    │    │   ├ basic-pitch → 音符事件 (onset/offset/pitch/velocity)
    │    │   ├ 节拍稳定性: 变异系数、局部加速检测
    │    │   ├ 动态范围: velocity分布、dB范围
    │    │   ├ 音色分析: 频谱质心、起音速度、泛音衰减率
    │    │   └ 杂音检测: 高频噪声占比、非谐波能量
    │    │
    │    └── 3D 手型分析 (video_processor.py)
    │        ├ MediaPipe HandLandmarker → 21点真实3D坐标(x,y,z,米)
    │        ├ 每指3关节: MCP + PIP + DIP → 真实空间角度
    │        ├ 指板平面拟合 → 指尖垂直度
    │        ├ 手腕姿态: 尺偏角度 + 屈伸角度
    │        ├ 时间稳定性: 连续帧滑动窗口，瞬态过滤
    │        └ 输出: 量化特征向量 + 问题帧排序
    │
    ├──→ [3] VL 视觉验证层 (vision_analyzer.py)
    │    ├ 智能帧选择: 从信号层获取高分问题帧
    │    ├ 上下文注入: "本帧检测到食指 PIP 158°(偏平)，请验证"
    │    ├ Qwen-VL 多图分析: 参考图→用户帧→对比判断
    │    └ 输出: 结构化手型问题描述 + 置信度
    │
    ├──→ [4] RAG 知识检索层 (新增)
    │    ├ 对检测到的问题做 embedding
    │    ├ 搜索教学知识库 (techniques + problems + exercises)
    │    └ 返回: 原因分析 + 练习方法 + 预期周期 + 参考图片
    │
    └──→ [5] AI 报告生成层 (deepseek_agent.py)
         ├ DeepSeek 接收: 量化指标 + VL验证 + RAG知识
         ├ 生成: 总评 → 音频分析 → 手型分析 → 练习建议
         └ 用户可追问 → RAG 注入对话上下文 → AI 回答
```

### 2.2 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| 手部追踪 | MediaPipe HandLandmarker (3D) | 医学验证 r=0.84-0.99，开源免费 |
| 视觉AI | Qwen-VL-Max | 中文最优，支持多图对比，成本可控 |
| 音频转录 | basic-pitch (Spotify) | 完整音符事件含velocity |
| 报告生成 | DeepSeek-V3 | 中文最优，性价比最高 |
| RAG嵌入 | text-embedding-3-small / bge-large-zh | 中文语义搜索 |
| RAG向量库 | ChromaDB / LanceDB | 轻量级，嵌入Python进程 |
| 参考图库 | 已有 SQLite + 磁盘存储 | 路径穿越防护，vector_search 就绪 |
| 后端框架 | FastAPI | 已有，异步友好 |
| 前端 | 原生 JS/HTML/CSS | 已有，无需改动 |

### 2.3 手型分析的可靠性基础

MediaPipe 角度测量已被多个同行评议医学研究验证：

- **与医用角度尺对比**: 均值误差 2~7°，相关系数 r=0.84~0.97
- **加 ML 校正后**: r 提升至 0.94~0.99
- **最佳场景**: 手指伸展状态（吉他按弦恰好是这种场景）
- **已知局限**: MCP 关节握拳时误差大(14~25°)，但吉他按弦是伸展态

吉他手型生物力学有权威文献可校准：

- Ricardo Iznaola《Summa Kitharologica》— 吉他生物力学圣经
- Larsen (2016): 手腕 15° 尺偏为最优
- 2D 生物力学模型(2013): 各和弦肌腱受力可量化

### 2.4 竞品市场确认（调研结论）

- **国内**: 小叶子、西西魔法等全是纯音频（99%音准但无手型）。小叶子没有 AI 手型功能。
- **国际**: ROLI 做视觉但需 $299 专用红外硬件。Yousician 是纯音频+LLM对话。Google 没有音乐教育产品。
- **结论**: VirtuCoach 的「纯软件 + 3D手型分析」在国内 B2C 市场是**唯一的**。

---

## 三、后续怎么做

### 第一阶段：核心管线改造（1-2 周）

#### Step 1: MediaPipe 升级到 3D

```
文件: backend/video_processor.py
改动:
  - 安装新版 mediapipe (0.10.x+)，下载 hand_landmarker.task
  - 替换 mp.solutions.hands → mp.tasks.vision.HandLandmarker
  - 使用 world_landmarks (真实3D坐标, 单位米)
  - _calc_angles() → 基于真实空间坐标计算 MCP+PIP+DIP 三关节
  - 新增: 指板平面拟合 → 指尖垂直度指标
  - 新增: z_depth 从相对值 → 真实毫米距离
  - _detect_issues() → 阈值用生物力学文献校准
  - 帧评分 + 滑动窗口平滑
  - 启用 _draw_hand_annotations
验证: 用已知角度的手部照片测试，与量角器读数对比
```

#### Step 2: 信号分析层改造

```
文件: backend/audio_analyzer.py
改动:
  - 新增 analyze_audio_features() → 5维度定量指标
    ├ 节拍稳定性: 变异系数 CV = std(IOI) / mean(IOI)
    ├ 动态范围: velocity P5/P50/P95, dB range
    ├ 音色: 频谱质心, 泛音衰减斜率
    ├ 起音质量: 起音上升时间, 起音清晰度
    └ 杂音: 非谐波能量比, 高频噪声占比
  - 保留 basic-pitch 转录用于定位
  - 不再依赖 DeepSeek 猜测错误

文件: backend/deepseek_agent.py
改动:
  - _build_user_prompt() → 输入改为结构化 metrics JSON
  - 不再传 raw notes 列表给 LLM
  - 新增 _build_freeplay_prompt() → 描述性语言（不纠错）
```

#### Step 3: RAG 知识库搭建

```
新增: backend/knowledge_db.py

知识库结构:
  knowledge/
  ├── techniques/    (10+ 技巧条目)
  ├── problems/      (10+ 常见问题条目)
  └── exercises/     (10+ 练习方法条目)

每个条目格式:
  ---
  id, type, severity, difficulty
  related_techniques, related_problems
  ---
  # 标题
  ## 表现 / ## 原因 / ## 正确做法
  ## 练习方法 / ## 预期改善周期 / ## 参考图片

功能:
  - 启动时加载所有 .md 文件
  - 用 bge-large-zh 或 DeepSeek embedding 做向量化
  - ChromaDB 存储 + 语义搜索
  - retrieve(problem_type, top_k=3) → 返回相关知识条目
  - 知识条目注入 deepseek_agent prompt
  - ask_question() 对话时同样检索
```

### 第二阶段：报告与体验升级（1 周）

#### Step 4: LLM Prompt 重构

```
文件: backend/deepseek_agent.py
改动:
  - 系统 prompt: 角色从「分析者」→「资深吉他教师」
  - 用户 prompt: 接收结构化 metrics + RAG检索结果
  - Few-shot 范例: 好的/差的/不确定的 各1个标注样例
  - 报告风格: 对话式，先肯定再建议
  - nHand 机制: 保留（已验证有效）
  - 置信度: 每个结论附带 high/medium/low
```

#### Step 5: 参考图上传分析流程接入 RAG

```
文件: backend/main.py (POST /api/analyze-hand-posture)
改动:
  - 上传参考图 → Qwen-VL分析 → 提取手型特征
  - 将分析结果存为知识条目（也做 embedding）
  - 后续用户手型匹配时，既匹配图片也匹配知识文字

文件: backend/reference_db.py
改动:
  - 已有 vector_search() 基于 MediaPipe 63维向量
  - 新增 semantic_search() 基于文本 embedding
  - 双通道检索: 图片相似度 + 问题类型语义匹配
```

### 第三阶段：验证与上线（1 周）

#### Step 6: 端到端测试

```
准备测试集: 10个视频覆盖
  - 好: 3个（专业/熟练业余/正确但平淡）
  - 中: 4个（轻微塌指/节奏微不稳/音色偏闷/拇指偏高）
  - 差: 2个（明显手型错误/节奏混乱）
  - 边界: 1个（无手/静音）

每个视频跑新旧两套流水线，对比:
  - 手型检测: 关节数 5→15, 问题定位准确率
  - 音频分析: 描述性 vs 纠错性语言
  - 报告质量: 人工评审（找吉他老师打分）
  - RAG 检索: 知识匹配准确率
```

#### Step 7: 清理跟弹模块

```
需要移除/禁用的文件:
  - backend/practice_engine.py → 移出路由，保留代码以备后用
  - backend/tab_parser.py → 同上
  - frontend/index.html → 移除练习室 section
  - frontend/app.js → 移除 WebSocket 练习相关代码
  - backend/main.py → 移除 /ws/practice, /api/tabs, /api/parse-tab-image

需要保留的:
  - 谱面解析的预处理管道 (vision_analyzer._preprocess_tab_image)
    → 移到独立模块，后续做参考谱面上传用
```

---

## 四、关键文件清单

| 文件 | 角色 | 改动幅度 |
|------|------|----------|
| `backend/video_processor.py` | 3D 手型分析引擎 | 🔴 大改 |
| `backend/vision_analyzer.py` | VL 视觉验证 + 参考图对比 | 🟡 中改 |
| `backend/audio_analyzer.py` | 音频多维特征提取 | 🟡 中改 |
| `backend/deepseek_agent.py` | LLM 报告生成 + RAG 集成 | 🟡 中改 |
| `backend/knowledge_db.py` | **新增**: RAG 知识库模块 | 🟢 新增 |
| `backend/reference_db.py` | 双通道检索扩展 | 🟡 中改 |
| `backend/main.py` | 流水线整合 + 清理跟弹模块 | 🟡 中改 |
| `knowledge/*.md` | **新增**: 吉他教学知识条目 | 🟢 新增 |
| `frontend/index.html` | 移除练习室 UI | 🔵 轻改 |
| `frontend/app.js` | 移除跟弹逻辑 | 🔵 轻改 |

---

## 五、数据积累策略

```
现在 ── 文献标准 + 已有参考图库
        └→ 校准阈值、构建RAG知识库、few-shot prompt

上线后 ── 用户使用数据
        └→ 收集角度误差分布、改进阈值
        └→ 识别高频问题类型、补充知识条目

有量后 ── 人工审核标注
        └→ 筛选高质量分析 → 请吉他老师审核
        └→ 训练专用手型分类器
        └→ 微调 LLM prompt
```
