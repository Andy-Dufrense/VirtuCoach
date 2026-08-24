# VirtuCoach-Graduate — AI 吉他陪练教练

毕业设计项目：基于**多模态融合**（音频 + 视觉 + LLM）的吉他演奏智能诊断与陪练系统。上传一段弹奏视频，系统自动分析音准、节奏、技巧、手型，生成分维度评分与改进建议，并支持 AI 对话答疑、练习记录追踪与管理员数据面板。

## 核心功能

| 模块 | 说明 |
|---|---|
| 视频分析 | 音频（Basic Pitch 音符转录 + 和弦识别 pcset 匹配 + 扫弦/分解技巧分类）× 视觉（MediaPipe 63 维手部关键点 + 通义千问 VL 手型诊断）双通道融合评分 |
| 手型检查 | 上传手型照片/视频，对照参考图库（RAG 检索）给出针对性纠错 |
| AI 答疑 | DeepSeek 对话，内置幻觉三防线（结构化清洗器 + 知识库 grounding + 事后校验） |
| 用户系统 | JWT 登录、90s 心跳单会话机制（顶号下线）、游客试用 → 登录后自动保存记录 |
| 练习记录 | 历史评分曲线、双记录对比、练习时长统计 |
| 管理面板 | 数据概览 / 反馈工单 / 用户管理（搜索 + 在线筛选 + 列排序）/ 参考图管理 / 五维数据分析，全部端点带 admin 鉴权 |

## 技术架构

```
frontend/                  多页面原生 JS（app.js / hand-check.js / admin.html ...）
backend/
  main.py                  FastAPI 入口（薄路由层，端口 1218）
  routers/                 analysis / auth / chat / feedback / admin 等路由
  services/                AnalysisService 编排层（音视频并行 → 融合评分 → 报告生成）
  pipeline/                分析流水线组件
  audio_analyzer.py        音频分析核心（转录/和弦/扫弦/节奏）
  chord_analyzer.py        和弦识别（pcset 掩码屏蔽人声）
  vision_analyzer.py       MediaPipe 手部关键点
  video_processor.py       视频处理 + HEVC 代理转码
  deepseek_agent.py        LLM 答疑 agent（三防线 + RAG 推荐）
  guitar_tab_assigner.py   吉他品位分配（动态规划指法求解）
  db/                      user / practice / feedback / reference / knowledge 子包
knowledge/                 吉他知识库（AI 答疑 grounding 数据源）
tests/                     三套回归测试（见下）
```

## 快速开始

```bat
# Windows 一键启动（需 .env，见下）
start.bat
# 浏览器打开 http://localhost:1218
```

手动启动：

```bat
F:\Python310\python.exe run.py
```

### 环境变量（.env，已被 .gitignore 排除）

```ini
DEEPSEEK_API_KEY=...        # AI 答疑（必需）
DASHSCOPE_API_KEY=...       # 通义千问 VL 手型视觉分析（必需）
VIRTUCOACH_ADMIN_PASSWORD=...   # 管理面板密码（未配置则管理面板不可进入，fail-closed）
JWT_SECRET_KEY=...          # 登录令牌签名密钥（未配置时使用开发回退值）
```

## 回归测试（一键）

三套带预想结果的基线回归，验证算法参数改动不引入退化。基线视频随仓库分发于 `tests/videos/`：

```bat
scripts\run_regression_tests.bat
```

或分套运行：

```bat
python -m pytest tests/chord_regression/ -v    # 和弦检测（3 例：must_contain / must_not_contain）
python -m pytest tests/strum_regression/ -v    # 扫弦分类（5 例：扫弦/非扫弦判定 + 2 个已知限制 xfail）
python -m pytest tests/score_regression/ -v    # 评分基线（2 例：四维分数须落在预想区间）
```

评分公式有意调整后，运行 `python tests/score_regression/update_baselines.py` 更新基线。

## 文档

- [创新点说明书.md](创新点说明书.md) — 7 个创新点定位 + 与 Yousician 对比 + 答辩 Q&A
- [项目功能说明.md](项目功能说明.md) — 功能清单与页面截图索引
- [PROJECT_BLUEPRINT.md](PROJECT_BLUEPRINT.md) — 项目蓝图
