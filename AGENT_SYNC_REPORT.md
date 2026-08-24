# VirtuCoach-Graduate 项目进度同步报告（给并行 Agent）

> 生成日期：2026-08-24
> 用途：让另一个 Agent 与主会话同步 Graduate 项目的进度、上下文、约束和待办。
> 项目位置：`F:\VirtuCoach-Graduate`（独立毕业项目，端口 1218，**不是** E:\VirtuCoach 的 7160）

---

## 0. 必读铁律（最高优先级，违反会被用户纠正）

这些约束同时适用于两个项目，Graduate 另有自己的补充：

| # | 规则 | 说明 |
|---|------|------|
| 1 | **不主动启动服务器** | 只有用户明确说"你测试一下/帮我测"才启动，测完立即关闭。 |
| 2 | **不主动 git commit/push** | GitHub 也一样。完成修复后只汇报，等用户指示再存档。**未经同意绝不 push**（用户 8/13 起多次未答复推送）。 |
| 3 | **不提交 `problem_snapshot/`** | 用户的问题截图文件夹，**永远不要 commit**。 |
| 4 | **不提交运行时产物** | `.gitignore` 已排除 `models/`、`*.db`、`backend/snapshots/`、缓存目录；`users.db` 等也勿提交。 |
| 5 | **改后端必须重启 1218 才生效** | 前端改动刷新即生效（且已禁浏览器缓存），后端改动需重启服务。 |
| 6 | **合并 analysis_service.py / analysis.py 时禁止整文件覆盖** | Graduate 有 auth + 练习记录保存块，VirtuCoach 没有，必须用 Edit 逐块合。 |
| 7 | **移植代码时注意行尾差异** | VirtuCoach 文件多为 LF，Graduate 为 CRLF。对比用 `diff --strip-trailing-cr`，否则 diff 全是行尾噪声。 |

---

## 1. 项目定位

**VirtuCoach-Graduate** = 从主项目 `E:\VirtuCoach` 分叉出的**独立毕业项目**。在保留主项目核心分析能力（音频/手型/AI 报告）的基础上，**新增了主项目没有的完整产品层**：

- **用户系统**：注册/登录/JWT/认证中间件
- **练习记录系统**：自动保存、历史、统计、双记录对比
- **多页面前端**：落地页/登录/注册/控制台（dashboard）/分析页/管理面板
- **管理员面板**：数据概览 + 意见收集 + 用户管理（三标签页）
- 全局浅蓝音乐主题 CSS 设计系统
- AI 聊天改为侧栏精灵 + 划词提问

**定位**：这是用户的**毕业设计/论文项目**，代码基线 = 主项目 GitLab `49862f6`（2026-08-07 状态），之后独立演进。

---

## 2. 环境

- **Python**：`E:\python.exe`（Python 3.10.11）
- **端口**：**1218**（不是 7160）
- **启动**：
  ```bash
  # 方式一（推荐）
  F:/VirtuCoach-Graduate/start.bat
  # 方式二
  cd F:/VirtuCoach-Graduate && E:/python.exe run.py
  ```
- **数据库**：SQLite（`users.db`、`practice_sessions.db`、`feedbacks.db`、`references.db`，路径统一在 `backend/data/`）
- **依赖坑**：
  - FastAPI 0.104.1 + starlette 0.27.0（与主项目一致）
  - `E:\python.exe` 有 `sys.prefix='E:'` bug，需设 PYTHONPATH
  - ffmpeg/ffprobe 在 PATH（`E:\ffmpeg.exe`）
  - aiofiles 已在 requirements

---

## 3. 核心架构

### 后端目录结构（`backend/`）

| 路径 | 职责 | 状态 |
|------|------|------|
| `backend/main.py` | FastAPI 主服务 + 路由注册 | 有未提交改动 |
| `backend/config.py` | 配置（端口/密钥/ONLINE_TIMEOUT_SECONDS=90） | 有未提交改动 |
| `backend/user_service.py` / `user_models.py` | 用户注册/登录/会话 | 有未提交改动 |
| `backend/auth_deps.py` | JWT 认证依赖 | 有未提交改动 |
| `backend/db/`（新包） | user/practice/feedback/reference/knowledge_db 移入的子包 | **未提交，新目录** |
| `backend/routers/` | auth.py / analysis.py / analytics.py / chat.py / hand_check.py / practice.py | 多个有未提交改动 |
| `backend/services/analysis_service.py` | 流水线编排 + 练习记录保存 | 有未提交改动 |
| `backend/services/hand_check_service.py` | 和弦手型检查 | 有未提交改动 |
| `backend/audio_analyzer.py` | 音频分析（从主项目同步） | 有未提交改动 |
| `backend/video_processor.py` / `vision_analyzer.py` | 手型/视觉 | 有未提交改动 |
| `backend/deepseek_agent.py` | AI 报告 + 聊天（含划词提问） | 有未提交改动（8/20） |
| `backend/_seed_test_data.py` | 测试数据种子脚本（50 用户） | 有未提交改动 |

### 前端（多页面，`frontend/`）

| 文件 | 职责 |
|------|------|
| `login.html` / `register.html` | 登录/注册（密码 ≥8 位含大小写+数字） |
| `dashboard.html` | 控制台（最近7天/练习历史/双记录对比/统计） |
| `analysis.html` | 视频分析页 |
| `admin.html` | 管理面板三标签页 |
| `js/auth.js` | 认证/心跳/offline 信标 |
| `js/hand-check.js` | 和弦手型检查 |

---

## 4. Git 状态（重要）

- **Git 远端（origin）**：`https://github.com/Andy-Dufrense/VirtuCoach.git`（个人 GitHub）
  - 注意：这个 GitHub 仓库**原来是主项目 VirtuCoach 的地址**，现已 force-push 覆盖为 Graduate 内容。主项目用的是 GitLab（`gitlab.enyamusic.cn`）。
- **分支**：`master`，**ahead of origin/master by 4 commits（未推送）**

### 4 个未推送本地提交
```
cc94da9  页面类文件(Html/CSS/JS)禁用浏览器缓存，避免缓存掩盖修复
f9b8a5a  测试数据扩充至50用户，意愿分布调整为中等意愿占主体
acb3e21  修复练习记录无法保存的根因 + 测试数据种子脚本
52309c1  admin 数据分析功能 + 五项测试问题修复
4386403  2026-08-07 升级基线：用户系统 + 练习记录 + 前端页面体系
b378f88  VirtuCoach snapshot from GitLab (2026-08-06) — init
```

> ⚠️ **4 个 commit 只在本地，未 push GitHub**。用户 8/13 起多次未答复推送，未经同意不要 push。

### 未提交改动（工作树，叠加在 4 个未推送 commit 之上）

`git status` 显示 **约 30 个文件改动**，核心是 **db/ 子包重构 + 单会话机制 + 登录延迟优化**这条线（8/14 存档点的工作），加上 8/18-8/21 的新工作：

| 类别 | 内容 |
|------|------|
| **db 重构** | `backend/db/` 新子包（user/practice/feedback/reference/knowledge_db.py 移入，import 改 `from db.xxx import ...`）；根目录旧 db 文件标记删除（D）；db 文件统一到 `backend/data/` |
| **单会话机制** | `user_db.py` online 列 + 迁移；`config.py` +ONLINE_TIMEOUT_SECONDS=90；`user_service.py` 单会话；`routers/auth.py` +heartbeat/offline；`js/auth.js` 心跳+offline 信标 |
| **登录延迟优化** | 三个 db 的 get_db 加 `PRAGMA synchronous=NORMAL`；login.html 登录按钮加「登录中…」+禁用 |
| **8/18-8/21 新工作** | `deepseek_agent.py`（8/20）、`创新点说明书.md`（8/20 论文）、`项目功能说明.md`（8/18）、`csv/` 导出（practice+users CSV）、`start.bat`（8/21 禁缓存） |

> ⚠️ 记忆只完整覆盖到 **8/14 存档点**；8/15-8/21 的新改动（deepseek_agent 8/20、创新点说明书、项目功能说明、CSV 导出）我观察到文件存在但**未逐行核实内容**。接手时先看这些文件的 mtime 和 diff。

---

## 5. 功能模块开发状态

### 5.1 用户系统 — 已完成（8/7 基线）
- 注册/登录/JWT/认证中间件，密码强度校验（≥8 位含大小写+数字）
- **JWT 密钥随机 bug 已修**：原 `os.urandom(32).hex()` 导致每次重启 token 全失效 → 改固定默认密钥（env 可覆盖）
- `auth.js` 有 `VC.sessionExpired()` + `validateSession` 调 `/api/auth/me`

### 5.2 练习记录系统 — 已完成（三层根因已修）
- 自动保存/历史/统计/双记录对比弹窗
- **三层根因（都已修）**：①report dict 绑给 report_text → sqlite InterfaceError 被吞 ②JWT 密钥随机 ③practice_db foreign_keys 跨库外键

### 5.3 管理员面板 — 已完成
- 数据概览 + 意见收集 + 用户管理三标签页
- admin/overview、admin/users 端点；反馈自动关联用户
- `routers/analytics.py` 五指标 + CSV BOM 导出

### 5.4 反馈系统 — 已完成（意见箱形式）
- 从 bug tracker 改为意见箱

### 5.5 音频/手型/AI 报告（核心分析）— 基线 49862f6，**落后于主项目**
- 当前 = 主项目 8/7 状态，**缺失主项目 8/13-14 的多项改进**（详见 findings.md 差距清单）

### 5.6 单会话机制 — 已实现（8/14，待验证）
- online 布尔 + 心跳新鲜度，跨浏览器/跨窗口登录拦截
- 两套机制并存：session_version 踢旧 + online 拦新

### 5.7 登录延迟优化 — 已实现（8/14）
- PRAGMA synchronous=NORMAL，commit 从 ~400ms → 几十ms

### 5.8 论文文档 — 进行中（8/18-8/20）
- `创新点说明书.md`（8/20）、`项目功能说明.md`（8/18）、桌面 `毕业论文_基于多模态融合的AI吉他陪练系统.md`

---

## 6. 关键参考文档

| 文件 | 位置 | 用途 |
|------|------|------|
| `findings.md` | `F:/VirtuCoach-Graduate/` | **核心**：主项目 8/13-14 改动学习报告 + 移植差距清单（P0-P3） |
| `task_plan.md` | 同上 | 8/14 会话任务计划（Phase 0-4 移植计划） |
| `PROJECT_BLUEPRINT.md` | 同上 | 项目蓝图（早期） |
| `创新点说明书.md` | 同上 | 论文创新点 |
| `项目功能说明.md` | 同上 | 项目功能说明 |
| `docs/2026-08-13-admin-analytics-plan.md` | `docs/` | admin 数据分析计划 |
| `毕业论文_基于多模态融合的AI吉他陪练系统.md` | 桌面 | 毕业论文 |

---

## 7. 待办清单（按优先级）

### P0 — 移植主项目 8/13-14 改进（findings.md 已列完整清单）
| 项 | 内容 | 风险 |
|----|------|------|
| P0-1 | 扫弦误判单音四根因（audio_analyzer.py 111行，**可整文件替换**） | 低 |
| P0-2 | HEVC 长视频代理转码（video_processor.py 145行，**不修则 >1分钟手机视频全超时**） | 低 |
| P0-3 | 手型检查 UI 四个 bug（和弦 select/stopCameraPreview/进度条回退/上传复位） | 低 |
| P0-4 | 视频分析上传两个 bug（capo alert 硬阻塞/重选同文件无反应） | 低 |
| P1-5 | 原视频回放模块（206 端点 + 代理留存 + 音频错误跳转芯片） | **中（analysis_service 不能整文件覆盖）** |
| P1-6 | chord_classifier.py 恢复（横按和弦修正静默失效中） | 低 |
| P2-7 | 聊天推荐/防线套件（deepseek_agent.py + chat.py） | 中 |
| P2-8 | recording_quality 透传（一行修复） | 低 |
| P3-9 | 工程化：Graduate 完全没有测试 | — |

### P1 — 验证 8/14 存档点的改动
1. 重启 1218 后端（online 列迁移 + heartbeat/offline 端点启动时自动执行）
2. 重新登录验证：跨浏览器/跨窗口登录被拦「该账号已在其他设备登录」；退出或关页面后能立即重登

### P2 — 论文/答辩准备
3. 完善创新点说明书、项目功能说明
4. 毕业演示必踩 HEVC 转码（P0-2），务必先修

### 决策待用户拍板
5. 是否 push 4 个本地 commit 到 GitHub（用户 8/13 起未答复）
6. 移植顺序（findings.md 建议：P0 四项一次提交 → P1 回放+chord_classifier 一次 → P2 聊天套件一次）

---

## 8. 下一步计划（主会话意图）

1. **先确认 8/15-8/21 的新改动**（deepseek_agent 8/20、创新点说明书、项目功能说明、CSV 导出）——这些记忆没覆盖，接手时先 `git diff` 和看文件 mtime。
2. **验证 8/14 存档点的单会话 + 登录延迟**（需用户授权重启 1218）。
3. **按 findings.md 移植主项目 8/13-14 改进**（P0 四项优先，尤其 HEVC 转码是毕业演示硬需求）。
4. 移植时**禁止整文件覆盖** analysis_service.py / analysis.py，用 Edit 逐块合。
5. 所有 git 操作等用户明确同意。

---

## 9. 协作建议（并行 Agent 分工）

- **Graduate 和主项目 VirtuCoach 是两个独立 git 仓库、独立端口（1218 vs 7160），改动互不影响。** 如果两个 Agent 分别负责一个项目，冲突风险低。
- **Graduate 的独特工作**：用户系统/练习记录/admin 面板/论文文档/从主项目移植。
- **动手前必做**：①`git status` 看未提交改动 ②读 `findings.md` 对应移植项 ③确认在 Graduate 目录（`F:/VirtuCoach-Graduate`）。
- **改前端 bump 缓存版本**：每次改前端 `bump analysis.html 的 ?v=N→N+1`（虽然已禁浏览器缓存，但保持惯例）。
- **Windows 编码坑**：诊断脚本 emoji 触发 GBK 崩溃，加 `sys.stdout.reconfigure(encoding="utf-8")`。

---

## 10. 与主项目 VirtuCoach 的区别（速查）

| | VirtuCoach（主项目） | VirtuCoach-Graduate（本报告） |
|---|---|---|
| 路径 | `E:\VirtuCoach` | `F:\VirtuCoach-Graduate` |
| 端口 | 7160 | 1218 |
| Git 远端 | GitLab `gitlab.enyamusic.cn/enya-guangzhou-ai/VirtuCoach` | GitHub `github.com/Andy-Dufrense/VirtuCoach` |
| 定位 | 纯分析系统（公司项目） | 独立毕业项目（+用户系统/练习记录/admin/论文） |
| 前端 | 单页（index.html） | 多页（login/dashboard/analysis/admin） |
| 认证 | 无 | JWT 用户系统 |
| 代码基线 | 持续演进（最新 bc9dded 8/21） | 49862f6（8/7）+ 独立演进，**落后主项目** |
| 触发词 | "VirtuCoach" | "VirtuCoach-Graduate" / "Graduate" |

**代码关系**：音频/手型/视觉检测代码有同步关系（Graduate 基线来自主项目），但 Graduate 需要**手动移植**主项目的后续改进，不是自动同步。
