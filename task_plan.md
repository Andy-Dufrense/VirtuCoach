# Task Plan — VirtuCoach-Graduate 恢复会话 (2026-08-14)

## Goal
完成 8/13 存档点遗留的验证 + 提交/推送决策。

## 当前状态（8/14 检查）
- ✅ git 状态与 8/13 存档点一致：4 个未推送提交 (52309c1/acb3e21/f9b8a5a/cc94da9)，11 个未提交改动文件
- ✅ 端口 1218 空闲 → 后端未运行
- ⚠️ problem_snapshot/ 未跟踪 → 用户问题截图，**不要提交**

## Phases

### Phase 1: 用户验证修复效果 `in_progress`
- 启动后端 (start.bat 或 python run.py) → 重新登录 Andy → 跑一次视频分析
- 检查：练习历史 / 统计 / 最近7天 / 双记录对比弹窗是否正常
- 依据：8/13 修了练习记录保存三层根因，e2e 已实测通过，待用户 UI 确认

### Phase 2: 提交未提交改动 `pending`
- 验证通过后提议 commit 本轮 11 个文件改动（排除 problem_snapshot/）
- **需用户明确同意**

### Phase 3: 推送决策 `pending`
- 询问是否连同 4 个本地提交一起推 GitHub origin/master
- 用户 8/13 未答复，**未经同意不 push**

## Rules
- 不主动启动服务器（用户 feedback：只在明确要求测试时启动，测完即关）
- 不提交 problem_snapshot/
- 改后端必须重启 1218 才生效

### Phase 0: 学习 VirtuCoach 8/13-14 改动 `complete` (8/14)
- VirtuCoach 在 Graduate 基线(49862f6)之后又提交 921239b + 8fcbb70（均已推 GitLab）
- 完整差距清单见 **findings.md**：P0 扫弦误判四根因/HEVC代理转码/手型UI四bug/上传两bug；P1 原视频回放模块/chord_classifier恢复；P2 聊天推荐套件/recording_quality透传；P3 测试套件
- 环境已验证：ffmpeg/ffprobe 在 PATH，aiofiles 已在 requirements
- **待用户测试完交流后决定是否移植、移植顺序**

### Phase 4: 移植（待用户拍板） `pending`
- 按 findings.md 移植顺序执行；analysis_service.py/analysis.py 禁止整文件覆盖（Graduate 有 auth+练习记录保存块）

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| diff 全是行尾差异 | 1 | 改用 `diff --strip-trailing-cr` 得到真实差异 |
