# Findings — VirtuCoach 8/13-14 改动学习报告 (2026-08-14)

## 背景
- Graduate 基线 = GitLab 49862f6 (8/7: 录音质量诊断 + containment_bonus)
- VirtuCoach 在其后又推了两个提交（均已推 GitLab，master 与 origin 同步）：
  - **921239b (8/13)**: HEVC代理转码 + 扫弦误判四根因 + 推荐多样性 + 聊天防线/level + 和弦select改造 + 上传复位 + 进度条单调化（18文件 +1380行）
  - **8fcbb70 (8/14)**: 原视频回放模块 — Range 206端点 + 代理留存 + 音频错误跳转芯片（9文件 +518行）
- 环境确认：ffmpeg/ffprobe 在 PATH (E:\ffmpeg.exe)；aiofiles 已在 Graduate requirements ✓
- 行尾差异干扰大：对比必须用 `diff --strip-trailing-cr`（VirtuCoach 文件多为 LF，Graduate 为 CRLF）

## 差距清单（Graduate 缺失项，按优先级）

### P0-1 扫弦误判单音四根因（audio_analyzer.py，111行diff，整体可移植）
VirtuCoach 项目铁律：**扫弦只有 confidence ≥0.5 才算数**。Graduate 四处违反：
1. spectral merge 无门槛：onset_clusters 里 strumming 段直接并入 technique_segments → 加 `>=0.5` 过滤
2. activation 检测 conf 上限 0.75→0.45（jump/sustain 两规则），只作旁证不能单独定性扫弦
3. `_classify_onset_cluster` 加集中频谱否决：`spread<900 and peaks<1.5 → strumming-0.30/arpeggio+0.10`
4. `detect_technique_segments` 加 `audio_duration` 参数校准帧率（basic-pitch 实测~86fps，旧假设31.25fps 差2.75×→时间戳超出视频长度）
→ 两边文件除这111行外完全一致，**直接整文件替换风险极低**（替换后 Graduate 该文件=VirtuCoach 版）

### P0-2 HEVC 长视频代理转码（video_processor.py，145行diff，整体可移植）
- `_probe_video`(ffprobe) + `_make_proxy_video`(720p30 H.264 ultrafast+aac音轨)
- 触发：codec∈{hevc,h265,vp9,av1,mpeg4} 或 >720p 或 >35fps
- 动态超时 `min(max(45, dur*2+60), 480)`
- **不修的话：>1分钟手机视频全部"未检测到手型"超时**——毕业演示必踩
- 除这145行外两边完全一致，可整体移植

### P0-3 手型检查 UI 四个 bug（Graduate 全中）
| bug | Graduate 现状 | VirtuCoach 修法 |
|---|---|---|
| 和弦无法选择 | analysis.html 仍是 `<input>+<datalist id="chordDatalist">`，手机浏览器不弹候选 | 原生 `<select>` optgroup 分组（选项来自 /api/chords，Graduate 已有该端点 hand_check.py:68），末尾"其他和弦"手动输入兜底 |
| stopCameraPreview 未定义 | hand-check.js:45 调用但无定义 → 切"技巧检查"抛 ReferenceError，切换后半段静默中断 | 补函数定义（停录制/停流/复位预览） |
| 进度条回退再拉满 | startHandProgress 同款抛物线 `5+6t−0.24t²`（12.5s 顶点后倒退） | `computeHandProgress`=`min(95,max(5,5+90*(1−e^(−t/12))))` 单调渐近 |
| 第二次上传同文件无反应 | handleChordVideoUpload 不复位 input.value | 读到文件后立即 `input.value=""` |

### P0-4 视频分析上传两个 bug
- `app.js:291` capo alert 硬阻塞 + capoSelect 默认空占位 → 首次上传必被挡；后端本来就自动检测 capo（analysis_service detected_capo）→ 删阻塞，默认项改 `0品(无变调夹，自动识别)`
- handleFileSelect 不复位 videoInput.value（只在 resetUI:1153 复位）→ 重选同一文件无 change 事件

### P1-5 原视频回放模块（8fcbb70，新功能，论文演示加分）
后端三处（Graduate 都缺）：
- video_processor.py `proxy_dest` 参数（P0-2 的扩展，留存代理+`result["proxy_path"]`）
- analysis_service.py run()/_handle_no_audio 传 `proxy_dest=UPLOAD_DIR/{task_id}_proxy.mp4`；⚠️**合并不能整文件覆盖**——Graduate 有 practice_db_getter+保存块，VirtuCoach 没有
- routers/analysis.py：`_parse_range`+`_stream_file`(aiofiles 1MB块)+`GET /api/task/{id}/video` 206端点+task status 返回 video_url+_remove_task 同删 proxy；⚠️移植时保留 Graduate 的 `Depends(get_current_user)` 和 user_id
- starlette 0.27 FileResponse 不支持 Range，必须手写流式
前端（需适配，不能直拷——Graduate 是多页+自己的设计系统）：
- 结果页插 videoReplayCard（`<video>` + 芯片区）→ 目标是 analysis.html 结果区
- `collectVideoMarkers` 只收 audio_errors（用户定稿：手型有截图对照不需跳视频；音频无视觉锚点才跳；无错误显示提示不显示芯片）、跳≤0.5s、升序、<1s合并、上限12
- 测试参考 backend/test_video_replay_range.py：**不用 TestClient**（httpx 对 starlette 0.27 太新），直接 ASGI scope 驱动；receive 第二次调用必须永远挂起否则流被中止

### P1-6 chord_classifier.py 恢复
- Graduate 8/13 删了它（当时 GitLab 没有）；VirtuCoach 921239b 已提交（348行）
- analysis_service.py:826 `from chord_classifier import LandmarkChordClassifier` try/except → 现在 Graduate 里横按和弦修正（barre→限定F系候选、矛盾降音频置信度-0.10）**静默失效**
- 修复=从 VirtuCoach 拷回单文件即可

### P2-7 聊天推荐/防线套件（deepseek_agent.py 221行diff + chat.py 26行）
- `_load_song_list`（解析 recommended-songs.md）/`_is_song_question`/`_norm_chord_name`/`_rank_and_sample_songs`（和弦重叠排序 top4+随机补足8首+打乱）/`_song_recommendation_block`（等级过滤+演奏者类型过滤+禁儿歌+结尾问偏好）
- `song_guess_note` 禁猜曲目铁律（两条聊天路径注入）
- kb_section 水平过滤警告；_filter_beginner_songs 去掉仅 beginner 门槛
- chat.py 两端点 `setdefault("level"/"instrument")`（否则聊天永远 beginner）
- ⚠️ Graduate 聊天只走 /api/ask/stream（app.js:900）= 恰是 VirtuCoach 曾经绕过防线的路径；sendQuestion 只传 `context: result`，不带 level → 同病
- 前置依赖：Graduate 的 recommended-songs.md 是旧版（缺 chords 列格式、缺《遇见》《开始懂了》、缺中级单音曲）→ 先整体同步 md
- Graduate 优势可利用：用户系统有 skill_level，可从用户档案取 level 而不是靠前端下拉

### P2-8 recording_quality 透传
- audio_analyzer 已算（49862f6 就有），但 Graduate analysis_service 的 result 组装漏了 `recording_quality` 字段 → 前端/报告拿不到
- 一行修复

### P3-9 工程化：Graduate 完全没有测试
VirtuCoach 有：tests/strum_regression（6用例）/tests/hand_regression/backend/test_chord_select_ui.js（38项 DOM-stub）/test_video_replay_range.py（29项）/verify_fix.py/diag_single_note.py。建议随对应功能一起移植。

## 反向：Graduate 有而 VirtuCoach 没有的（以后交流素材）
用户系统/JWT固定密钥教训/练习记录三层根因/recent_7d/双记录对比弹窗/admin数据分析/密码强度/禁浏览器缓存。VirtuCoach 无登录系统（analysis.py 无 auth）。

## 移植顺序建议
1. P0 四项（扫弦/HEVC/手型UI/上传UI）——一次提交
2. P1 回放模块 + chord_classifier 恢复——一次提交
3. P2 聊天套件（先同步歌单md）——一次提交
4. 每次改前端 bump analysis.html 的 ?v=14→15...
5. 合并 analysis_service.py / analysis.py 时用 Edit 逐块合，**禁止整文件覆盖**
