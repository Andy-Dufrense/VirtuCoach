# admin 数据分析功能实施计划

## Context

VirtuCoach-Graduate 的 admin 面板已有用户/练习/反馈数据，但只能看原始列表。用户（测试+毕业论文需要）要求增加数据分析能力：在线人数、登录频率、练习分析、用户增长、使用意愿五个独立分析按钮，每个都可导出 CSV 拿去做进一步分析。设计已与用户确认：独立「📈 数据分析」标签页、按钮式切换、Chart.js 画图、后端算好再返回。

关键约束：
- **登录历史此前无记录**（last_login 只存最近一次），需新建 login_events 表，数据从今天起积累
- 前端为无构建步骤的 vanilla JS，Chart.js 走 CDN，离线时降级为纯表格
- CSV 需带 BOM（utf-8-sig），否则 Excel 打开中文乱码
- admin 接口用现有 `_verify_admin(admin_token)` 模式保护（main.py:245）

## 改动清单

### 1. 数据层 — `backend/user_db.py`
- CREATE TABLE 增加 `login_events (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, login_at DATETIME DEFAULT (datetime('now','localtime')), FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE)`
- 加索引 `idx_login_events_user_time(user_id, login_at)`
- init 里照 users 表的迁移模式（PRAGMA table_info 检查后 CREATE）

### 2. 登录埋点 — `backend/user_service.py`
- `login_user()` 验证成功、更新 last_login 之后，INSERT 一条 login_events（同一连接同一事务）

### 3. 阈值配置 — `backend/config.py`
- `WILLINGNESS_HIGH = 70`、`WILLINGNESS_LOW = 40`、`WILLINGNESS_WINDOW_DAYS = 30`、`ONLINE_WINDOW_MINUTES = 30`

### 4. 后端聚合 API — 新文件 `backend/routers/analytics.py`
模式参考 `backend/routers/practice.py`（模块级 `router = APIRouter(prefix="/api", tags=["analytics"])`），直接 import 三个 db getter。每个指标写成**接收 conn 参数的纯函数**（便于用 :memory: 库测试），端点层负责取连接、校验 admin_token（`Query(...)` + main 里的 `_verify_admin`——把 `_verify_admin` 从 main.py 挪到 analytics 可 import 的位置，或 analytics 内复制一个同样的校验函数并 import ADMIN_PASSWORD；选择后者，改动最小）。

端点（全部 `admin_token: str = Query(...)`）：
| 端点 | 返回要点 |
|---|---|
| `GET /api/admin/analytics/online` | current_online（last_activity≤30min）、today_active、week_active、top10 活跃时长排行 |
| `GET /api/admin/analytics/logins?days=30` | daily 数组（日期补齐为0）+ 用户登录次数排行 |
| `GET /api/admin/analytics/practice?days=30` | daily 趋势、用户排行、曲目分布(chord_or_track)、分数段分布(0-60/60-75/75-90/90+) |
| `GET /api/admin/analytics/growth?days=30` | 每日注册数 + 累计用户曲线 |
| `GET /api/admin/analytics/willingness` | 每用户综合分+档位、高/中/低人数 |
| `GET /api/admin/analytics/export?type=logins\|practice\|users\|willingness&days=30` | text/csv 下载，utf-8-sig，Content-Disposition 带文件名 |

日期补齐：Python 侧生成 today-days+1..today 的日期序列，SQL 用 `date(login_at)` 分组（库内时间为 localtime，直接 date() 即可）。

**使用意愿算法**（近 WILLINGNESS_WINDOW_DAYS 天）：
- 维度：登录次数×0.3 + 练习次数×0.3 + active_seconds×0.2 + 反馈提交数×0.2
- 每维在全体用户间 min-max 归一化（全为0时该维度记0）→ 加权×100 → 保留1位小数
- 分档：≥70 高 / 40-69 中 / <40 低

### 5. 注册路由 — `backend/main.py`
- `from routers.analytics import router as analytics_router` + `app.include_router(analytics_router)`（在 practice_router 之后），更新启动日志里的 Routes registered 列表

### 6. 前端 — `frontend/admin.html`
- 第 5 个标签按钮 `📈 数据分析` + `switchTab` 的 tabNames 加 `analytics: '数据分析'`
- `section-analytics`：
  - 顶部按钮行：👥 在线统计 / 🔑 登录频率 / 🎸 练习分析 / 📈 用户增长 / 💗 使用意愿（复用 .btn-outline/.btn-primary 样式，激活态高亮）
  - 登录/练习/增长区块带时间范围 select（7/30/90 天）
  - 内容区：数字卡（复用 .overview-stat-card）+ `<canvas>` 图表 + 排行表（复用 .user-table）+ 每块右上「⬇ 导出CSV」（`<a href="/api/admin/analytics/export?type=...&admin_token=...">`）
- `<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js">` + 渲染前检查 `window.Chart`，缺失则内容区提示"图表库加载失败，已降级为表格"
- 图表配色取现有主题变量（#5c6bc0 / #e53935 / #f59e0b / #43a047）

## 验证

1. `E:\python.exe -m py_compile` 过所有改动的后端文件；admin.html 内联脚本提取后 `node --check`
2. 聚合逻辑测试：python 脚本用 sqlite :memory: 构造 users/login_events/practice_sessions/feedbacks 样例数据，调用 analytics 纯函数，断言 daily 补齐、排行排序、意愿分档正确
3. 用户重启后端（端口 1218）+ Ctrl+F5 后，人工点五个按钮 + 导出一份 CSV 用 Excel 验证中文无乱码
4. 提醒用户：登录/活跃数据从今天才开始积累，趋势图头几天数据少属正常

## 收尾

全部验证通过后提交一个 commit（含计划存档），推 GitHub origin/master（推送前询问用户）。
