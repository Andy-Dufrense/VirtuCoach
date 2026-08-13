"""批量生成测试用户 + 模拟过去30天的登录/练习/反馈数据，用于数据分析页面测试。

用法：
  E:\\python.exe _seed_test_data.py          # 生成数据
  E:\\python.exe _seed_test_data.py --clean  # 清除全部测试数据（按邮箱域名识别）

所有测试账号密码统一为 Test123456，邮箱域名 virtucoach.test 便于识别和清理。
"""
import os
import random
import sys
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import user_db
import practice_db
import feedback_db
from user_service import hash_password

SEED = 20260813
random.seed(SEED)

PASSWORD = "Test123456"
DOMAIN = "virtucoach.test"
FMT = "%Y-%m-%d %H:%M:%S"
NOW = datetime.now()
TODAY = NOW.date()


def fmt(dt: datetime) -> str:
    return dt.strftime(FMT)


# (用户名, 注册距今天数, 活跃度档位) —— 共49个测试账号(+真实账号Andy=50用户)
# 档位设计目标：意愿分档 高≈6 / 中≈29 / 低≈15，中等意愿占主体
USERS = [
    ("琴弦少年", 52, "heavy"), ("指弹大叔", 48, "heavy"), ("和弦控", 44, "heavy"),
    ("民谣小白", 38, "mid"), ("夜曲练习生", 33, "mid"), ("吉他女神", 27, "mid"),
    ("扫弦达人", 23, "mid"), ("木吉他手", 20, "mid"),
    ("初学小哥", 24, "light"), ("业余爱好者", 18, "light"), ("弹琴的猫", 12, "light"),
    ("隔壁小刚", 8, "light"), ("慢热学员", 5, "light"),
    ("路过看看", 14, "dormant"), ("注册试试", 9, "dormant"), ("三分钟热度", 3, "dormant"),
    # ── 第二批（扩到50用户）──
    ("吉他老炮", 58, "near_high"), ("琴房常客", 50, "near_high"), ("练琴狂人", 44, "near_high"),
    ("节奏控", 42, "mid_core"), ("和声迷", 39, "mid_core"), ("弹唱青年", 35, "mid_core"),
    ("木箱琴声", 31, "mid_core"), ("深夜琴房", 28, "mid_core"), ("弦上漫步", 26, "mid_core"),
    ("慢练派", 24, "mid_core"), ("爬格子选手", 22, "mid_core"), ("琴弦暖男", 20, "mid_core"),
    ("扫弦青年", 18, "mid_core"), ("左手茧子", 17, "mid_core"), ("右手节奏", 16, "mid_core"),
    ("民谣小站", 15, "mid_core"), ("琴房邻居", 14, "mid_core"), ("晚风琴声", 13, "mid_core"),
    ("半音阶", 12, "mid_core"), ("调弦师", 11, "mid_core"), ("变调夹", 10, "mid_core"),
    ("随便看看", 19, "mid_low"), ("新手上路", 17, "mid_low"), ("偶尔练琴", 15, "mid_low"),
    ("琴弦吃灰", 13, "mid_low"), ("三天打鱼", 11, "mid_low"), ("佛系学员", 9, "mid_low"),
    ("摸鱼乐手", 8, "mid_low"), ("随缘练习", 7, "casual"), ("琴在角落", 6, "casual"),
    ("初来乍到", 5, "casual"), ("试水玩家", 4, "casual"), ("路过学琴", 3, "casual"),
]

PROFILES = {
    "heavy":     dict(logins=(20, 26), practices=(10, 16), active=(25000, 60000), score=(70, 95),
                      levels=("intermediate", "advanced"), feedback=(1, 1)),
    "near_high": dict(logins=(27, 32), practices=(12, 15), active=(42000, 58000), score=(75, 95),
                      levels=("intermediate", "advanced"), feedback=(1, 1)),
    "mid":       dict(logins=(12, 18), practices=(6, 9),   active=(10000, 20000), score=(55, 85),
                      levels=("beginner", "intermediate"), feedback=(1, 1)),
    "mid_core":  dict(logins=(16, 24), practices=(7, 11),  active=(16000, 32000), score=(55, 85),
                      levels=("beginner", "intermediate"), feedback=(1, 1)),
    "mid_low":   dict(logins=(13, 19), practices=(5, 8),   active=(10000, 18000), score=(45, 80),
                      levels=("beginner",), feedback=(1, 1)),
    "light":     dict(logins=(2, 5),   practices=(0, 2),   active=(500, 4000),    score=(40, 75),
                      levels=("beginner",), feedback=(0, 0)),
    "casual":    dict(logins=(2, 8),   practices=(0, 3),   active=(500, 8000),    score=(40, 75),
                      levels=("beginner",), feedback=(0, 0)),
    "dormant":   dict(logins=(0, 1),   practices=(0, 0),   active=(0, 300),       score=None,
                      levels=("beginner",), feedback=(0, 0)),
}

CHORDS = ["C和弦", "G和弦", "Am和弦", "Em和弦", "F和弦", "Dm和弦", "Bm和弦"]
SONGS = ["晴天", "夜曲", "平凡之路", "成都", "海阔天空", "丁香花", "老男孩"]
TECHNIQUES = ["扫弦", "推弦", "击勾弦", "分解和弦"]

FEEDBACK_POOL = [
    ("video_analysis", "suggestion", "希望报告能导出PDF", "分析完的报告要是能下载成PDF就更好了，方便保存给老师看。"),
    ("video_analysis", "confusion", "综合分是怎么算的？", "报告里的综合分看不出来是怎么计算的，能否加个说明？"),
    ("hand_check", "experience", "手型检查有时识别不到手指", "光线暗的时候手型检查经常说检测不到手，希望能提示调整光线。"),
    ("hand_check", "suggestion", "希望多几个技巧的参考图", "推弦和滑音的参考图太少了，希望补充。"),
    ("video_analysis", "other", "分析速度有点慢", "一段两分钟的视频要等将近一分钟，希望能更快。"),
]


def rand_time_on(day: date) -> datetime:
    """某天随机一个 8:00-22:00 的时间；如果是今天则不超过当前时刻。"""
    if day == TODAY:
        start = datetime.combine(day, datetime.min.time()) + timedelta(hours=8)
        if NOW <= start:
            start = datetime.combine(day, datetime.min.time())
        span = max(int((NOW - start).total_seconds()) - 60, 60)
        return start + timedelta(seconds=random.randint(0, span))
    hour, minute = random.randint(8, 22), random.randint(0, 59)
    return datetime(day.year, day.month, day.day, hour, minute)


def pick_login_days(n: int, force_today: bool) -> list[int]:
    """从过去30天里挑 n 个登录日（返回距今天数，越近权重越高）。"""
    weights = [30 - age for age in range(30)]
    days = set()
    if force_today:
        days.add(0)
    while len(days) < n:
        d = random.choices(range(30), weights=weights, k=1)[0]
        days.add(d)
    return sorted(days)


def clean():
    uc = user_db.get_db()
    ids = [r["id"] for r in uc.execute("SELECT id FROM users WHERE email LIKE ?", [f"%@{DOMAIN}"])]
    if not ids:
        print("没有找到测试数据，无需清理")
        uc.close()
        return
    pc, fc = practice_db.get_db(), feedback_db.get_db()
    q = ",".join("?" * len(ids))
    pc.execute(f"DELETE FROM practice_sessions WHERE user_id IN ({q})", ids)
    pc.commit()
    fc.execute(f"DELETE FROM feedbacks WHERE user_id IN ({q})", ids)
    fc.commit()
    uc.execute(f"DELETE FROM users WHERE id IN ({q})", ids)  # login_events 级联删除
    uc.commit()
    print(f"已清理 {len(ids)} 个测试用户及其登录/练习/反馈数据")
    uc.close(); pc.close(); fc.close()


def seed():
    # 确保表结构是最新的（后端未重启时 login_events 等可能尚未创建）
    user_db.init_user_db()
    practice_db.init_practice_db()
    feedback_db.init_feedback_db()
    uc = user_db.get_db()
    if uc.execute("SELECT COUNT(*) FROM users WHERE email LIKE ?", [f"%@{DOMAIN}"]).fetchone()[0]:
        print("已存在测试数据，请先运行 --clean 再重新生成")
        uc.close()
        return
    pc, fc = practice_db.get_db(), feedback_db.get_db()
    pw_hash = hash_password(PASSWORD)

    stats = []
    feedback_users = []
    online_quota = 3  # 前几个高活跃用户设为"当前在线"
    for i, (name, reg_ago, tier) in enumerate(USERS, start=1):
        prof = PROFILES[tier]
        email = f"test{i:02d}@{DOMAIN}"
        created = NOW - timedelta(days=reg_ago, hours=random.randint(0, 12))
        uc.execute(
            "INSERT INTO users (username, email, password_hash, created_at, active_seconds) "
            "VALUES (?,?,?,?,?)",
            [name, email, pw_hash, fmt(created), random.randint(*prof["active"])],
        )
        uc.commit()
        uid = uc.execute("SELECT last_insert_rowid()").fetchone()[0]

        # ── 登录事件 ──
        n_logins = random.randint(*prof["logins"])
        login_times = []
        if n_logins:
            force = tier in ("heavy", "near_high", "mid", "mid_core", "mid_low") and random.random() < 0.6
            days = pick_login_days(min(n_logins, 26), force_today=force)
            for age in days:
                for _ in range(random.randint(1, 2)):
                    if len(login_times) < n_logins or random.random() < 0.3:
                        login_times.append(rand_time_on(TODAY - timedelta(days=age)))
            login_times.sort()
            for t in login_times:
                uc.execute("INSERT INTO login_events (user_id, login_at) VALUES (?,?)", [uid, fmt(t)])
            uc.execute("UPDATE users SET last_login = ? WHERE id = ?", [fmt(login_times[-1]), uid])

        # 最近活跃：前几个高活跃用户设为"刚刚在线"，其余=最后登录时间
        if online_quota and tier in ("heavy", "near_high"):
            uc.execute("UPDATE users SET last_activity = ? WHERE id = ?",
                       [fmt(NOW - timedelta(seconds=random.randint(120, 900))), uid])
            online_quota -= 1
        elif login_times:
            uc.execute("UPDATE users SET last_activity = ? WHERE id = ?", [fmt(login_times[-1]), uid])

        # ── 练习记录 ──
        n_prac = random.randint(*prof["practices"])
        if n_prac and login_times:
            session_days = random.choices(login_times, k=n_prac)
            base_lo, base_hi = prof["score"]
            for t in sorted(session_days):
                roll = random.random()
                if roll < 0.5:
                    mode, target = "chord_check", random.choice(CHORDS)
                elif roll < 0.9:
                    mode, target = "video_analysis", random.choice(SONGS)
                else:
                    mode, target = "technique_check", random.choice(TECHNIQUES)
                if random.random() < 0.08:
                    overall = audio = hand = 0  # 少量分析失败、无评分的记录
                else:
                    overall = random.randint(base_lo, base_hi)
                    audio = max(1, min(100, overall + random.randint(-10, 10)))
                    hand = max(1, min(100, overall + random.randint(-12, 8)))
                pc.execute(
                    "INSERT INTO practice_sessions (user_id, video_filename, instrument, "
                    "skill_level, chord_or_track, overall_score, audio_score, hand_score, "
                    "report_text, mode, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    [uid, f"seed_{uid}_{t.strftime('%m%d%H%M')}.mp4", "guitar",
                     random.choice(prof["levels"]), target, overall, audio, hand,
                     "", mode, fmt(t + timedelta(minutes=random.randint(5, 60)))],
                )
        fb_count = random.randint(*prof["feedback"])
        if fb_count:
            feedback_users.append((uid, name, fb_count))
        stats.append((name, email, tier, len(login_times), n_prac))

    uc.commit(); pc.commit()

    # ── 反馈 ──
    n_feedbacks = 0
    for uid, name, fb_count in feedback_users:
        for _ in range(fb_count):
            cat, sev, title, desc = random.choice(FEEDBACK_POOL)
            fc.execute(
                "INSERT INTO feedbacks (project, category, severity, title, description, "
                "tester_name, user_id, username, browser_info, status, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ["VirtuCoach", cat, sev, title, desc, name, uid, name,
                 "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0",
                 random.choice(["pending", "investigating", "resolved"]),
                 fmt(NOW - timedelta(days=random.randint(0, 15), hours=random.randint(0, 20)))],
            )
            n_feedbacks += 1
    fc.commit()

    print(f"已生成 {len(USERS)} 个测试用户（密码统一：{PASSWORD}）")
    print(f"{'用户名':<8} {'邮箱':<24} {'档位':<10} 登录  练习")
    for name, email, tier, lg, pc_n in stats:
        print(f"{name:<8} {email:<24} {tier:<10} {lg:>3}  {pc_n:>3}")
    total_logins = uc.execute("SELECT COUNT(*) FROM login_events").fetchone()[0]
    total_prac = pc.execute("SELECT COUNT(*) FROM practice_sessions").fetchone()[0]
    total_users = uc.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    print(f"合计：用户总数 {total_users}、{total_logins} 条登录记录、"
          f"{total_prac} 条练习记录、{n_feedbacks} 条新反馈")
    uc.close(); pc.close(); fc.close()


if __name__ == "__main__":
    if "--clean" in sys.argv:
        clean()
    else:
        seed()
