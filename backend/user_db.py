"""User database — sqlite3 with WAL, following feedback_db.py pattern."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "users.db"


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_user_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT (datetime('now', 'localtime')),
            last_login DATETIME,
            last_activity DATETIME,
            active_seconds INTEGER DEFAULT 0
        )
    """)
    # 迁移：旧库补充新列
    cols = {r[1] for r in conn.execute("PRAGMA table_info(users)")}
    if "last_activity" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN last_activity DATETIME")
    if "active_seconds" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN active_seconds INTEGER DEFAULT 0")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")

    # 登录日志（用于登录频率等数据分析；自建表起积累）
    conn.execute("""
        CREATE TABLE IF NOT EXISTS login_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            login_at DATETIME DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_login_events_user_time "
        "ON login_events(user_id, login_at)"
    )
    conn.commit()
    conn.close()
