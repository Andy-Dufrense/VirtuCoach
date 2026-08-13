"""Practice sessions database — following feedback_db.py pattern."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "practice_sessions.db"


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    # 注意：不开 foreign_keys。practice_sessions.user_id 外键指向 users 表，
    # 但 users 在另一个数据库文件里，SQLite 不支持跨库外键——开启后所有
    # INSERT 都会报 "no such table: main.users"，导致练习记录无法保存。
    return conn


def init_practice_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS practice_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            video_filename TEXT DEFAULT '',
            instrument TEXT DEFAULT 'guitar',
            skill_level TEXT DEFAULT 'beginner',
            chord_or_track TEXT DEFAULT '',
            overall_score REAL DEFAULT 0,
            audio_score REAL DEFAULT 0,
            hand_score REAL DEFAULT 0,
            report_text TEXT DEFAULT '',
            mode TEXT DEFAULT 'video_analysis',
            created_at DATETIME DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ps_user_id ON practice_sessions(user_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ps_created ON practice_sessions(created_at DESC)"
    )
    conn.commit()
    conn.close()
