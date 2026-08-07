import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "feedbacks.db"


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_feedback_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedbacks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT NOT NULL DEFAULT 'VirtuCoach',
            category TEXT NOT NULL,
            severity TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT DEFAULT '',
            steps_to_reproduce TEXT DEFAULT '',
            screenshot TEXT DEFAULT '',
            tester_name TEXT DEFAULT '',
            browser_info TEXT DEFAULT '',
            status TEXT DEFAULT 'pending',
            admin_note TEXT DEFAULT '',
            created_at DATETIME DEFAULT (datetime('now', 'localtime'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fb_project ON feedbacks(project)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fb_status ON feedbacks(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fb_created ON feedbacks(created_at DESC)")
    conn.commit()
    conn.close()
