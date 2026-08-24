"""User service: password hashing, JWT, registration, login."""

import re
import bcrypt
from datetime import datetime, timedelta
from jose import jwt, JWTError

from config import JWT_SECRET_KEY, JWT_ALGORITHM, JWT_EXPIRE_HOURS, ONLINE_TIMEOUT_SECONDS
from db.user_db import get_db
from user_models import UserResponse

_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_一-龥]{3,30}$")
_EMAIL_RE = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def _validate_register(username: str, email: str, password: str) -> str | None:
    """Return error message or None if valid."""
    if not _USERNAME_RE.match(username):
        return "用户名需3-30个字符，仅支持字母、数字、下划线和中文"
    if not _EMAIL_RE.match(email):
        return "邮箱格式不正确"
    if not (8 <= len(password) <= 128):
        return "密码不少于8个字符"
    if not (re.search(r"[A-Z]", password) and re.search(r"[a-z]", password) and re.search(r"\d", password)):
        return "密码需同时包含大小写英文字母和数字"
    return None


def register_user(username: str, email: str, password: str) -> UserResponse:
    error = _validate_register(username, email, password)
    if error:
        raise ValueError(error)

    conn = get_db()
    try:
        if conn.execute(
            "SELECT id FROM users WHERE username = ?", [username]
        ).fetchone():
            raise ValueError("用户名已存在")
        if conn.execute(
            "SELECT id FROM users WHERE email = ?", [email]
        ).fetchone():
            raise ValueError("该邮箱已被注册")

        pw_hash = hash_password(password)
        conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            [username, email, pw_hash],
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM users WHERE id = last_insert_rowid()"
        ).fetchone()
        return UserResponse.from_row(row)
    finally:
        conn.close()


def _is_actively_online(row) -> bool:
    """online=1 且最近有心跳（last_activity 新鲜）才算真正在线。

    仅靠 online 布尔不够：浏览器崩溃/断电时 beforeunload 不会触发，
    online 会卡在 1。必须叠加 last_activity 的新鲜度判断，
    超过 ONLINE_TIMEOUT_SECONDS 无心跳即视为离线。
    """
    if not row["online"]:
        return False
    la = row["last_activity"]
    if not la:
        return False
    try:
        last = datetime.strptime(la, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return False
    return (datetime.now() - last).total_seconds() <= ONLINE_TIMEOUT_SECONDS


def login_user(username_or_email: str, password: str) -> tuple[UserResponse, str]:
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? OR email = ?",
            [username_or_email, username_or_email],
        ).fetchone()
        if not row:
            raise ValueError("用户名或密码错误")

        if not verify_password(password, row["password_hash"]):
            raise ValueError("用户名或密码错误")

        # 跨浏览器单会话：账号仍在其他设备活跃时拒绝新登录（先到者优先）。
        # 校验放在密码验证之后，避免泄露"哪些账号在线"。
        if _is_actively_online(row):
            raise ValueError("该账号已在其他设备登录，请先退出后再试")

        # 单会话：每次登录递增 session_version，使旧窗口的 token 立即失效（后登录者踢掉先登录者）。
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            "UPDATE users SET last_login = ?, last_activity = ?, online = 1, "
            "session_version = session_version + 1 WHERE id = ?",
            [now, now, row["id"]],
        )
        session_version = conn.execute(
            "SELECT session_version FROM users WHERE id = ?", [row["id"]]
        ).fetchone()[0]
        # 登录日志：供 admin 登录频率等数据分析使用
        conn.execute(
            "INSERT INTO login_events (user_id, login_at) "
            "VALUES (?, datetime('now','localtime'))",
            [row["id"]],
        )
        conn.commit()

        token = create_access_token(row["id"], row["username"], session_version)
        user = UserResponse.from_row(row)
        user.last_login = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return user, token
    finally:
        conn.close()


def create_access_token(user_id: int, username: str, session_version: int = 0) -> str:
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS)
    payload = {
        "user_id": user_id,
        "username": username,
        "sv": session_version,
        "exp": expire,
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def get_session_version(user_id: int) -> int:
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT session_version FROM users WHERE id = ?", [user_id]
        ).fetchone()
        return row[0] if row else 0
    finally:
        conn.close()


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except JWTError:
        raise ValueError("登录已过期，请重新登录")


def get_user_by_id(user_id: int) -> UserResponse | None:
    conn = get_db()
    try:
        row = conn.execute("SELECT * FROM users WHERE id = ?", [user_id]).fetchone()
        return UserResponse.from_row(row) if row else None
    finally:
        conn.close()


def logout_user(user_id: int):
    """退出登录：清除活跃标记并递增 session_version，使当前 token 立即失效。"""
    conn = get_db()
    try:
        conn.execute(
            "UPDATE users SET last_activity = NULL, online = 0, "
            "session_version = session_version + 1 WHERE id = ?",
            [user_id],
        )
        conn.commit()
    finally:
        conn.close()


def mark_offline(user_id: int):
    """仅将用户标记为离线（online=0），不递增 session_version。

    用于页面关闭/刷新的 beforeunload 信标：只释放 online 占用，
    但保留当前 token 有效——否则刷新页面会把 session_version 递增、
    导致用户被误登出。
    """
    conn = get_db()
    try:
        conn.execute(
            "UPDATE users SET online = 0 WHERE id = ?", [user_id]
        )
        conn.commit()
    finally:
        conn.close()


def check_username_available(username: str) -> bool:
    """用户名是否可用（格式合法且未被占用）。"""
    if not _USERNAME_RE.match(username or ""):
        return False
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT id FROM users WHERE username = ?", [username]
        ).fetchone()
        return row is None
    finally:
        conn.close()


# 两次请求间隔超过该秒数视为会话结束，不累计在线时长
ACTIVITY_GAP_CAP = 1800


def touch_activity(user_id: int):
    """记录用户活跃时间；与上次活跃间隔不超过上限时累计在线时长。"""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT last_activity FROM users WHERE id = ?", [user_id]
        ).fetchone()
        if not row:
            return
        now = datetime.now()
        add = 0
        if row["last_activity"]:
            try:
                last = datetime.strptime(row["last_activity"], "%Y-%m-%d %H:%M:%S")
                gap = (now - last).total_seconds()
                if 0 < gap <= ACTIVITY_GAP_CAP:
                    add = int(gap)
            except ValueError:
                pass
        conn.execute(
            "UPDATE users SET last_activity = ?, online = 1, "
            "active_seconds = active_seconds + ? WHERE id = ?",
            [now.strftime("%Y-%m-%d %H:%M:%S"), add, user_id],
        )
        conn.commit()
    finally:
        conn.close()


def admin_reset_password(user_id: int, new_password: str):
    """管理员重置用户密码。"""
    if len(new_password) < 6 or len(new_password) > 128:
        raise ValueError("新密码需6-128个字符")
    conn = get_db()
    try:
        row = conn.execute("SELECT id FROM users WHERE id = ?", [user_id]).fetchone()
        if not row:
            raise ValueError("用户不存在")
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            [hash_password(new_password), user_id],
        )
        conn.commit()
    finally:
        conn.close()
