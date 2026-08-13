"""User service: password hashing, JWT, registration, login."""

import re
import bcrypt
from datetime import datetime, timedelta
from jose import jwt, JWTError

from config import JWT_SECRET_KEY, JWT_ALGORITHM, JWT_EXPIRE_HOURS
from user_db import get_db
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
    if len(password) < 6 or len(password) > 128:
        return "密码需6-128个字符"
    return None


def register_user(username: str, email: str, password: str) -> UserResponse:
    error = _validate_register(username, email, password)
    if error:
        raise ValueError(error)

    conn = get_db()
    try:
        existing = conn.execute(
            "SELECT id FROM users WHERE username = ? OR email = ?",
            [username, email],
        ).fetchone()
        if existing:
            raise ValueError("用户名或邮箱已被注册")

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

        conn.execute(
            "UPDATE users SET last_login = datetime('now','localtime') WHERE id = ?",
            [row["id"]],
        )
        conn.commit()

        token = create_access_token(row["id"], row["username"])
        user = UserResponse.from_row(row)
        user.last_login = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return user, token
    finally:
        conn.close()


def create_access_token(user_id: int, username: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS)
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": expire,
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


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
