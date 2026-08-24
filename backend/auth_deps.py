"""FastAPI dependencies for JWT authentication.

get_current_user — optional, returns None for guests (backward compatible).
require_user — raises 401 if no valid token.
"""

from fastapi import Header, HTTPException
from typing import Optional

from user_service import (
    decode_access_token,
    get_user_by_id,
    get_session_version,
    touch_activity,
)
from user_models import UserResponse


def _touch(user: Optional[UserResponse]) -> Optional[UserResponse]:
    """认证成功后记录活跃时间（失败不影响认证本身）。"""
    if user:
        try:
            touch_activity(user.id)
        except Exception:
            pass
    return user


def _session_valid(payload: dict) -> bool:
    """旧 token 无 sv 字段视为有效（兼容）；有则须与库中 session_version 一致。"""
    sv = payload.get("sv")
    if sv is None:
        return True
    try:
        return int(sv) == get_session_version(int(payload["user_id"]))
    except Exception:
        return False


async def get_current_user(
    authorization: Optional[str] = Header(None),
) -> Optional[UserResponse]:
    """Extract user from Bearer token. Returns None for guest mode."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization[7:]
    try:
        payload = decode_access_token(token)
    except ValueError:
        return None
    if not _session_valid(payload):
        return None
    user = get_user_by_id(payload["user_id"])
    return _touch(user)


async def require_user(
    authorization: Optional[str] = Header(None),
) -> UserResponse:
    """Extract user from Bearer token. Raises 401 for guests."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="请先登录")
    token = authorization[7:]
    try:
        payload = decode_access_token(token)
    except ValueError:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
    if not _session_valid(payload):
        raise HTTPException(status_code=401, detail="该账号已在其他窗口登录，当前会话已失效")
    user = get_user_by_id(payload["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在")
    return _touch(user)
