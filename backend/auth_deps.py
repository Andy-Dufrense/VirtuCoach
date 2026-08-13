"""FastAPI dependencies for JWT authentication.

get_current_user — optional, returns None for guests (backward compatible).
require_user — raises 401 if no valid token.
"""

from fastapi import Header, HTTPException
from typing import Optional

from user_service import decode_access_token, get_user_by_id
from user_models import UserResponse


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
    user = get_user_by_id(payload["user_id"])
    return user


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
    user = get_user_by_id(payload["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在")
    return user
