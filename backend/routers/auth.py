"""Authentication router — factory function pattern following existing routers."""

from fastapi import APIRouter, HTTPException, Depends

from user_models import UserRegisterRequest, UserLoginRequest, TokenResponse
from user_service import (
    register_user, login_user, logout_user, mark_offline,
    check_username_available, touch_activity,
)
from auth_deps import get_current_user, require_user


def create_auth_router():
    router = APIRouter(prefix="/api", tags=["auth"])

    @router.post("/auth/register")
    async def register(body: UserRegisterRequest):
        try:
            user = register_user(body.username, body.email, body.password)
            return {"ok": True, "user": user}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.post("/auth/login")
    async def login(body: UserLoginRequest):
        try:
            user, token = login_user(body.username, body.password)
            return TokenResponse(access_token=token, user=user)
        except ValueError as e:
            raise HTTPException(status_code=401, detail=str(e))

    @router.get("/auth/me")
    async def me(current_user=Depends(get_current_user)):
        if current_user is None:
            raise HTTPException(status_code=401, detail="请先登录")
        return current_user

    @router.post("/auth/logout")
    async def logout(current_user=Depends(require_user)):
        logout_user(current_user.id)
        return {"ok": True}

    @router.post("/auth/heartbeat")
    async def heartbeat(current_user=Depends(require_user)):
        # 每次心跳刷新 last_activity 并置 online=1，保持"在线"状态新鲜
        touch_activity(current_user.id)
        return {"ok": True}

    @router.post("/auth/offline")
    async def offline(current_user=Depends(require_user)):
        # 页面关闭/刷新信标：仅释放 online 占用，不使 token 失效
        mark_offline(current_user.id)
        return {"ok": True}

    @router.get("/auth/check-username")
    async def check_username(username: str):
        return {"available": check_username_available(username)}

    return router
