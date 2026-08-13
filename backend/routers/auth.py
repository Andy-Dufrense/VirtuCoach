"""Authentication router — factory function pattern following existing routers."""

from fastapi import APIRouter, HTTPException, Depends

from user_models import UserRegisterRequest, UserLoginRequest, TokenResponse
from user_service import register_user, login_user
from auth_deps import get_current_user


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

    return router
