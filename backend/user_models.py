"""Pydantic models for user authentication, following feedback_models.py pattern."""

from pydantic import BaseModel
from typing import Optional


class UserRegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class UserLoginRequest(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: str
    last_login: Optional[str] = None

    @classmethod
    def from_row(cls, row):
        return cls(**dict(row))


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse
