from pydantic import BaseModel
from typing import Optional


class FeedbackUpdate(BaseModel):
    status: Optional[str] = None
    admin_note: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: int
    project: str
    category: str
    severity: str
    title: str
    description: str
    steps_to_reproduce: str
    screenshot: str
    tester_name: str
    user_id: int | None = None
    username: str = ""
    browser_info: str
    status: str
    admin_note: str
    created_at: str

    @classmethod
    def from_row(cls, row):
        d = dict(row)
        d.setdefault("user_id", None)
        d.setdefault("username", "")
        return cls(**d)


class StatsResponse(BaseModel):
    total: int
    pending: int
    investigating: int
    resolved: int
    today: int
    by_project: dict
