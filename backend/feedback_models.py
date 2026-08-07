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
    browser_info: str
    status: str
    admin_note: str
    created_at: str

    @classmethod
    def from_row(cls, row):
        return cls(**dict(row))


class StatsResponse(BaseModel):
    total: int
    pending: int
    investigating: int
    resolved: int
    today: int
    by_project: dict
