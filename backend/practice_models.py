"""Pydantic models for practice sessions, following feedback_models.py pattern."""

from pydantic import BaseModel
from typing import Optional


class PracticeSessionResponse(BaseModel):
    id: int
    user_id: int
    video_filename: str
    instrument: str
    skill_level: str
    chord_or_track: str
    overall_score: float
    audio_score: float
    hand_score: float
    report_text: str
    mode: str
    created_at: str

    @classmethod
    def from_row(cls, row):
        return cls(**dict(row))


class PracticeStatsResponse(BaseModel):
    total_sessions: int
    average_score: float
    highest_score: float
    lowest_score: float
    recent_scores: list  # last 10 scores for trend chart
    recent_7d: int = 0   # sessions in the last 7 days
    by_mode: dict
