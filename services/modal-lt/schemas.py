"""Pydantic schemas for LanguageTool service."""

from pydantic import BaseModel


class CheckRequest(BaseModel):
    """Request model for grammar checking."""

    language: str = "en-GB"
    text: str
    answer_id: str | None = None
