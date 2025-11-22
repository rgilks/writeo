"""Pydantic schemas for LanguageTool service."""

from pydantic import BaseModel
from typing import Optional


class CheckRequest(BaseModel):
    """Request model for grammar checking."""

    language: str = "en-GB"
    text: str
    answer_id: Optional[str] = None

