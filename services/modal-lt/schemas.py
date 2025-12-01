"""Pydantic schemas for LanguageTool service."""

from pydantic import BaseModel, Field


class CheckRequest(BaseModel):
    """Request model for grammar checking.

    Attributes:
        language: Language code for grammar checking (e.g., "en-GB", "en-US", "en").
                 Defaults to "en-GB" (British English).
        text: Text to check for grammar and language errors. Must not be empty.
        answer_id: Optional identifier for tracking purposes (e.g., answer UUID).
    """

    language: str = Field(
        default="en-GB",
        description="Language code for grammar checking",
        examples=["en-GB", "en-US", "en", "de-DE", "fr-FR"],
    )
    text: str = Field(
        ...,
        description="Text to check for grammar and language errors",
        min_length=1,
        examples=["I goes to the store. He don't like it."],
    )
    answer_id: str | None = Field(
        default=None,
        description="Optional identifier for tracking purposes",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )


__all__ = ["CheckRequest"]
