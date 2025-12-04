"""Pydantic schemas for API requests and responses."""

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    """Request for scoring an essay."""

    text: str = Field(..., description="Essay text to score")
    max_length: int = Field(512, description="Maximum sequence length")


class ScoreResponse(BaseModel):
    """Response with CEFR score."""

    score: float = Field(..., description="Numeric CEFR score (2.0-8.5)")
    cefr_level: str = Field(..., description="CEFR level (A1 to C2)")
    model: str = Field(..., description="Model name used for scoring")


class ModelInfo(BaseModel):
    """Information about the scoring model."""

    name: str = Field(..., description="Model name")
    description: str = Field(..., description="Model description")
    version: str = Field(..., description="Model version")
