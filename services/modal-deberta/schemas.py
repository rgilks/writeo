"""Request/response schemas for DeBERTa AES API."""

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    """Request body for essay scoring."""

    text: str = Field(..., description="Essay text to score", min_length=10)
    max_length: int = Field(512, description="Maximum token length", ge=128, le=512)


class ScoreResponse(BaseModel):
    """Response body for essay scoring."""

    score: float = Field(..., description="CEFR numeric score (2.0-8.5)")
    cefr_level: str = Field(..., description="CEFR level (A2-C2)")
    model: str = Field(..., description="Model identifier")
    confidence: float | None = Field(None, description="Confidence score (0-1)")


class HealthResponse(BaseModel):
    """Response body for health check."""

    status: str
    model: str
    gpu_available: bool


class ErrorResponse(BaseModel):
    """Response body for errors."""

    error: str
    detail: str | None = None
