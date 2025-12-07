"""Pydantic schemas for Modal service request/response."""

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class DimensionsDict(TypedDict, total=False):
    """Essay scoring dimensions."""

    TA: float
    CC: float
    Vocab: float
    Grammar: float
    Overall: float


class ModalAnswer(BaseModel):
    """Answer data with question and answer text for scoring."""

    id: str = Field(
        ...,
        description="Unique identifier for the answer (UUID)",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    question_id: str = Field(
        ...,
        description="Unique identifier for the question (UUID)",
        examples=["660e8400-e29b-41d4-a716-446655440000"],
    )
    question_text: str = Field(
        ...,
        description="The question text that the answer responds to",
        examples=["Describe your weekend. What did you do?"],
        max_length=10000,
    )
    answer_text: str = Field(
        ...,
        description="The student's answer text to be scored",
        examples=[
            "I went to the park yesterday and played football with my friends. It was a beautiful sunny day."
        ],
        max_length=50000,
    )

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "question_id": "660e8400-e29b-41d4-a716-446655440000",
                "question_text": "Describe your weekend. What did you do?",
                "answer_text": "I went to the park yesterday and played football with my friends. It was a beautiful sunny day.",
            }
        }


class ModalPart(BaseModel):
    """A part of the submission containing multiple answers."""

    part: int = Field(..., description="Part number (typically 1 or 2)", examples=[1], ge=1)
    answers: list[ModalAnswer] = Field(
        ..., description="List of answers in this part", min_length=1
    )


class ModalRequest(BaseModel):
    """Request model for essay scoring."""

    submission_id: str = Field(
        ...,
        description="Unique identifier for the submission (UUID)",
        examples=["770e8400-e29b-41d4-a716-446655440000"],
    )
    parts: list[ModalPart] = Field(
        ..., description="List of submission parts to be scored", min_length=1
    )

    class Config:
        json_schema_extra = {
            "example": {
                "submission_id": "770e8400-e29b-41d4-a716-446655440000",
                "parts": [
                    {
                        "part": "1",
                        "answers": [
                            {
                                "id": "550e8400-e29b-41d4-a716-446655440000",
                                "question_id": "660e8400-e29b-41d4-a716-446655440000",
                                "question_text": "Describe your weekend. What did you do?",
                                "answer_text": "I went to the park yesterday and played football with my friends. It was a beautiful sunny day.",
                            }
                        ],
                    }
                ],
            }
        }


class LanguageToolError(BaseModel):
    """LanguageTool error structure."""

    start: int = Field(..., description="Character offset (0-based)", examples=[10])
    end: int = Field(..., description="Character offset (exclusive)", examples=[14])
    length: int = Field(..., description="end - start (for convenience)", examples=[4])
    sentenceIndex: int | None = Field(
        None, description="Optional: sentence number (0-based)", examples=[0]
    )
    category: str = Field(
        ..., description="Error category (e.g., GRAMMAR, TYPOGRAPHY, STYLE)", examples=["GRAMMAR"]
    )
    rule_id: str = Field(..., description="LanguageTool rule identifier", examples=["SVA"])
    message: str = Field(
        ...,
        description="Human-readable error message",
        examples=["Possible subject–verb agreement error."],
    )
    suggestions: list[str] | None = Field(
        None, description="Array of suggested corrections (top 3-5)", examples=[["go", "went"]]
    )
    source: Literal["LT"] = Field("LT", description="Always 'LT' for LanguageTool")
    severity: Literal["warning", "error"] = Field(
        ..., description="Error severity level", examples=["warning"]
    )


class AssessorResult(BaseModel):
    """Result from an assessor (scoring model)."""

    id: str = Field(..., description="Assessor identifier", examples=["T-AES-ESSAY"])
    name: str = Field(..., description="Human-readable assessor name", examples=["Essay scorer"])
    type: Literal["grader", "conf", "ard", "feedback"] = Field(
        ...,
        description="Type of assessor: grader (scores essays), conf (confidence), ard (automated response detection), feedback (grammar errors)",
    )
    overall: float | None = Field(
        None, description="Overall band score (0-9, 0.5 increments)", examples=[7.5], ge=0.0, le=9.0
    )
    label: str | None = Field(
        None, description="CEFR level label (A2, B1, B2, C1, C2)", examples=["C1"]
    )
    dimensions: DimensionsDict | None = Field(
        None,
        description="Detailed scores by dimension (TA, CC, Vocab, Grammar, Overall)",
        examples=[{"TA": 7.5, "CC": 7.0, "Vocab": 8.0, "Grammar": 7.5, "Overall": 7.5}],
    )
    errors: list[LanguageToolError] | None = Field(
        None, description="LanguageTool errors (for type: 'feedback')"
    )
    meta: dict[str, Any] | None = Field(None, description="Assessor metadata")


class AnswerResult(BaseModel):
    """Answer result with assessor results."""

    id: str = Field(
        ..., description="Answer ID (UUID)", examples=["550e8400-e29b-41d4-a716-446655440000"]
    )
    assessor_results: list[AssessorResult] = Field(
        ...,
        alias="assessorResults",
        serialization_alias="assessorResults",
        description="List of assessor results for this answer",
        min_length=1,
    )


class AssessmentPart(BaseModel):
    """Assessment results for a single part of the submission."""

    part: int = Field(..., description="Part number", examples=[1], ge=1)
    status: Literal["success", "error"] = Field(..., description="Processing status for this part")
    answers: list[AnswerResult] = Field(
        ..., description="List of answer results with assessor results", min_length=1
    )

    class Config:
        populate_by_name = True


class AssessmentResults(BaseModel):
    """Complete assessment results for a submission."""

    status: Literal["success", "error", "pending", "bypassed"] = Field(
        ..., description="Overall processing status"
    )
    results: dict[str, list[AssessmentPart]] | None = Field(
        None,
        description="Assessment results organized by parts. Contains a 'parts' key with list of AssessmentPart objects.",
    )
    error_message: str | None = Field(
        None,
        description="Error message if status is 'error'",
        examples=["RuntimeError: Failed to load model engessay"],
    )
    meta: dict[str, Any] | None = Field(
        None, description="Metadata (e.g., answer texts for frontend)"
    )


def map_score_to_cefr(overall: float) -> str:
    """Map overall band score to CEFR level."""
    if overall >= 8.5:
        return "C2"
    elif overall >= 7.0:
        return "C1"
    elif overall >= 5.5:
        return "B2"
    elif overall >= 4.0:
        return "B1"
    else:
        return "A2"
