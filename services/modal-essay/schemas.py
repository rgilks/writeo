"""Schema imports with fallback definitions."""

import os
import sys
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

# Try importing from shared package first, then fall back to inline definitions
_shared_py_path = os.path.join(os.path.dirname(__file__), "../../packages/shared/py")
_imported = False

if os.path.exists(_shared_py_path):
    sys.path.insert(0, _shared_py_path)
    try:
        # Import from schemas module (DimensionsDict is in schemas.py but not in __init__.py)
        from schemas import (  # type: ignore[attr-defined]
            AnswerResult,
            AssessmentPart,
            AssessmentResults,
            AssessorResult,
            DimensionsDict,
            ModalRequest,
            map_score_to_cefr,
        )

        _imported = True
    except ImportError:
        pass

# If import failed, define schemas inline
if not _imported:

    class DimensionsDict(TypedDict, total=False):
        """Essay scoring dimensions."""

        TA: float
        CC: float
        Vocab: float
        Grammar: float
        Overall: float

    class ModalAnswer(BaseModel):  # type: ignore[no-redef]
        id: str
        question_id: str
        question_text: str
        answer_text: str

    class ModalPart(BaseModel):  # type: ignore[no-redef]
        part: int
        answers: list[ModalAnswer]

    class ModalRequest(BaseModel):  # type: ignore[no-redef]
        submission_id: str
        template: dict[str, Any]
        parts: list[ModalPart]

    class AssessorResult(BaseModel):  # type: ignore[no-redef]
        id: str
        name: str
        type: Literal["grader", "conf", "ard"]
        overall: float | None = None
        label: str | None = None
        dimensions: dict[str, float] | None = None

    class AnswerResult(BaseModel):  # type: ignore[no-redef]
        model_config = ConfigDict(populate_by_name=True)

        id: str
        assessor_results: list[AssessorResult] = Field(
            ..., alias="assessorResults", serialization_alias="assessorResults"
        )

    class AssessmentPart(BaseModel):  # type: ignore[no-redef]
        part: int
        status: Literal["success", "error"]
        answers: list[AnswerResult]

    class AssessmentResults(BaseModel):  # type: ignore[no-redef]
        status: Literal["success", "error", "pending", "bypassed"]
        results: dict[str, list[AssessmentPart]] | None = None
        template: dict[str, Any]
        error_message: str | None = None

    def map_score_to_cefr(overall: float) -> str:  # type: ignore[no-redef]
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


__all__ = [
    "ModalRequest",
    "AssessmentResults",
    "AssessmentPart",
    "AssessorResult",
    "AnswerResult",
    "DimensionsDict",
    "map_score_to_cefr",
]
