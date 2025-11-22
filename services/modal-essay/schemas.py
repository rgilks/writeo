"""Schema imports with fallback definitions."""

import sys
import os
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict

# Try importing from shared package first, then fall back to inline definitions
_imported = False
shared_py_path = os.path.join(os.path.dirname(__file__), "../../packages/shared/py")
if os.path.exists(shared_py_path):
    sys.path.insert(0, shared_py_path)
    try:
        from schemas import (  # type: ignore
            ModalRequest,
            AssessmentResults,
            AssessmentPart,
            AssessorResult,
            AnswerResult,
            map_score_to_cefr,
        )
        _imported = True
    except ImportError:
        pass

# If import failed, define schemas inline
if not _imported:
    class ModalAnswer(BaseModel):  # type: ignore[no-redef]
        id: str
        question_id: str
        question_text: str
        answer_text: str
    
    class ModalPart(BaseModel):  # type: ignore[no-redef]
        part: int
        answers: List[ModalAnswer]
    
    class ModalRequest(BaseModel):  # type: ignore[no-redef]
        submission_id: str
        template: Dict[str, Any]
        parts: List[ModalPart]
    
    class AssessorResult(BaseModel):  # type: ignore[no-redef]
        id: str
        name: str
        type: Literal["grader", "conf", "ard"]
        overall: Optional[float] = None
        label: Optional[str] = None
        dimensions: Optional[Dict[str, float]] = None
    
    class AnswerResult(BaseModel):  # type: ignore[no-redef]
        model_config = ConfigDict(populate_by_name=True)
        
        id: str
        assessor_results: List[AssessorResult] = Field(..., alias="assessor-results", serialization_alias="assessor-results")
    
    class AssessmentPart(BaseModel):  # type: ignore[no-redef]
        part: int
        status: Literal["success", "error"]
        answers: List[AnswerResult]
    
    class AssessmentResults(BaseModel):  # type: ignore[no-redef]
        status: Literal["success", "error", "pending", "bypassed"]
        results: Optional[Dict[str, List[AssessmentPart]]] = None
        template: Dict[str, Any]
        error_message: Optional[str] = None
    
    def map_score_to_cefr(overall: float) -> str:  # type: ignore[no-redef]
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
    "map_score_to_cefr",
]

