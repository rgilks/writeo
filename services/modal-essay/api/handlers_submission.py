"""Submission processing handler."""

from typing import Dict, Any
from schemas import (
    ModalRequest,
    AssessmentResults,
    AssessmentPart,
    AssessorResult,
    AnswerResult,
    map_score_to_cefr,
)
from config import DEFAULT_MODEL, MODEL_CONFIGS
from model_loader import get_model
from scoring import score_essay


def get_fallback_scores(answer_text: str) -> Dict[str, float]:
    """Generate fallback heuristic scores."""
    answer_length = len(answer_text.split())
    base_score = min(9.0, max(4.0, answer_length / 50.0 * 2.0 + 5.0))
    return {
        "TA": round(base_score * 2) / 2,
        "CC": round((base_score - 0.5) * 2) / 2,
        "Vocab": round(base_score * 2) / 2,
        "Grammar": round((base_score - 0.5) * 2) / 2,
        "Overall": round(base_score * 2) / 2,
    }


def create_assessor_result(scores: Dict[str, float]) -> AssessorResult:
    """Create assessor result from scores."""
    overall = scores.get("Overall", scores.get("overall", 0.0))
    cefr_label = map_score_to_cefr(overall)
    return AssessorResult(
        id="T-AES-ESSAY",
        name="Essay scorer",
        type="grader",
        overall=overall,
        label=cefr_label,
        dimensions={
            "TA": scores.get("TA", overall),
            "CC": scores.get("CC", overall),
            "Vocab": scores.get("Vocab", overall),
            "Grammar": scores.get("Grammar", overall),
            "Overall": overall,
        }
    )


def process_answer(answer: Any, model: Any, tokenizer: Any, model_key: str) -> AnswerResult:
    """Process a single answer and return result."""
    if model_key == "fallback":
        scores = get_fallback_scores(answer.answer_text)
    else:
        scores = score_essay(
            answer.question_text,
            answer.answer_text,
            model,
            tokenizer,
            model_key=model_key
        )
    
    assessor_result = create_assessor_result(scores)
    return AnswerResult(
        id=answer.id,
        assessor_results=[assessor_result]
    )


def process_submission(request: ModalRequest, model_key: str) -> AssessmentResults:
    """Process submission and return assessment results."""
    if model_key not in MODEL_CONFIGS:
        model_key = DEFAULT_MODEL
    
    config = MODEL_CONFIGS[model_key]
    assert isinstance(config, dict), f"Invalid config for model {model_key}"
    print(f"Using model: {model_key} ({config['name']})")
    
    model = None
    tokenizer = None
    
    if model_key == "fallback":
        print("⚠️ Using fallback heuristic scorer (explicitly requested)")
    else:
        print(f"Loading model: {model_key}...")
        model, tokenizer = get_model(model_key)
        if model is None or tokenizer is None:
            raise RuntimeError(f"Model {model_key} returned None after loading.")
        print(f"✅ Model {model_key} loaded successfully!")
    
    parts = []
    for part_data in request.parts:
        answer_results = []
        for answer in part_data.answers:
            answer_result = process_answer(answer, model, tokenizer, model_key)
            answer_results.append(answer_result)
        
        assessment_part = AssessmentPart(
            part=part_data.part,
            status="success",
            answers=answer_results
        )
        parts.append(assessment_part)
    
    return AssessmentResults(
        status="success",
        results={"parts": parts},
        template=request.template
    )

