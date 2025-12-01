"""Submission processing handler."""

from typing import TYPE_CHECKING, Any

from config import DEFAULT_MODEL, MODEL_CONFIGS
from model_loader import get_model
from schemas import (
    AnswerResult,
    AssessmentPart,
    AssessmentResults,
    AssessorResult,
    DimensionsDict,
    ModalAnswer,
    ModalRequest,
    map_score_to_cefr,
)
from scoring import score_essay

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

    ModelType = PreTrainedModel
    TokenizerType = PreTrainedTokenizer
else:
    ModelType = Any
    TokenizerType = Any


def create_assessor_result(
    scores: DimensionsDict, model_name: str = "Essay scorer"
) -> AssessorResult:
    """Create assessor result from dimension scores."""
    overall = scores.get("Overall", scores.get("overall", 0.0))
    cefr_label = map_score_to_cefr(overall)
    dimensions = {
        "TA": scores.get("TA", overall),
        "CC": scores.get("CC", overall),
        "Vocab": scores.get("Vocab", overall),
        "Grammar": scores.get("Grammar", overall),
        "Overall": overall,
    }
    return AssessorResult(
        id="T-AES-ESSAY",
        name=model_name,
        type="grader",
        overall=overall,
        label=cefr_label,
        dimensions=dimensions,
    )


def process_answer(
    answer: ModalAnswer,
    model: ModelType | None,
    tokenizer: TokenizerType | None,
    model_key: str,
    config: dict[str, Any],
) -> AnswerResult:
    """Process a single answer and return result."""
    if model is None or tokenizer is None:
        raise RuntimeError(f"Model or tokenizer is None for model_key: {model_key}")

    scores = score_essay(
        answer.question_text, answer.answer_text, model, tokenizer, model_key=model_key
    )
    model_name = config.get("name", "Essay scorer")

    assessor_result = create_assessor_result(scores, model_name)
    return AnswerResult(id=answer.id, assessor_results=[assessor_result])


def process_submission(request: ModalRequest, model_key: str) -> AssessmentResults:
    """Process submission and return assessment results."""
    if model_key not in MODEL_CONFIGS:
        model_key = DEFAULT_MODEL

    config: dict[str, Any] = MODEL_CONFIGS[model_key]
    model_name = config.get("name") or "Unknown model"
    print(f"Using model: {model_key} ({model_name})")
    model, tokenizer = get_model(model_key)
    if model is None or tokenizer is None:
        raise RuntimeError(f"Model {model_key} returned None after loading.")

    parts = []
    for part_data in request.parts:
        answer_results = []
        for answer in part_data.answers:
            answer_result = process_answer(answer, model, tokenizer, model_key, config)
            answer_results.append(answer_result)

        assessment_part = AssessmentPart(
            part=part_data.part, status="success", answers=answer_results
        )
        parts.append(assessment_part)

    return AssessmentResults(status="success", results={"parts": parts}, template=request.template)
