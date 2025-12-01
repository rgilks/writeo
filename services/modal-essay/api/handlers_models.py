"""Model management handlers."""

from typing import Any, TypedDict

from config import DEFAULT_MODEL, MODEL_CONFIGS
from model_loader import get_model
from schemas import ModalRequest
from scoring import score_essay


class ModelStatusDict(TypedDict):
    """Model status information."""

    name: str
    type: str
    status: str
    is_default: bool


class ModelsResponseDict(TypedDict):
    """Response for list models endpoint."""

    models: dict[str, ModelStatusDict]
    default: str


async def handle_list_models() -> ModelsResponseDict:
    """Handle list models endpoint."""
    models_status: dict[str, ModelStatusDict] = {}
    for key, config in MODEL_CONFIGS.items():
        status = "unknown"
        try:
            model, tokenizer = get_model(key)
            status = "loaded" if model is not None and tokenizer is not None else "error"
        except Exception as e:
            status = f"error: {str(e)[:50]}"

        config_dict: dict[str, Any] = config
        model_status: ModelStatusDict = {
            "name": str(config_dict.get("name", "")),
            "type": str(config_dict.get("type", "")),
            "status": status,
            "is_default": key == DEFAULT_MODEL,
        }
        models_status[key] = model_status

    return {"models": models_status, "default": DEFAULT_MODEL}


class ComparisonResultDict(TypedDict, total=False):
    """Comparison result for a single model."""

    TA: float
    CC: float
    Vocab: float
    Grammar: float
    Overall: float
    error: str


class ComparisonResponseDict(TypedDict):
    """Response for compare models endpoint."""

    comparison: dict[str, ComparisonResultDict]


async def handle_compare_models(request: ModalRequest) -> ComparisonResponseDict:
    """Handle compare models endpoint."""
    results: dict[str, ComparisonResultDict] = {}

    # Extract answer from request if available
    if not request.parts or not request.parts[0].answers:
        return {"comparison": results}

    answer = request.parts[0].answers[0]

    for model_key in MODEL_CONFIGS:
        try:
            model, tokenizer = get_model(model_key)
            scores = score_essay(
                answer.question_text, answer.answer_text, model, tokenizer, model_key=model_key
            )
            results[model_key] = scores
        except Exception as e:
            results[model_key] = {"error": str(e)}

    return {"comparison": results}
