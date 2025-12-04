"""Processing for overall score models (single output regression)."""

from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

from schemas import DimensionsDict

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer  # type: ignore[import-untyped]

    ModelType: TypeAlias = PreTrainedModel
    TokenizerType: TypeAlias = PreTrainedTokenizer
else:
    ModelType: TypeAlias = Any
    TokenizerType: TypeAlias = Any


def process_overall_score_scoring(logits_np: np.ndarray, answer_text: str) -> DimensionsDict:
    """Process overall score model logits into dimension scores."""
    # Extract single overall score
    # Handle both scalar and array cases
    overall_score = float(logits_np) if logits_np.shape == () else float(logits_np.squeeze())

    # Clamp to valid range (2.0-9.0)
    overall_score = max(2.0, min(9.0, overall_score))

    # Map overall score to dimensions
    # For now, use the same score for all dimensions
    # This can be refined later with heuristics or additional models
    scores: DimensionsDict = {
        "TA": overall_score,
        "CC": overall_score,
        "Vocab": overall_score,
        "Grammar": overall_score,
        "Overall": overall_score,
    }

    return scores
