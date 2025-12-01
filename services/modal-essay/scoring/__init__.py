"""Essay scoring module."""

import traceback
from typing import TYPE_CHECKING, Any

import numpy as np

from config import DEFAULT_MODEL, MODEL_CONFIGS
from schemas import DimensionsDict

from .calibration import calibrate_from_corpus
from .dimension_mapping import map_distilbert_to_dimensions, map_engessay_to_assessment
from .inference import encode_input, run_model_inference
from .logits_processing import (
    normalize_distilbert_score,
    process_distilbert_logits,
    process_engessay_logits,
)
from .quality_analysis import analyze_essay_quality

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

    ModelType = PreTrainedModel
    TokenizerType = PreTrainedTokenizer
else:
    ModelType = Any
    TokenizerType = Any


def process_engessay_scoring(logits_np: np.ndarray, answer_text: str) -> DimensionsDict:
    """Process Engessay model logits into dimension scores."""
    raw_scores = process_engessay_logits(logits_np)

    # Ensure we have exactly 6 scores (pad or truncate if needed)
    if len(raw_scores) != 6:
        print(f"⚠️ Expected 6 scores but got {len(raw_scores)}, padding or truncating")
        if len(raw_scores) < 6:
            default_score = raw_scores[-1] if raw_scores else 3.0
            raw_scores.extend([default_score] * (6 - len(raw_scores)))
        else:
            raw_scores = raw_scores[:6]

    essay_quality = analyze_essay_quality(answer_text)
    word_count = essay_quality["word_count"]
    vocab_diversity = essay_quality["vocab_diversity"]

    # Convert from 1-5 scale to 2-9 scale: 2.0 + (score - 1.0) * (7.0 / 4.0)
    base_scores = [
        2.0 + (max(1.0, min(5.0, float(score))) - 1.0) * (7.0 / 4.0) for score in raw_scores
    ]

    avg_base_score = sum(base_scores) / len(base_scores)
    calibrated_avg = calibrate_from_corpus(avg_base_score, int(word_count), float(vocab_diversity))

    scores, _ = map_engessay_to_assessment(raw_scores, calibrated_avg, avg_base_score)
    return scores


def process_distilbert_scoring(logits_np: np.ndarray) -> DimensionsDict:
    """Process DistilBERT model scoring."""
    raw_score = process_distilbert_logits(logits_np)
    overall_score = normalize_distilbert_score(raw_score)
    return map_distilbert_to_dimensions(overall_score)


def score_essay(
    question_text: str,
    answer_text: str,
    model: ModelType,
    tokenizer: TokenizerType,
    model_key: str | None = None,
) -> DimensionsDict:
    """Score essay using the scoring model."""
    if model is None or tokenizer is None:
        raise ValueError(
            f"Model or tokenizer is None. Model must be loaded before scoring. Model key: {model_key}"
        )

    try:
        encoded_input = encode_input(question_text, answer_text, tokenizer)
        logits_np = run_model_inference(model, encoded_input)

        model_key = model_key or DEFAULT_MODEL
        config: dict[str, Any] = MODEL_CONFIGS.get(model_key, MODEL_CONFIGS[DEFAULT_MODEL])

        if config.get("type") == "roberta" and config.get("output_dims") == 6:
            return process_engessay_scoring(logits_np, answer_text)
        else:
            return process_distilbert_scoring(logits_np)

    except Exception as e:
        print(f"❌ ERROR: Failed to score essay: {e}")
        print(f"❌ Model key: {model_key}")
        print(f"❌ Question length: {len(question_text)} chars")
        print(f"❌ Answer length: {len(answer_text)} chars")
        traceback.print_exc()
        raise RuntimeError(f"Failed to score essay with model {model_key}: {e}") from e
