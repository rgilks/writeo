"""Essay scoring module."""

from typing import Dict, Optional, Any, TYPE_CHECKING
import numpy as np
from config import DEFAULT_MODEL, MODEL_CONFIGS
from schemas import map_score_to_cefr, DimensionsDict
from .inference import encode_input, run_model_inference
from .logits_processing import process_engessay_logits, process_distilbert_logits, normalize_distilbert_score
from .quality_analysis import analyze_essay_quality
from .calibration import calibrate_from_corpus
from .dimension_mapping import map_engessay_to_assessment, map_distilbert_to_dimensions

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer  # type: ignore
    ModelType = PreTrainedModel
    TokenizerType = PreTrainedTokenizer
else:
    ModelType = Any
    TokenizerType = Any


def process_engessay_scoring(logits_np: np.ndarray, answer_text: str) -> DimensionsDict:
    """Process Engessay model scoring."""
    raw_scores = process_engessay_logits(logits_np)
    if len(raw_scores) != 6:
        print(f"⚠️ Expected 6 scores but got {len(raw_scores)}, padding or truncating")
        if len(raw_scores) < 6:
            raw_scores.extend([raw_scores[-1] if raw_scores else 3.0] * (6 - len(raw_scores)))
        else:
            raw_scores = raw_scores[:6]
    
    essay_quality = analyze_essay_quality(answer_text)
    print(f"Essay quality analysis: {essay_quality}")
    
    word_count = essay_quality["word_count"]
    vocab_diversity = essay_quality["vocab_diversity"]
    
    base_scores = []
    for score_1to5 in raw_scores:
        score_1to5 = max(1.0, min(5.0, float(score_1to5)))
        base_score = 2.0 + (score_1to5 - 1.0) * (7.0 / 4.0)
        base_scores.append(base_score)
    
    avg_base_score = sum(base_scores) / len(base_scores)
    calibrated_avg = calibrate_from_corpus(avg_base_score, word_count, vocab_diversity)
    
    scores, _ = map_engessay_to_assessment(raw_scores, calibrated_avg, avg_base_score)
    print(f"Final scores: {scores}")
    print(f"Overall score: {scores['Overall']:.2f}, CEFR: {map_score_to_cefr(scores['Overall'])}")
    return scores


def process_distilbert_scoring(logits_np: np.ndarray) -> DimensionsDict:
    """Process DistilBERT model scoring."""
    raw_score = process_distilbert_logits(logits_np)
    print(f"DistilBERT raw score: {raw_score}, logits shape: {logits_np.shape}")
    overall_score = normalize_distilbert_score(raw_score)
    print(f"DistilBERT normalized overall score: {overall_score}")
    return map_distilbert_to_dimensions(overall_score)


def score_essay(
    question_text: str,
    answer_text: str,
    model: ModelType,
    tokenizer: TokenizerType,
    model_key: Optional[str] = None
) -> DimensionsDict:
    """Score essay using the scoring model."""
    if model is None or tokenizer is None:
        raise ValueError(f"Model or tokenizer is None. Model must be loaded before scoring. Model key: {model_key}")
    
    try:
        encoded_input = encode_input(question_text, answer_text, tokenizer)
        logits_np = run_model_inference(model, encoded_input)
        
        if model_key is None:
            model_key = DEFAULT_MODEL
        
        config = MODEL_CONFIGS.get(model_key, MODEL_CONFIGS["distilbert"])
        assert isinstance(config, dict), f"Invalid config for model {model_key}"
        
        if config["type"] == "roberta" and config["output_dims"] == 6:
            return process_engessay_scoring(logits_np, answer_text)
        else:
            return process_distilbert_scoring(logits_np)
        
    except Exception as e:
        print(f"❌ ERROR: Failed to score essay: {e}")
        print(f"❌ Model key: {model_key}")
        print(f"❌ Question length: {len(question_text)} chars")
        print(f"❌ Answer length: {len(answer_text)} chars")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to score essay with model {model_key}: {e}") from e
