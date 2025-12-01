"""Dimension mapping utilities."""


def _round_to_half(value: float) -> float:
    """Round value to nearest 0.5 increment."""
    return round(value * 2) / 2


def _clamp_score(value: float, min_val: float = 0.0, max_val: float = 9.0) -> float:
    """Clamp score value between min and max."""
    return max(min_val, min(max_val, value))


def _convert_1to5_to_2to9(score_1to5: float) -> float:
    """Convert score from 1-5 scale to 2-9 scale."""
    score_1to5 = max(1.0, min(5.0, float(score_1to5)))
    return 2.0 + (score_1to5 - 1.0) * (7.0 / 4.0)


# Engessay dimension names
ENGEESSAY_DIMS = [
    "cohesion",
    "syntax",
    "vocabulary",
    "phraseology",
    "grammar",
    "conventions",
]

# Mapping from Engessay dimensions to assessment dimensions
ENGEESSAY_TO_ASSESSMENT = ["CC", "Grammar", "Vocab", "Vocab", "Grammar", "TA"]


def map_engessay_to_assessment(
    raw_scores: list[float], calibrated_avg: float, avg_base_score: float
) -> tuple[dict[str, float], list[float]]:
    """Map Engessay dimensions to assessment dimensions with calibration."""
    calibration_factor = calibrated_avg / avg_base_score if avg_base_score > 0 else 1.0

    # Convert from 1-5 scale to 2-9 scale
    base_scores = [_convert_1to5_to_2to9(score) for score in raw_scores]

    # Apply calibration factor to get Engessay dimension scores
    engessay_scores = {}
    for i, dim in enumerate(ENGEESSAY_DIMS):
        base_score = base_scores[i]
        calibrated_score = _clamp_score(base_score * calibration_factor)
        engessay_scores[dim] = calibrated_score

    # Map Engessay dimensions to assessment dimensions and average
    scores = {"TA": 0.0, "CC": 0.0, "Vocab": 0.0, "Grammar": 0.0}
    dim_counts = {"TA": 0, "CC": 0, "Vocab": 0, "Grammar": 0}

    for eng_dim, assessment_dim in zip(ENGEESSAY_DIMS, ENGEESSAY_TO_ASSESSMENT, strict=True):
        scores[assessment_dim] += engessay_scores[eng_dim]
        dim_counts[assessment_dim] += 1

    # Calculate averages, round to 0.5 increments, and clamp to valid range
    for dim in scores:
        if dim_counts[dim] > 0:
            scores[dim] = scores[dim] / dim_counts[dim]
        scores[dim] = _clamp_score(_round_to_half(scores[dim]))

    # Calculate overall as average of all dimensions
    overall_score = sum(scores.values()) / len(scores)
    scores["Overall"] = _clamp_score(_round_to_half(overall_score))

    return scores, base_scores


def map_distilbert_to_dimensions(overall_score: float) -> dict[str, float]:
    """Map DistilBERT single score to multiple dimensions."""
    dimension_names = ["TA", "CC", "Vocab", "Grammar", "Overall"]
    variations = [0.0, -0.5, 0.0, -0.5, 0.0]

    scores = {}
    for i, dim in enumerate(dimension_names):
        dim_score = overall_score + variations[i]
        scores[dim] = _clamp_score(_round_to_half(dim_score))

    scores["Overall"] = overall_score
    return scores
