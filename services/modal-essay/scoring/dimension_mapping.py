"""Dimension mapping utilities."""


def map_engessay_to_assessment(
    raw_scores: list[float], calibrated_avg: float, avg_base_score: float
) -> tuple[dict[str, float], list[float]]:
    """Map Engessay dimensions to assessment dimensions with calibration."""
    engessay_dims = [
        "cohesion",
        "syntax",
        "vocabulary",
        "phraseology",
        "grammar",
        "conventions",
    ]
    engessay_scores = {}

    calibration_factor = calibrated_avg / avg_base_score if avg_base_score > 0 else 1.0

    base_scores = []
    for i, _dim in enumerate(engessay_dims):
        score_1to5 = max(1.0, min(5.0, float(raw_scores[i])))
        base_score = 2.0 + (score_1to5 - 1.0) * (7.0 / 4.0)
        base_scores.append(base_score)

    for i, dim in enumerate(engessay_dims):
        score_1to5 = max(1.0, min(5.0, float(raw_scores[i])))
        base_score = base_scores[i]
        calibrated_score = base_score * calibration_factor
        calibrated_score = max(0.0, min(9.0, calibrated_score))
        engessay_scores[dim] = calibrated_score
        print(
            f"  {dim}: {score_1to5:.2f} (1-5) -> {base_score:.2f} (base score) -> {calibrated_score:.2f} (calibrated, factor={calibration_factor:.3f})"
        )

    assessment_dims = ["CC", "Grammar", "Vocab", "Vocab", "Grammar", "TA"]
    scores = {"TA": 0.0, "CC": 0.0, "Vocab": 0.0, "Grammar": 0.0}
    dim_counts = {"TA": 0, "CC": 0, "Vocab": 0, "Grammar": 0}

    for eng_dim, assessment_dim in zip(engessay_dims, assessment_dims, strict=False):
        scores[assessment_dim] += engessay_scores[eng_dim]
        dim_counts[assessment_dim] += 1

    for dim in scores:
        if dim_counts[dim] > 0:
            scores[dim] = scores[dim] / dim_counts[dim]
        scores[dim] = max(0.0, min(9.0, round(scores[dim] * 2) / 2))

    overall_score = sum(scores.values()) / len(scores)
    overall_score = round(overall_score * 2) / 2
    scores["Overall"] = max(0.0, min(9.0, overall_score))

    return scores, base_scores


def map_distilbert_to_dimensions(overall_score: float) -> dict[str, float]:
    """Map DistilBERT single score to multiple dimensions."""
    dimension_names = ["TA", "CC", "Vocab", "Grammar", "Overall"]
    scores = {}
    variations = [0.0, -0.5, 0.0, -0.5, 0.0]

    for i, dim in enumerate(dimension_names):
        dim_score = overall_score + variations[i]
        scores[dim] = max(0.0, min(9.0, round(dim_score * 2) / 2))

    scores["Overall"] = overall_score
    return scores
