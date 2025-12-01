"""Essay quality analysis utilities."""

# Quality score calculation constants
MIN_WORD_COUNT_THRESHOLD = 30
MIN_AVG_WORD_LENGTH = 4.0
MIN_VOCAB_DIVERSITY = 0.3
MIN_COMPLEX_WORD_RATIO = 0.15

# Quality estimate weights
QUALITY_WEIGHTS = {
    "length": 0.1,
    "complexity": 0.2,
    "diversity": 0.5,
    "complex_words": 0.2,
}


def calculate_length_score(word_count: int) -> float:
    """Calculate length-based quality score."""
    if word_count < MIN_WORD_COUNT_THRESHOLD:
        return (word_count / MIN_WORD_COUNT_THRESHOLD) * 0.3
    return min(1.0, 0.3 + (word_count - MIN_WORD_COUNT_THRESHOLD) / 120.0)


def calculate_complexity_score(avg_word_length: float) -> float:
    """Calculate complexity-based quality score."""
    if avg_word_length < MIN_AVG_WORD_LENGTH:
        return (avg_word_length / MIN_AVG_WORD_LENGTH) * 0.5
    return min(1.0, 0.5 + (avg_word_length - MIN_AVG_WORD_LENGTH) / 2.0)


def calculate_diversity_score(vocab_diversity: float) -> float:
    """Calculate vocabulary diversity score."""
    if vocab_diversity < MIN_VOCAB_DIVERSITY:
        return (vocab_diversity / MIN_VOCAB_DIVERSITY) * 0.4
    return min(1.0, 0.4 + (vocab_diversity - MIN_VOCAB_DIVERSITY) / 0.2)


def calculate_complex_word_score(complex_word_ratio: float) -> float:
    """Calculate complex word ratio score."""
    if complex_word_ratio < MIN_COMPLEX_WORD_RATIO:
        return (complex_word_ratio / MIN_COMPLEX_WORD_RATIO) * 0.4
    return min(1.0, 0.4 + (complex_word_ratio - MIN_COMPLEX_WORD_RATIO) / 0.15)


def apply_quality_scaling(quality_estimate: float) -> float:
    """Apply non-linear scaling to quality estimate."""
    if quality_estimate < 0.3:
        return quality_estimate * 0.5
    elif quality_estimate < 0.5:
        return 0.15 + (quality_estimate - 0.3) * 0.7
    elif quality_estimate > 0.7:
        scaled = 0.5 + (quality_estimate - 0.7) * 1.67
        return min(1.0, scaled)
    return quality_estimate


def analyze_essay_quality(text: str) -> dict[str, float]:
    """Analyze essay characteristics to help with calibration."""
    words = text.split()
    word_count = len(words)

    if word_count == 0:
        return {
            "word_count": 0,
            "quality_estimate": 0.0,
            "avg_word_length": 0.0,
            "vocab_diversity": 0.0,
            "complex_word_ratio": 0.0,
        }

    avg_word_length = sum(len(w) for w in words) / word_count
    unique_words = len({w.lower() for w in words})
    vocab_diversity = unique_words / word_count
    complex_words = sum(1 for w in words if len(w) >= 6)
    complex_word_ratio = complex_words / word_count

    length_score = calculate_length_score(word_count)
    complexity_score = calculate_complexity_score(avg_word_length)
    diversity_score = calculate_diversity_score(vocab_diversity)
    complex_word_score = calculate_complex_word_score(complex_word_ratio)

    quality_estimate = (
        QUALITY_WEIGHTS["length"] * length_score
        + QUALITY_WEIGHTS["complexity"] * complexity_score
        + QUALITY_WEIGHTS["diversity"] * diversity_score
        + QUALITY_WEIGHTS["complex_words"] * complex_word_score
    )

    quality_estimate = apply_quality_scaling(quality_estimate)

    return {
        "word_count": word_count,
        "quality_estimate": quality_estimate,
        "avg_word_length": avg_word_length,
        "vocab_diversity": vocab_diversity,
        "complex_word_ratio": complex_word_ratio,
    }
