"""Essay quality analysis utilities."""

import re


def calculate_length_score(word_count: int) -> float:
    """Calculate length-based quality score."""
    if word_count < 30:
        return word_count / 30.0 * 0.3
    return min(1.0, 0.3 + (word_count - 30) / 120.0)


def calculate_complexity_score(avg_word_length: float) -> float:
    """Calculate complexity-based quality score."""
    if avg_word_length < 4.0:
        return avg_word_length / 4.0 * 0.5
    return min(1.0, 0.5 + (avg_word_length - 4.0) / 2.0)


def calculate_diversity_score(vocab_diversity: float) -> float:
    """Calculate vocabulary diversity score."""
    if vocab_diversity < 0.3:
        return vocab_diversity / 0.3 * 0.4
    return min(1.0, 0.4 + (vocab_diversity - 0.3) / 0.2)


def calculate_complex_word_score(complex_word_ratio: float) -> float:
    """Calculate complex word ratio score."""
    if complex_word_ratio < 0.15:
        return complex_word_ratio / 0.15 * 0.4
    return min(1.0, 0.4 + (complex_word_ratio - 0.15) / 0.15)


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
    avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0

    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    unique_words = len({w.lower() for w in words})
    vocab_diversity = unique_words / word_count if word_count > 0 else 0

    complex_words = sum(1 for w in words if len(w) >= 6)
    complex_word_ratio = complex_words / word_count if word_count > 0 else 0

    length_score = calculate_length_score(word_count)
    complexity_score = calculate_complexity_score(avg_word_length)
    diversity_score = calculate_diversity_score(vocab_diversity)
    complex_word_score = calculate_complex_word_score(complex_word_ratio)

    quality_estimate = (
        0.1 * length_score
        + 0.2 * complexity_score
        + 0.5 * diversity_score
        + 0.2 * complex_word_score
    )

    quality_estimate = apply_quality_scaling(quality_estimate)

    return {
        "word_count": word_count,
        "quality_estimate": quality_estimate,
        "avg_word_length": avg_word_length,
        "vocab_diversity": vocab_diversity,
        "complex_word_ratio": complex_word_ratio,
    }
