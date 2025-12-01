"""Score calibration utilities."""


def _clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


def calibrate_short_essay(model_score: float, word_count: int) -> float:
    """Calibrate scores for very short essays (<50 words)."""
    multiplier = 0.5 if model_score > 6.0 else 0.6
    return _clamp(model_score * multiplier, 2.0, 3.5)


def calibrate_medium_short_essay(model_score: float, word_count: int) -> float:
    """Calibrate scores for short essays (50-100 words)."""
    if model_score < 4.0:
        return _clamp(model_score, 2.0, 3.5)
    elif model_score < 5.0:
        return _clamp(model_score + 0.6, 4.0, 5.0)
    else:
        return _clamp(model_score - 1.0, 4.0, 5.0)


def calibrate_high_quality_medium(model_score: float, vocab_diversity: float) -> float | None:
    """Calibrate scores for high-quality medium essays."""
    if vocab_diversity >= 0.80:
        if model_score >= 7.0:
            return _clamp(model_score + 0.5, 8.5, 9.0)
        elif model_score >= 6.0:
            return _clamp(model_score + 1.5, 8.5, 9.0)
        else:
            return _clamp(model_score + 2.0, 8.5, 9.0)
    elif vocab_diversity >= 0.75:
        if model_score >= 7.0:
            return _clamp(model_score, 7.0, 8.5)
        elif model_score >= 6.0:
            return _clamp(model_score + 1.0, 7.0, 8.5)
        else:
            return _clamp(model_score + 1.5, 7.0, 8.5)
    return None


def calibrate_medium_essay(model_score: float, vocab_diversity: float) -> float:
    """Calibrate scores for medium essays (100-200 words)."""
    high_quality_result = calibrate_high_quality_medium(model_score, vocab_diversity)
    if high_quality_result is not None:
        return high_quality_result

    if model_score > 6.5:
        return _clamp(model_score - 1.5, 4.5, 5.5)
    elif model_score > 5.5:
        return _clamp(model_score - 1.0, 4.5, 5.5)
    elif model_score > 4.5:
        return _clamp(model_score - 0.25, 5.5, 6.5)
    elif model_score >= 4.0:
        return _clamp(model_score, 4.0, 5.0)
    else:
        return _clamp(model_score + 0.6, 4.0, 5.0)


def calibrate_long_essay(model_score: float, word_count: int, vocab_diversity: float) -> float:
    """Calibrate scores for long essays (200+ words)."""
    if model_score >= 8.0:
        if word_count >= 200 and vocab_diversity >= 0.7:
            return _clamp(model_score, 8.5, 9.0)
        return _clamp(model_score, 7.0, 8.5)
    elif model_score >= 7.0:
        if model_score >= 7.5 and word_count >= 250 and vocab_diversity >= 0.8:
            return _clamp(model_score + 0.5, 8.5, 9.0)
        return _clamp(model_score, 7.0, 8.5)
    elif model_score >= 6.5:
        return _clamp(model_score - 0.25, 5.5, 6.5)
    elif model_score >= 5.0:
        return _clamp(model_score - 0.5, 4.5, 6.5)
    else:
        return _clamp(model_score + 0.6, 4.0, 5.0)


def calibrate_from_corpus(model_score: float, word_count: int, vocab_diversity: float) -> float:
    """Calibrate model scores to match corpus findings."""
    if word_count < 50:
        return calibrate_short_essay(model_score, word_count)
    elif word_count < 100:
        return calibrate_medium_short_essay(model_score, word_count)
    elif word_count < 200:
        return calibrate_medium_essay(model_score, vocab_diversity)
    else:
        return calibrate_long_essay(model_score, word_count, vocab_diversity)
