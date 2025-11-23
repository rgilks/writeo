"""Score calibration utilities."""


def calibrate_short_essay(model_score: float, word_count: int) -> float:
    """Calibrate scores for very short essays (<50 words)."""
    if model_score > 6.0:
        return max(2.0, min(3.5, model_score * 0.5))
    else:
        return max(2.0, min(3.5, model_score * 0.6))


def calibrate_medium_short_essay(model_score: float, word_count: int) -> float:
    """Calibrate scores for short essays (50-100 words)."""
    if model_score < 3.5 or model_score < 4.0:
        return max(2.0, min(3.5, model_score))
    elif model_score < 5.0:
        return max(4.0, min(5.0, model_score + 0.6))
    else:
        return max(4.0, min(5.0, model_score - 1.0))


def calibrate_high_quality_medium(model_score: float, vocab_diversity: float) -> float | None:
    """Calibrate scores for high-quality medium essays."""
    if vocab_diversity >= 0.80:
        if model_score >= 7.0:
            return max(8.5, min(9.0, model_score + 0.5))
        elif model_score >= 6.0:
            return max(8.5, min(9.0, model_score + 1.5))
        else:
            return max(8.5, min(9.0, model_score + 2.0))
    elif vocab_diversity >= 0.75:
        if model_score >= 7.0:
            return max(7.0, min(8.5, model_score))
        elif model_score >= 6.0:
            return max(7.0, min(8.5, model_score + 1.0))
        else:
            return max(7.0, min(8.5, model_score + 1.5))
    return None


def calibrate_medium_essay(model_score: float, vocab_diversity: float) -> float:
    """Calibrate scores for medium essays (100-200 words)."""
    high_quality_result = calibrate_high_quality_medium(model_score, vocab_diversity)
    if high_quality_result is not None:
        return high_quality_result

    if model_score > 6.5:
        return max(4.5, min(5.5, model_score - 1.5))
    elif model_score > 5.5:
        return max(4.5, min(5.5, model_score - 1.0))
    elif model_score > 4.5:
        return max(5.5, min(6.5, model_score - 0.25))
    elif model_score >= 4.0:
        return max(4.0, min(5.0, model_score))
    else:
        return max(4.0, min(5.0, model_score + 0.6))


def calibrate_long_essay(model_score: float, word_count: int, vocab_diversity: float) -> float:
    """Calibrate scores for long essays (200+ words)."""
    if model_score >= 8.0:
        if word_count >= 200 and vocab_diversity >= 0.7:
            return max(8.5, min(9.0, model_score))
        return max(7.0, min(8.5, model_score))
    elif model_score >= 7.0:
        if model_score >= 7.5 and word_count >= 250 and vocab_diversity >= 0.8:
            return max(8.5, min(9.0, model_score + 0.5))
        return max(7.0, min(8.5, model_score))
    elif model_score >= 6.5:
        return max(5.5, min(6.5, model_score - 0.25))
    elif model_score >= 5.0:
        return max(4.5, min(6.5, model_score - 0.5))
    else:
        return max(4.0, min(5.0, model_score + 0.6))


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
