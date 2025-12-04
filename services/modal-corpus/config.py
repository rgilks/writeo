"""Configuration for corpus-trained CEFR model service."""

MODEL_PATH = "/vol/models/corpus-trained-roberta"  # Correct path on volume
MODEL_NAME = "corpus-roberta"

# CEFR score mapping (matches training data)
CEFR_MAPPING = {
    "A1": 2.0,
    "A1+": 2.5,
    "A2": 3.0,
    "A2+": 3.5,
    "B1": 4.5,
    "B1+": 5.0,
    "B2": 6.0,
    "B2+": 6.5,
    "C1": 7.5,
    "C1+": 8.0,
    "C2": 8.5,
}

# Reverse mapping for score to CEFR
SCORE_TO_CEFR = {v: k for k, v in CEFR_MAPPING.items()}


def score_to_cefr(score: float) -> str:
    """Convert numeric score to CEFR level."""
    # Find closest CEFR level
    closest = min(SCORE_TO_CEFR.keys(), key=lambda x: abs(x - score))
    return SCORE_TO_CEFR[closest]
