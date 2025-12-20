"""Configuration for DeBERTa-v3-large AES model."""

import os
import pathlib

# Model configuration
MODEL_NAME = "microsoft/deberta-v3-large"
MODEL_DISPLAY_NAME = "DeBERTa-v3-large AES"

# Ordinal regression configuration
NUM_THRESHOLDS = 7  # Thresholds for scores 2-8 (A2 to C2)
MIN_SCORE = 2.0
MAX_SCORE = 8.5

# CEFR level thresholds (standard CEFR-to-score mapping)
CEFR_THRESHOLDS = {
    "C2": 8.25,
    "C1+": 7.75,
    "C1": 7.0,
    "B2+": 6.25,
    "B2": 5.5,
    "B1+": 4.75,
    "B1": 4.0,
    "A2+": 3.25,
    "A2": 2.0,
}

# Model storage path
# Model storage path
# Check for baked-in model first (fastest)
if os.path.exists("/model/deberta-v3-aes/pytorch_model.bin"):
    MODEL_PATH = "/model/deberta-v3-aes"
elif os.path.exists("/model/pytorch_model.bin"):
    MODEL_PATH = "/model"
elif os.path.exists("/vol") and os.access("/vol", os.W_OK):
    # Fallback to volume (slow, network read)
    MODEL_PATH = "/vol/models/deberta-v3-aes"
else:
    # Local development
    MODEL_PATH = str(pathlib.Path.home() / ".cache" / "writeo" / "deberta-aes")


def score_to_cefr(score: float) -> str:
    """Map numeric score to CEFR level."""
    for level, threshold in CEFR_THRESHOLDS.items():
        if score >= threshold:
            return level
    return "A2"
