"""Model configuration constants."""

import os
import pathlib
from typing import Any

# Default model (can be overridden via environment variable or request)
DEFAULT_MODEL = os.getenv("MODEL_NAME", "engessay")

# Engessay dimension mapping
ENGEESSAY_DIMENSION_MAPPING: dict[str, str] = {
    "cohesion": "CC",
    "syntax": "Grammar",
    "vocabulary": "Vocab",
    "phraseology": "Vocab",
    "grammar": "Grammar",
    "conventions": "TA",
}

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "engessay": {
        "name": "KevSun/Engessay_grading_ML",
        "type": "roberta",
        "output_dims": 6,
        "dimension_mapping": ENGEESSAY_DIMENSION_MAPPING,
    },
    "distilbert": {
        "name": "Michau96/distilbert-base-uncased-essay_scoring",
        "type": "distilbert",
        "output_dims": 1,
        "dimension_mapping": None,
    },
}

# Model storage path
# Use Modal volume path when deployed, local cache directory when running locally
VOLUME_PATH = "/vol"
if os.path.exists(VOLUME_PATH) and os.access(VOLUME_PATH, os.W_OK):
    MODEL_PATH = f"{VOLUME_PATH}/hf/models"
else:
    # Local development: use user's home directory cache
    MODEL_PATH = str(pathlib.Path.home() / ".cache" / "writeo" / "models")
