"""Model configuration constants."""

import os
import pathlib

# Default model (can be overridden via environment variable or request)
DEFAULT_MODEL = os.getenv("MODEL_NAME", "engessay")

MODEL_CONFIGS = {
    "engessay": {
        "name": "KevSun/Engessay_grading_ML",
        "type": "roberta",
        "output_dims": 6,
        "dimension_mapping": {
            "cohesion": "CC",
            "syntax": "Grammar",
            "vocabulary": "Vocab",
            "phraseology": "Vocab",
            "grammar": "Grammar",
            "conventions": "TA",
        },
    },
    "distilbert": {
        "name": "Michau96/distilbert-base-uncased-essay_scoring",
        "type": "distilbert",
        "output_dims": 1,
        "dimension_mapping": None,
    },
    "fallback": {
        "name": None,
        "type": "heuristic",
        "output_dims": 0,
        "dimension_mapping": None,
    },
}

# Use local cache directory when running locally, Modal volume path when deployed
if os.path.exists("/vol") and os.access("/vol", os.W_OK):
    MODEL_PATH = "/vol/hf/models"
else:
    # Local development: use user's home directory cache
    MODEL_PATH = str(pathlib.Path.home() / ".cache" / "writeo" / "models")
MAX_CHUNK_TOKENS = 400
CHUNK_OVERLAP = 100
