"""Modal FastAPI service for GEC (Grammatical Error Correction).

Uses Seq2Seq model (google/flan-t5-base) for grammar correction,
with ERRANT for edit extraction.
"""

import os
from typing import Any

import modal

# Configuration
APP_NAME = "writeo-gec-service"
VOLUME_NAME = "writeo-gec-models"
VOLUME_MOUNT = "/checkpoints"
REMOTE_APP_PATH = "/app"

# Function configuration
TIMEOUT_SECONDS = 600
GPU_TYPE = "A10G"
MEMORY_MB = 8192
SCALEDOWN_WINDOW_SECONDS = 30

# Modal app and image setup
app = modal.App(APP_NAME)

# Create image with dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.1.0",
    "transformers==4.36.0",
    "accelerate==0.25.0",
    "errant==3.0.0",
    "spacy==3.7.2",
    "fastapi[standard]",
    "sentencepiece>=0.1.99",
    "pydantic>=2.5.0",
    "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl",
)

# Add the current directory to the image
image = image.add_local_dir(os.path.dirname(__file__), remote_path=REMOTE_APP_PATH, copy=True)

# Create volume for model storage
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    timeout=TIMEOUT_SECONDS,
    gpu=GPU_TYPE,
    memory=MEMORY_MB,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    secrets=[modal.Secret.from_name("MODAL_API_KEY")],
)
@modal.asgi_app()
def fastapi_app() -> Any:
    """FastAPI app for GEC (Grammatical Error Correction) service."""
    import sys

    if REMOTE_APP_PATH not in sys.path:
        sys.path.insert(0, REMOTE_APP_PATH)

    from api import create_fastapi_app

    return create_fastapi_app()
