"""Modal FastAPI service for T-AES-FEEDBACK model."""

import os
from typing import Any

import modal

# Modal app configuration
APP_NAME = "writeo-feedback"
VOLUME_NAME = "writeo-feedback-models"
VOLUME_MOUNT = "/checkpoints"

# Function configuration
TIMEOUT_SECONDS = 60
GPU_TYPE = "T4"
MEMORY_MB = 4096
SCALEDOWN_WINDOW_SECONDS = 30

# Remote paths
REMOTE_APP_PATH = "/app"

# Modal app and image setup
app = modal.App(APP_NAME)

# Create image with dependencies (same as training)
base_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi==0.104.1",
    "uvicorn==0.24.0",
    "transformers==4.35.0",
    "torch==2.1.0",
    "scikit-learn==1.3.2",
    "pydantic==2.5.0",
    "sentencepiece==0.1.99",  # Required for DeBERTa tokenizer
)

# Create volume for model storage (same as training)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

# Add the current directory to the image
image = base_image.add_local_dir(
    os.path.dirname(__file__), remote_path=REMOTE_APP_PATH, copy=True
)


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    timeout=TIMEOUT_SECONDS,
    gpu=GPU_TYPE,
    memory=MEMORY_MB,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    secrets=[modal.Secret.from_name("api-key")],
)
@modal.asgi_app()
def fastapi_app() -> Any:
    """FastAPI app for T-AES-FEEDBACK scoring."""
    import sys

    sys.path.insert(0, REMOTE_APP_PATH)

    from api import create_fastapi_app

    return create_fastapi_app()
