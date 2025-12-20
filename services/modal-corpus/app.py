"""Modal FastAPI service for corpus-trained CEFR scorer."""

import os
from typing import Any

import modal

# Configuration
APP_NAME = "writeo-corpus"
VOLUME_NAME = "writeo-models"
VOLUME_MOUNT = "/vol"
REMOTE_APP_PATH = "/app"

# Function configuration
TIMEOUT_SECONDS = 60
GPU_TYPE = "T4"
MEMORY_MB = 4096
SCALEDOWN_WINDOW_SECONDS = 30

# Modal app and image setup
app = modal.App(APP_NAME)

# Create image with dependencies
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi==0.104.1",
    "uvicorn==0.24.0",
    "transformers>=4.40.0",
    "torch==2.2.0",
    "numpy>=1.24.0,<2.0",
    "pydantic==2.5.0",
    "safetensors==0.4.2",
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
)
@modal.asgi_app()
def fastapi_app() -> Any:
    """FastAPI app for corpus-trained CEFR scoring."""
    import sys

    if REMOTE_APP_PATH not in sys.path:
        sys.path.insert(0, REMOTE_APP_PATH)

    from api import create_fastapi_app

    return create_fastapi_app()
