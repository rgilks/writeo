"""Modal FastAPI service for AES-DEBERTA scoring."""

import os
from typing import Any

import modal

# Modal app configuration
APP_NAME = "writeo-deberta"
VOLUME_NAME = "writeo-deberta-models"
VOLUME_MOUNT = "/vol"

# Function configuration
TIMEOUT_SECONDS = 120  # Longer for cold starts with large model
GPU_TYPE = "A10G"  # 24GB VRAM for DeBERTa-v3-large
MEMORY_MB = 8192
SCALEDOWN_WINDOW_SECONDS = 60  # Keep warm longer

# Remote paths
REMOTE_APP_PATH = "/app"

# Modal app and image setup
app = modal.App(APP_NAME)

# Create image with dependencies
base_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi==0.104.1",
    "uvicorn==0.24.0",
    "transformers>=4.40.0",
    "torch>=2.6.0",
    "pydantic==2.5.0",
    "sentencepiece>=0.1.99",
    "safetensors>=0.4.2",
    "numpy>=1.24.0,<2.0",
)

# Create volume for model storage
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Add the current directory to the image
image = base_image.add_local_dir(os.path.dirname(__file__), remote_path=REMOTE_APP_PATH, copy=True)


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
    """FastAPI app for AES-DEBERTA scoring."""
    import sys

    sys.path.insert(0, REMOTE_APP_PATH)

    from api import create_fastapi_app

    return create_fastapi_app()
