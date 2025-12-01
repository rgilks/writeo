"""Modal FastAPI service for essay scoring."""

import os
from typing import Any

import modal

# Modal app configuration
APP_NAME = "writeo-essay"
VOLUME_NAME = "writeo-models"
VOLUME_MOUNT = "/vol"

# Function configuration
TIMEOUT_SECONDS = 60
GPU_TYPE = "T4"
MEMORY_MB = 4096  # 4GB memory for GPU models
SCALEDOWN_WINDOW_SECONDS = 30  # Keep warm for 30s

# Remote paths
REMOTE_APP_PATH = "/app"
REMOTE_SHARED_PKG_PATH = "/pkg/writeo-shared"

# Modal app and image setup
app = modal.App(APP_NAME)

# Get path to shared package
shared_py_path = os.path.join(os.path.dirname(__file__), "../../packages/shared/py")

# Create image with dependencies
base_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi==0.104.1",
    "uvicorn==0.24.0",
    "transformers>=4.40.0",
    "torch==2.2.0",
    "numpy>=1.24.0,<2.0",  # torch requires numpy <2.0
    "pydantic==2.5.0",
    "sentencepiece==0.1.99",
    "safetensors==0.4.2",
    "huggingface-hub>=0.20.0",
)

# Install shared package if it exists
if os.path.exists(shared_py_path):
    image = base_image.add_local_dir(
        shared_py_path, remote_path=REMOTE_SHARED_PKG_PATH, copy=True
    ).run_commands(f"cd {REMOTE_SHARED_PKG_PATH} && pip install -e .")
else:
    image = base_image

# Create volume for model caching
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Add the current directory to the image so we can import modules
image = image.add_local_dir(os.path.dirname(__file__), remote_path=REMOTE_APP_PATH, copy=True)


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    timeout=TIMEOUT_SECONDS,
    gpu=GPU_TYPE,  # GPU acceleration for transformer models
    memory=MEMORY_MB,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    # Note: Using api-key secret (middleware checks MODAL_API_KEY env var)
    secrets=[modal.Secret.from_name("api-key")],
)
@modal.asgi_app()
def fastapi_app() -> Any:
    """FastAPI app for essay scoring service."""
    import sys

    sys.path.insert(0, REMOTE_APP_PATH)

    from api import create_fastapi_app

    return create_fastapi_app()
