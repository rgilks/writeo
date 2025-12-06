"""Modal FastAPI service for GECToR (Fast Grammar Error Correction).

Uses GECToR (Tag, Not Rewrite) approach for ~10x faster inference
compared to Seq2Seq models.
"""

import os
from typing import Any

import modal

# Modal app configuration
APP_NAME = "writeo-gector-service"
VOLUME_NAME = "writeo-gector-models"
VOLUME_MOUNT = "/models"

# Function configuration
TIMEOUT_SECONDS = 300
GPU_TYPE = "T4"  # Cheaper GPU sufficient for encoder-only model
SCALEDOWN_WINDOW_SECONDS = 60  # 1 min keep-warm

# Path for app files
REMOTE_APP_PATH = "/app"

# Modal app and image setup
app = modal.App(APP_NAME)

# Create image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")  # Required for pip install from git
    .pip_install(
        "torch>=2.6.0",  # GECToR requires torch>=2.6.0
        "transformers>=4.36.0",
        "fastapi[standard]",
        "pydantic>=2.5.0",
        "git+https://github.com/gotutiyan/gector.git",  # GECToR package
    )
    .add_local_dir(os.path.dirname(__file__), remote_path=REMOTE_APP_PATH, copy=True)
)

# Create volume for model caching
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    timeout=TIMEOUT_SECONDS,
    gpu=GPU_TYPE,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
)
@modal.asgi_app()
def fastapi_app() -> Any:
    """FastAPI app for GECToR (Fast Grammar Error Correction) service."""
    import sys

    sys.path.insert(0, REMOTE_APP_PATH)
    from api import create_fastapi_app

    return create_fastapi_app()
