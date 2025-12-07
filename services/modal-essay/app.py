"""Modal FastAPI service for essay scoring."""

import os
from typing import Any

import modal

# Import factory from shared package (assumes shared package is installed in environment)
# Note: In local dev, we might need to adjust PYTHONPATH, but inside Modal it works via the package mount
try:
    from modal_utils import ModalServiceFactory
except ImportError:
    # For local dev where shared pkg might not be installed as a package yet
    import sys

    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../packages/shared/py"))
    )
    from modal_utils import ModalServiceFactory

# Configuration
APP_NAME = "writeo-essay"
VOLUME_NAME = "writeo-models"
VOLUME_MOUNT = "/vol"

# Dependencies
PIP_PACKAGES = [
    "fastapi==0.104.1",
    "uvicorn==0.24.0",
    "transformers>=4.40.0",
    "torch==2.2.0",
    "numpy>=1.24.0,<2.0",
    "pydantic==2.5.0",
    "sentencepiece>=0.2.0",
    "safetensors==0.4.2",
    "huggingface-hub>=0.20.0",
]

# Create app using factory
app, image = ModalServiceFactory.create_app(
    name=APP_NAME,
    pip_packages=PIP_PACKAGES,
    # Mount the local app directory so we can import 'api'
    app_dir=os.path.dirname(__file__),
)


@app.function(
    **ModalServiceFactory.get_default_function_kwargs(
        image=image,
        volume_name=VOLUME_NAME,
        volume_mount=VOLUME_MOUNT,
        gpu="T4",
        timeout=60,
        memory=4096,
        scaledown_window=30,
    )
)
@modal.asgi_app()
def fastapi_app() -> Any:
    """FastAPI app for essay scoring service."""
    import sys

    # Ensure app directory is in path (factory mounts it to /app)
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")

    from api import create_fastapi_app

    return create_fastapi_app()
