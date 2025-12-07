"""Modal FastAPI service for GECToR (Fast Grammar Error Correction).

Uses GECToR (Tag, Not Rewrite) approach for ~10x faster inference
compared to Seq2Seq models.
"""

import os
from typing import Any
import modal

# Import factory from shared package
try:
    from modal_utils import ModalServiceFactory
except ImportError:
    import sys

    sys.path.append(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../packages/shared/py")
        )
    )
    from modal_utils import ModalServiceFactory

# Configuration
APP_NAME = "writeo-gector-service"
VOLUME_NAME = "writeo-gector-models"
VOLUME_MOUNT = "/models"

# Dependencies
# GECToR requires Python 3.11 and specific torch version
PYTHON_VERSION = "3.11"
SYSTEM_PACKAGES = ["git"]
PIP_PACKAGES = [
    "torch>=2.6.0",
    "transformers>=4.36.0",
    "fastapi[standard]",
    "pydantic>=2.5.0",
    "git+https://github.com/gotutiyan/gector.git",
]

# Create app using factory
app, image = ModalServiceFactory.create_app(
    name=APP_NAME,
    image_python_version=PYTHON_VERSION,
    system_packages=SYSTEM_PACKAGES,
    pip_packages=PIP_PACKAGES,
    app_dir=os.path.dirname(__file__),
    # Original service didn't mount shared package, but we can include it or not.
    # Default is True, keeping it True is standard for new services.
)


@app.function(
    **ModalServiceFactory.get_default_function_kwargs(
        image=image,
        volume_name=VOLUME_NAME,
        volume_mount=VOLUME_MOUNT,
        gpu="T4",
        timeout=300,
        scaledown_window=30,
        # Default memory is 4096MB
    )
)
@modal.asgi_app()
def fastapi_app() -> Any:
    """FastAPI app for GECToR (Fast Grammar Error Correction) service."""
    import sys

    if "/app" not in sys.path:
        sys.path.insert(0, "/app")

    from api import create_fastapi_app

    return create_fastapi_app()
