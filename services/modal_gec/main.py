"""Modal FastAPI service for GEC (Grammatical Error Correction).

Uses Seq2Seq model (google/flan-t5-base) for grammar correction,
with ERRANT for edit extraction.
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
APP_NAME = "writeo-gec-service"
VOLUME_NAME = "writeo-gec-models"
VOLUME_MOUNT = "/checkpoints"

# Dependencies
# Uses Python 3.11 matching original
PYTHON_VERSION = "3.11"
PIP_PACKAGES = [
    "torch==2.1.0",
    "transformers==4.36.0",
    "accelerate==0.25.0",
    "errant==3.0.0",
    "spacy==3.7.2",
    "fastapi[standard]",
    "sentencepiece>=0.1.99",
    "pydantic>=2.5.0",
    "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl",
]

# Create app using factory
app, image = ModalServiceFactory.create_app(
    name=APP_NAME,
    image_python_version=PYTHON_VERSION,
    pip_packages=PIP_PACKAGES,
    app_dir=os.path.dirname(__file__),
)


@app.function(
    **ModalServiceFactory.get_default_function_kwargs(
        image=image,
        volume_name=VOLUME_NAME,
        volume_mount=VOLUME_MOUNT,
        gpu="A10G",
        timeout=600,
        memory=8192,  # Explicitly setting higher memory for A10G/LLM
        scaledown_window=60,
        secrets=[modal.Secret.from_name("MODAL_API_KEY")],
    )
)
@modal.asgi_app()
def fastapi_app() -> Any:
    """FastAPI app for GEC (Grammatical Error Correction) service."""
    import sys

    if "/app" not in sys.path:
        sys.path.insert(0, "/app")

    from api import create_fastapi_app

    return create_fastapi_app()
