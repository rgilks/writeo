"""Modal FastAPI service for LanguageTool grammar checking."""

import os
from typing import Any

import modal

# Import factory from shared package
try:
    from modal_utils import ModalServiceFactory
except ImportError:
    import sys

    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../packages/shared/py"))
    )
    from modal_utils import ModalServiceFactory  # type: ignore

# Configuration
APP_NAME = "writeo-lt"

# Volume names for LanguageTool caching
VOLUME_CACHE = "writeo-lt-cache"
VOLUME_JAR = "writeo-lt-jar"
VOLUME_NGRAMS = "writeo-lt-ngrams"

# Volume mount paths (must match config.py)
VOLUME_MOUNT_CACHE = "/vol/lt-cache"
VOLUME_MOUNT_JAR = "/vol/lt-jar"
VOLUME_MOUNT_NGRAMS = "/vol/lt-ngrams"

# Dependencies
SYSTEM_PACKAGES = ["openjdk-17-jre-headless"]
PIP_PACKAGES = [
    "fastapi>=0.104.0",
    "pydantic>=2.5.0",
    "language-tool-python",
]

# Create app using factory
# Note: we disable include_shared_package to match original behavior (keep image smaller)
app, image = ModalServiceFactory.create_app(
    name=APP_NAME,
    system_packages=SYSTEM_PACKAGES,
    pip_packages=PIP_PACKAGES,
    include_shared_package=True,
    app_dir=os.path.dirname(__file__),
)

# Custom volume setup
lt_cache_volume = modal.Volume.from_name(VOLUME_CACHE, create_if_missing=True)
lt_jar_volume = modal.Volume.from_name(VOLUME_JAR, create_if_missing=True)
lt_ngram_volume = modal.Volume.from_name(VOLUME_NGRAMS, create_if_missing=True)

custom_volumes = {
    VOLUME_MOUNT_CACHE: lt_cache_volume,
    VOLUME_MOUNT_JAR: lt_jar_volume,
    VOLUME_MOUNT_NGRAMS: lt_ngram_volume,
}


@app.function(  # type: ignore
    **ModalServiceFactory.get_default_function_kwargs(
        image=image,
        volumes=custom_volumes,
        timeout=300,
        memory=2048,
        cpu=2.0,
        scaledown_window=60,
        secrets=[modal.Secret.from_name("MODAL_API_KEY")],
    )
)
@modal.asgi_app()
def fastapi_app() -> Any:
    """FastAPI app for LanguageTool grammar checking endpoint."""
    import sys

    if "/app" not in sys.path:
        sys.path.insert(0, "/app")

    from api import create_fastapi_app

    return create_fastapi_app()
