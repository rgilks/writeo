"""Modal FastAPI service for LanguageTool grammar checking."""

import os
from typing import Any

import modal

# Configuration
APP_NAME = "writeo-lt"
REMOTE_APP_PATH = "/app"

# Volume names for LanguageTool caching
VOLUME_CACHE = "writeo-lt-cache"
VOLUME_JAR = "writeo-lt-jar"
VOLUME_NGRAMS = "writeo-lt-ngrams"

# Volume mount paths (must match config.py)
VOLUME_MOUNT_CACHE = "/vol/lt-cache"
VOLUME_MOUNT_JAR = "/vol/lt-jar"
VOLUME_MOUNT_NGRAMS = "/vol/lt-ngrams"

# Function configuration
TIMEOUT_SECONDS = 300
MEMORY_MB = 2048
CPU_COUNT = 2.0
SCALEDOWN_WINDOW_SECONDS = 30

# Modal app and image setup
app = modal.App(APP_NAME)

# Create image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("openjdk-17-jre-headless")
    .pip_install(
        "fastapi>=0.104.0",
        "pydantic>=2.5.0",
        "language-tool-python",
    )
)

# Add the current directory to the image
image = image.add_local_dir(os.path.dirname(__file__), remote_path=REMOTE_APP_PATH, copy=True)

# Create volumes for LanguageTool caching
lt_cache_volume = modal.Volume.from_name(VOLUME_CACHE, create_if_missing=True)
lt_jar_volume = modal.Volume.from_name(VOLUME_JAR, create_if_missing=True)
lt_ngram_volume = modal.Volume.from_name(VOLUME_NGRAMS, create_if_missing=True)


@app.function(
    image=image,
    volumes={
        VOLUME_MOUNT_CACHE: lt_cache_volume,
        VOLUME_MOUNT_JAR: lt_jar_volume,
        VOLUME_MOUNT_NGRAMS: lt_ngram_volume,
    },
    timeout=TIMEOUT_SECONDS,
    memory=MEMORY_MB,
    cpu=CPU_COUNT,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    secrets=[modal.Secret.from_name("MODAL_API_KEY")],
)
@modal.asgi_app()
def fastapi_app() -> Any:
    """FastAPI app for LanguageTool grammar checking endpoint."""
    import sys

    if REMOTE_APP_PATH not in sys.path:
        sys.path.insert(0, REMOTE_APP_PATH)

    from api import create_fastapi_app

    return create_fastapi_app()
