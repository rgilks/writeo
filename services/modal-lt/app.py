"""Modal FastAPI service for LanguageTool grammar checking."""

import os
from typing import Any

import modal

# Modal app configuration
APP_NAME = "writeo-lt"

# Volume names for LanguageTool caching
VOLUME_CACHE = "writeo-lt-cache"
VOLUME_JAR = "writeo-lt-jar"
VOLUME_NGRAMS = "writeo-lt-ngrams"

# Volume mount paths (must match config.py)
VOLUME_MOUNT_CACHE = "/vol/lt-cache"
VOLUME_MOUNT_JAR = "/vol/lt-jar"
VOLUME_MOUNT_NGRAMS = "/vol/lt-ngrams"

# Function configuration
TIMEOUT_SECONDS = 300  # 5 minutes for LanguageTool initialization
MEMORY_MB = 2048  # 2GB memory (LanguageTool can be memory-intensive)
CPU_COUNT = 2  # 2 CPUs for better performance
SCALEDOWN_WINDOW_SECONDS = 60  # Keep warm for 60s

# Modal app and image setup
app = modal.App(APP_NAME)

# Create image with Python 3.12 and dependencies
# LanguageTool requires Java, so we need to install it
base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("openjdk-17-jre-headless")
    .pip_install(
        "fastapi>=0.104.0",
        "pydantic>=2.5.0",
        "language-tool-python",
    )
)

# Add the current directory to the image so we can import modules
image = base_image.add_local_dir(os.path.dirname(__file__), remote_path="/app", copy=True)

# Create volumes for LanguageTool caching (JAR files, n-grams, cache)
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
    # Note: Using api-key secret (middleware checks MODAL_API_KEY env var)
    secrets=[modal.Secret.from_name("api-key")],
)
@modal.asgi_app()
def fastapi_app() -> Any:
    """FastAPI app for LanguageTool grammar checking endpoint."""
    # Import inside function to avoid loading dependencies at module level
    # This is a Modal best practice for ASGI apps
    import sys

    sys.path.insert(0, "/app")

    from api import create_fastapi_app

    return create_fastapi_app()
