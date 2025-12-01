"""Modal FastAPI service for LanguageTool grammar checking."""

import os
from typing import Any

import modal

# Modal app and image setup
app = modal.App("writeo-lt")

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
lt_cache_volume = modal.Volume.from_name("writeo-lt-cache", create_if_missing=True)
lt_jar_volume = modal.Volume.from_name("writeo-lt-jar", create_if_missing=True)
lt_ngram_volume = modal.Volume.from_name("writeo-lt-ngrams", create_if_missing=True)


@app.function(
    image=image,
    volumes={
        "/vol/lt-cache": lt_cache_volume,
        "/vol/lt-jar": lt_jar_volume,
        "/vol/lt-ngrams": lt_ngram_volume,
    },
    timeout=300,  # 5 minutes for LanguageTool initialization
    memory=2048,  # 2GB memory (LanguageTool can be memory-intensive)
    cpu=2,  # 2 CPUs for better performance
    scaledown_window=60,  # Keep warm for 60s
    # Note: Using api-key secret (middleware checks MODAL_API_KEY env var)
    secrets=[modal.Secret.from_name("api-key")],
)
@modal.asgi_app()
def fastapi_app() -> Any:
    """FastAPI app for LanguageTool grammar checking endpoint."""
    import sys

    sys.path.insert(0, "/app")

    from api import create_fastapi_app

    return create_fastapi_app()
