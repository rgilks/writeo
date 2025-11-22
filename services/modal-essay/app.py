"""Modal FastAPI service for essay scoring."""

import os
import modal  # type: ignore

# Modal app and image setup
app = modal.App("writeo-essay")

# Get path to shared package
shared_py_path = os.path.join(os.path.dirname(__file__), "../../packages/shared/py")

# Create image with dependencies
base_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi==0.104.1",
    "uvicorn==0.24.0",
    "transformers>=4.40.0",
    "torch==2.1.0",
    "numpy>=1.24.0,<2.0",  # torch 2.1.0 requires numpy <2.0
    "pydantic==2.5.0",
    "sentencepiece==0.1.99",
    "safetensors==0.4.2",
    "huggingface-hub>=0.20.0",
)

# Install shared package if it exists
if os.path.exists(shared_py_path):
    image = base_image.add_local_dir(
        shared_py_path, remote_path="/pkg/writeo-shared", copy=True
    ).run_commands("cd /pkg/writeo-shared && pip install -e .")
else:
    image = base_image

# Create volume for model caching
volume = modal.Volume.from_name("writeo-models", create_if_missing=True)

# Add the current directory to the image so we can import modules
image = image.add_local_dir(os.path.dirname(__file__), remote_path="/app", copy=True)


@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=60,
    gpu="T4",  # GPU acceleration for transformer models
    memory=4096,  # 4GB memory for GPU models
    scaledown_window=30,  # Keep warm for 30s
    # Note: MODAL_API_KEY secret is optional - middleware handles missing key gracefully
    # secrets=[modal.Secret.from_name("MODAL_API_KEY")],
)
@modal.asgi_app()
def fastapi_app():
    """FastAPI app for grading endpoint."""
    import sys

    sys.path.insert(0, "/app")

    from api import create_fastapi_app

    return create_fastapi_app()
