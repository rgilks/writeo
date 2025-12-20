"""Modal FastAPI service for AES-DEBERTA scoring."""

import os
from typing import Any

import modal

# Modal app configuration
APP_NAME = "writeo-deberta"
# Modal app configuration
APP_NAME = "writeo-deberta"
# Volume is only used for training/storage, not inference anymore (baked in)
VOLUME_NAME = "writeo-deberta-models"
VOLUME_MOUNT = "/vol"

# Function configuration
TIMEOUT_SECONDS = 120
GPU_TYPE = "A10G"  # 24GB VRAM for DeBERTa-v3-large
MEMORY_MB = 16384  # Increased memory for safe loading
SCALEDOWN_WINDOW_SECONDS = 30  # Standard keep-warm time

# Paths
REMOTE_APP_PATH = "/app"
REMOTE_MODEL_PATH = "/model"  # Baked-in model path

# Modal app and image setup
app = modal.App(APP_NAME)

# Create image with dependencies
base_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi==0.104.1",
    "uvicorn==0.24.0",
    "transformers>=4.40.0",
    "torch==2.2.0",
    "pydantic==2.5.0",
    "sentencepiece>=0.1.99",
    "safetensors>=0.4.2",
    "numpy>=1.24.0,<2.0",
)

# Create volume (still useful for training output persistence)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Add application code
image = base_image.add_local_dir(
    os.path.dirname(__file__),
    remote_path=REMOTE_APP_PATH,
    copy=True,
    ignore=[
        "model_local",
        "__pycache__",
        ".git",
    ],  # Avoid recursive copy of model if handled separately or duplicates
)

# Bake model into image if available locally
local_model_path = os.path.join(os.path.dirname(__file__), "model_local")
if os.path.exists(local_model_path):
    print("Found local model, baking into image...")
    image = image.add_local_dir(local_model_path, remote_path=REMOTE_MODEL_PATH)
else:
    print("WARNING: Local model not found. Container will fail if not using volume fallback.")


@app.cls(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    timeout=TIMEOUT_SECONDS,
    gpu=GPU_TYPE,
    memory=MEMORY_MB,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    secrets=[modal.Secret.from_name("api-key")],
)
class DebertaService:
    @modal.enter()
    def load_model(self):
        """Load model into memory on container startup."""
        import sys
        import time

        sys.path.insert(0, REMOTE_APP_PATH)
        from config import MODEL_PATH
        from model_loader import load_model_from_path

        print(f"🔄 Container starting. Loading model from {MODEL_PATH}...")
        start = time.time()

        # Use a new loader function that accepts explicit path
        self.model, self.tokenizer = load_model_from_path(MODEL_PATH)

        print(f"✅ Model loaded and ready in {time.time() - start:.2f}s")

    @modal.asgi_app()
    def fastapi_app(self) -> Any:
        """FastAPI app for AES-DEBERTA scoring."""
        import sys

        sys.path.insert(0, REMOTE_APP_PATH)

        from api import create_fastapi_app
        from model_loader import set_global_model

        # Pass the pre-loaded model/tokenizer to the factory or make them globally accessible?
        # Since ASGI app runs in the same process as the class instance in Modal,
        # we can inject dependencies or use a singleton pattern initialized by load_model.
        # Ideally, we pass "self" to the app creator, but FastAPI needs a callable.
        # Better: use a global in model_loader that we set here.

        set_global_model(self.model, self.tokenizer)

        return create_fastapi_app()
