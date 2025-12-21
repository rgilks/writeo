"""Model loading for DeBERTa-v3-large AES model."""

import os
import time
from typing import TYPE_CHECKING, Any, TypeAlias

import torch  # type: ignore[import-untyped]
from transformers import AutoTokenizer  # type: ignore[import-untyped]

from config import MODEL_PATH

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer  # type: ignore[import-untyped]

    from model import DeBERTaAESModel

    ModelType: TypeAlias = DeBERTaAESModel
    TokenizerType: TypeAlias = PreTrainedTokenizer
else:
    ModelType: TypeAlias = Any
    TokenizerType: TypeAlias = Any

# Global model storage (loaded once)
_model: ModelType | None = None
_tokenizer: TokenizerType | None = None


def load_model_from_path(path: str) -> tuple[ModelType, TokenizerType]:
    """Load model from a specific path with optimization."""
    print(f"📦 Loading DeBERTa-v3 AES model from {path}")

    if not os.path.exists(path):
        raise RuntimeError(f"Model not found at {path}")

    # Load tokenizer
    print("📥 Loading tokenizer...")
    start = time.time()
    tokenizer: TokenizerType = AutoTokenizer.from_pretrained(path)
    print(f"✅ Tokenizer loaded in {time.time() - start:.2f}s")

    # Load model
    print("📥 Loading model...")
    model_start = time.time()

    # Import here to avoid circular imports
    from model import DeBERTaAESModel

    # Load model efficiently
    # Detect device and precision
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"⚙️  Using device: {device}, dtype: {dtype}")

    # Initialize model structure
    # Initialize model structure with base config to ensure correct architecture (1024 vs 1536 mismatch fix)
    # We ignore the local config.json which might be incorrect
    model = DeBERTaAESModel(model_name="microsoft/deberta-v3-large")

    # Load state dict
    state_dict_path = os.path.join(path, "pytorch_model.bin")
    state_dict = torch.load(state_dict_path, map_location="cpu")  # Load to RAM first
    model.load_state_dict(state_dict)

    model.eval()

    # Move to GPU and cast to half precision if CUDA
    if device == "cuda":
        model = model.cuda().to(dtype)

        # Warmup
        try:
            print("Toasting GPU (Warmup)...")
            dummy_input = torch.randint(0, 1000, (1, 128), device="cuda")
            dummy_mask = torch.ones(1, 128, device="cuda")
            with torch.no_grad():
                _ = model(input_ids=dummy_input, attention_mask=dummy_mask)
            print("🔥 GPU warmed up")
        except Exception as e:
            print(f"⚠️ Warmup failed: {e}")

    print(f"✅ Model loaded in {time.time() - model_start:.2f}s")
    return model, tokenizer


def set_global_model(model: ModelType, tokenizer: TokenizerType) -> None:
    """Set the global model instance (called from Modal class)."""
    global _model, _tokenizer
    _model = model
    _tokenizer = tokenizer


def get_model() -> tuple[ModelType, TokenizerType]:
    """Get or load model (lazy loading fallback)."""
    global _model, _tokenizer

    if _model is None or _tokenizer is None:
        print("⚠️  Global model not found. Performing lazy load (slow path)...")
        _model, _tokenizer = load_model()

    return _model, _tokenizer


def load_model() -> tuple[ModelType, TokenizerType]:
    """Legacy load function (wrapper)."""

    return load_model_from_path(MODEL_PATH)
