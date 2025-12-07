"""Model loading for DeBERTa-v3-large AES model."""

import os
import time
from typing import TYPE_CHECKING, Any, TypeAlias

import torch  # type: ignore[import-untyped]
from transformers import AutoTokenizer  # type: ignore[import-untyped]

from config import MODEL_NAME, MODEL_PATH

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


def load_model() -> tuple[ModelType, TokenizerType]:
    """Load DeBERTa-v3-large AES model from Modal volume."""
    load_start = time.time()

    print(f"📦 Loading DeBERTa-v3 AES model from {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. "
            "Model may not be trained yet. Run training script first."
        )

    # Load tokenizer
    print("📥 Loading tokenizer...")
    tokenizer_start = time.time()
    tokenizer: TokenizerType = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer_time = time.time() - tokenizer_start
    print(f"✅ Tokenizer loaded in {tokenizer_time:.2f}s")

    # Load model
    print(f"📥 Loading model from {MODEL_PATH}...")
    model_start = time.time()

    # Import here to avoid circular imports
    from model import DeBERTaAESModel

    # Load the full model state
    model = DeBERTaAESModel(model_name=MODEL_NAME)
    model_state = torch.load(os.path.join(MODEL_PATH, "pytorch_model.bin"), map_location="cpu")
    model.load_state_dict(model_state)
    model.eval()

    model_time = time.time() - model_start
    print(f"✅ Model loaded in {model_time:.2f}s")

    # Move to GPU if available
    gpu_start = time.time()
    if torch.cuda.is_available():
        model = model.cuda()
        gpu_move_time = time.time() - gpu_start
        print(f"🚀 Model moved to GPU in {gpu_move_time:.2f}s")

        # Warmup
        try:
            warmup_start = time.time()
            dummy_input_ids = torch.randint(0, 1000, (1, 128), device="cuda")
            dummy_attention_mask = torch.ones(1, 128, device="cuda")
            with torch.no_grad():
                _ = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
            torch.cuda.empty_cache()
            warmup_time = time.time() - warmup_start
            print(f"🔥 GPU warmed up in {warmup_time:.2f}s")
        except Exception as e:
            print(f"⚠️  GPU warmup warning: {e}")

    total_time = time.time() - load_start
    device = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"✅ DeBERTa-v3 AES fully loaded on {device} in {total_time:.2f}s")

    return model, tokenizer


def get_model() -> tuple[ModelType, TokenizerType]:
    """Get or load model (lazy loading with caching)."""
    global _model, _tokenizer

    if _model is None or _tokenizer is None:
        _model, _tokenizer = load_model()

    return _model, _tokenizer
