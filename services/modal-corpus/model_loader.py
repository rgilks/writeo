"""Model loading for corpus-trained RoBERTa model."""

import os
import time
from typing import TYPE_CHECKING, Any, TypeAlias

import torch  # type: ignore[import-untyped]
from transformers import (  # type: ignore[import-untyped]
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from config import MODEL_NAME, MODEL_PATH

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer  # type: ignore[import-untyped]

    ModelType: TypeAlias = PreTrainedModel
    TokenizerType: TypeAlias = PreTrainedTokenizer
else:
    ModelType: TypeAlias = Any
    TokenizerType: TypeAlias = Any

# Global model storage (loaded once)
_model: ModelType | None = None
_tokenizer: TokenizerType | None = None


def load_model() -> tuple[ModelType, TokenizerType]:
    """Load corpus-trained model from Modal volume."""
    load_start = time.time()

    print(f"📦 Loading corpus-trained model from {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. "
            "Model may not be trained yet. Run training script first."
        )

    # Load tokenizer - use roberta-base (same as training)
    print("📥 Loading tokenizer...")
    tokenizer_start = time.time()
    tokenizer: TokenizerType = AutoTokenizer.from_pretrained("roberta-base")  # type: ignore[no-untyped-call]
    tokenizer_time = time.time() - tokenizer_start
    print(f"✅ Tokenizer loaded in {tokenizer_time:.2f}s")

    # Load model from volume
    print(f"📥 Loading model from {MODEL_PATH}...")
    model_start = time.time()
    model: ModelType = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, local_files_only=True
    )
    model.eval()
    model_time = time.time() - model_start
    print(f"✅ Model loaded in {model_time:.2f}s")

    # Move to GPU if available
    gpu_start = time.time()
    if torch.cuda.is_available():
        model = model.cuda()  # type: ignore[call-arg]
        gpu_move_time = time.time() - gpu_start
        print(f"🚀 Model moved to GPU in {gpu_move_time:.2f}s")

        # Warmup
        try:
            warmup_start = time.time()
            dummy_input_ids = torch.randint(0, 1000, (1, 10), device="cuda")
            dummy_attention_mask = torch.ones(1, 10, device="cuda")
            with torch.no_grad():
                _ = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
            torch.cuda.empty_cache()
            warmup_time = time.time() - warmup_start
            print(f"🔥 GPU warmed up in {warmup_time:.2f}s")
        except Exception as e:
            print(f"⚠️  GPU warmup warning: {e}")

    total_time = time.time() - load_start
    device = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"✅ {MODEL_NAME} fully loaded on {device} in {total_time:.2f}s")

    return model, tokenizer


def get_model() -> tuple[ModelType, TokenizerType]:
    """Get or load model (lazy loading with caching)."""
    global _model, _tokenizer

    if _model is None or _tokenizer is None:
        _model, _tokenizer = load_model()

    return _model, _tokenizer
