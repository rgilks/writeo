"""Model loading utilities."""

import os
import time
from typing import TYPE_CHECKING, Any

import torch
from huggingface_hub import list_repo_files, snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import DEFAULT_MODEL, MODEL_CONFIGS, MODEL_PATH

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

    ModelType = PreTrainedModel
    TokenizerType = PreTrainedTokenizer
else:
    ModelType = Any
    TokenizerType = Any

# Global model storage (loaded on first call, per model)
_models: dict[str, tuple[ModelType, TokenizerType]] = {}


def load_tokenizer(model_name: str, model_path: str) -> TokenizerType:
    """Load tokenizer from cache or HuggingFace."""
    tokenizer_start = time.time()
    try:
        tokenizer_path = os.path.join(model_path, "tokenizer_config.json")
        if os.path.exists(tokenizer_path) or os.path.exists(os.path.join(model_path, "vocab.json")):
            print(f"📥 Loading tokenizer from cache: {model_path}")
            tokenizer: TokenizerType = AutoTokenizer.from_pretrained(
                model_path, local_files_only=True
            )  # type: ignore[no-untyped-call]
            tokenizer_time = time.time() - tokenizer_start
            print(f"✅ Tokenizer loaded from cache in {tokenizer_time:.2f}s")
            return tokenizer
        else:
            print("📥 Loading tokenizer from HuggingFace...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)  # type: ignore[no-untyped-call]
            tokenizer_time = time.time() - tokenizer_start
            print(f"✅ Tokenizer loaded from HuggingFace in {tokenizer_time:.2f}s")
            return tokenizer
    except Exception as e:
        print(f"⚠️ WARNING: Error loading tokenizer for {model_name}: {e}")
        if "distilbert" in model_name.lower():
            print("⚠️ Attempting workaround: using base DistilBERT tokenizer...")
            try:
                tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")  # type: ignore[no-untyped-call]
                print("✅ Using base DistilBERT tokenizer (workaround)")
                return tokenizer
            except Exception as e2:
                print(f"❌ ERROR: Failed to load base DistilBERT tokenizer: {e2}")
                raise RuntimeError(
                    f"Failed to load tokenizer for {model_name} and fallback also failed: {e2}"
                ) from e2
        else:
            print(f"❌ ERROR: Tokenizer loading failed for {model_name}: {e}")
            raise RuntimeError(f"Failed to load tokenizer for {model_name}: {e}") from e


def check_model_repo(model_name: str) -> None:
    """Check if model repository has weight files."""
    try:
        repo_files = list(list_repo_files(model_name))
        print(f"Files in model repo: {repo_files}")
        has_weights = any(
            f.endswith((".bin", ".safetensors", ".pt", ".pth", ".ckpt", ".msgpack"))
            for f in repo_files
        )
        if not has_weights:
            print("WARNING: No model weight files found in repository!")
            print("The model repository appears to be missing weight files.")
            print("Attempting to load anyway - transformers may download from cache or base model.")
    except Exception as check_error:
        print(f"Error checking repo files: {check_error}")


def download_model(model_name: str, model_path: str) -> None:
    """Download model repository to cache."""
    try:
        snapshot_download(repo_id=model_name, local_dir=model_path, local_dir_use_symlinks=False)
    except Exception as download_error:
        print(f"Snapshot download warning: {download_error}")


def load_model_from_hf(model_name: str) -> ModelType:
    """Load model from HuggingFace, trying safetensors first."""
    model: ModelType | None = None
    for use_safetensors in [True, False]:
        try:
            print(f"Attempting to load model {model_name} (safetensors={use_safetensors})...")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=False,
                use_safetensors=use_safetensors,
            )
            print(f"Model {model_name} loaded successfully (safetensors={use_safetensors})!")
            break
        except Exception as load_error:
            print(f"Loading with safetensors={use_safetensors} failed: {load_error}")
            if not use_safetensors:
                raise
    if model is None:
        raise RuntimeError(f"Failed to load model {model_name}")
    return model


def save_model_to_cache(
    model: ModelType,
    tokenizer: TokenizerType,
    model_path: str,
) -> None:
    """Save model and tokenizer to cache volume."""
    save_start = time.time()
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path, safe_serialization=True)
    tokenizer.save_pretrained(model_path)
    save_time = time.time() - save_start
    print(f"💾 Model saved to cache in {save_time:.2f}s")


def setup_gpu(model: ModelType, model_name: str, load_start: float) -> None:
    """Move model to GPU and warm up if available."""
    gpu_start = time.time()
    if torch.cuda.is_available():
        model = model.cuda()  # type: ignore[call-arg]
        gpu_move_time = time.time() - gpu_start
        print(f"🚀 Model moved to GPU in {gpu_move_time:.2f}s")
        warmup_start = time.time()
        try:
            dummy_input_ids = torch.randint(0, 1000, (1, 10), device="cuda")
            dummy_attention_mask = torch.ones(1, 10, device="cuda")
            with torch.no_grad():
                _ = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
            torch.cuda.empty_cache()
            warmup_time = time.time() - warmup_start
            total_load_time = time.time() - load_start
            print(f"🔥 GPU warmed up in {warmup_time:.2f}s")
            print(f"✅ Model {model_name} fully loaded in {total_load_time:.2f}s")
        except Exception as warmup_error:
            total_load_time = time.time() - load_start
            print(f"⚠️  GPU warmup failed (non-critical): {warmup_error}")
            print(f"✅ Model {model_name} loaded on GPU in {total_load_time:.2f}s (warmup skipped)")  # noqa: F841
    else:
        total_load_time = time.time() - load_start
        print(f"✅ Model {model_name} loaded successfully (CPU mode) in {total_load_time:.2f}s")


def load_model(model_key: str | None = None) -> tuple[ModelType, TokenizerType]:
    """Load model and tokenizer, caching on volume."""
    load_start = time.time()
    if model_key is None:
        model_key = DEFAULT_MODEL

    if model_key == "fallback":
        # Fallback mode returns None - callers should handle this case
        return None, None  # type: ignore[return-value]

    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key: {model_key}")

    config = MODEL_CONFIGS[model_key]
    assert isinstance(config, dict), f"Invalid config for model {model_key}"
    model_name = config["name"]
    model_path = os.path.join(MODEL_PATH, model_name.replace("/", "_"))
    print(f"📦 Loading model: {model_key} ({model_name})")

    tokenizer = load_tokenizer(model_name, model_path)

    model_load_start = time.time()
    try:
        if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
            print(f"📥 Loading model from cache: {model_path}")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=True
            )
            model_load_time = time.time() - model_load_start
            print(f"✅ Model loaded from cache in {model_load_time:.2f}s")
        else:
            print(f"Downloading model: {model_name} (key: {model_key})")
            check_model_repo(model_name)
            download_model(model_name, model_path)
            model = load_model_from_hf(model_name)
            save_model_to_cache(model, tokenizer, model_path)
            model_load_time = time.time() - model_load_start
            print(f"✅ Model loaded and cached in {model_load_time:.2f}s")
    except Exception as e:
        print(f"❌ ERROR: Failed to load model {model_name}: {e}")
        print(f"❌ Model key: {model_key}")
        print(f"❌ Model path: {model_path}")
        import traceback

        traceback.print_exc()
        raise RuntimeError(f"Failed to load model {model_name} (key: {model_key}): {e}") from e

    model.eval()
    setup_gpu(model, model_name, load_start)

    return model, tokenizer


def get_model(model_key: str | None = None) -> tuple[ModelType, TokenizerType]:
    """Get or load model (lazy loading)."""
    if model_key is None:
        model_key = DEFAULT_MODEL

    if model_key not in _models:
        _models[model_key] = load_model(model_key)
    return _models[model_key]
