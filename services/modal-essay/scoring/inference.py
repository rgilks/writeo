"""Model inference utilities."""

from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer  # type: ignore[import-untyped]

    ModelType: TypeAlias = PreTrainedModel
    TokenizerType: TypeAlias = PreTrainedTokenizer
else:
    ModelType: TypeAlias = Any
    TokenizerType: TypeAlias = Any


def encode_input(
    question_text: str, answer_text: str, tokenizer: TokenizerType
) -> dict[str, torch.Tensor]:
    """Encode input text for model inference."""
    input_text = f"{question_text}\n\n{answer_text}"
    result = tokenizer(
        input_text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    return dict(result)


def run_model_inference(
    model: ModelType,
    encoded_input: dict[str, torch.Tensor],
) -> np.ndarray:
    """Run model inference and return logits."""
    # Move input to same device as model
    device = next(model.parameters()).device
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**encoded_input)
        logits = outputs.logits.squeeze()

    # Convert to numpy array
    logits_np: np.ndarray = logits.cpu().numpy()
    return logits_np
