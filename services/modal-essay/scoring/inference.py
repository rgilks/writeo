"""Model inference utilities."""

from typing import Dict, Any, TYPE_CHECKING, Union
import torch  # type: ignore
import numpy as np

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer  # type: ignore
    TokenizerType = PreTrainedTokenizer
else:
    TokenizerType = Any


def encode_input(question_text: str, answer_text: str, tokenizer: TokenizerType) -> Dict[str, torch.Tensor]:
    """Encode input text for model inference."""
    input_text = f"{question_text}\n\n{answer_text}"
    return tokenizer(
        input_text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )


def run_model_inference(
    model: Union[Any, "PreTrainedModel"],  # type: ignore
    encoded_input: Dict[str, torch.Tensor]
) -> np.ndarray:
    """Run model inference and return logits."""
    if next(model.parameters()).is_cuda:
        encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model(**encoded_input)
        logits = outputs.logits.squeeze()
    
    logits_np = logits.cpu().numpy() if hasattr(logits, 'cpu') else logits.numpy()
    print(f"Model output shape: {logits_np.shape}, dtype: {logits_np.dtype}")
    print(f"Model output sample: {logits_np.flatten()[:10]}")
    print(f"Model output range: min={np.min(logits_np):.2f}, max={np.max(logits_np):.2f}, mean={np.mean(logits_np):.2f}")
    return logits_np

