"""Model loader for AES-FEEDBACK."""

from pathlib import Path

import torch
from transformers import AutoTokenizer


def get_feedback_model():
    """
    Load AES-FEEDBACK model from checkpoint.

    Returns:
        tuple: (model, tokenizer)
    """
    print("Loading AES-FEEDBACK model...")

    # Import model class
    from feedback_model import FeedbackModel

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    # Create model
    model = FeedbackModel("microsoft/deberta-v3-base")

    # Load checkpoint
    checkpoint_path = Path("/checkpoints/feedback_model_best.pt")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Make sure the model was trained and saved to the volume."
        )

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"Model moved to GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU")

    model.eval()

    print(f"✅ Model loaded successfully (Epoch {checkpoint.get('epoch', 'unknown')})")

    dev_loss = checkpoint.get("dev_loss", "unknown")
    if isinstance(dev_loss, int | float):
        print(f"   Dev loss: {dev_loss:.4f}")
    else:
        print(f"   Dev loss: {dev_loss}")

    return model, tokenizer
