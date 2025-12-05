"""
Quick test training run on Modal to verify pipeline works.

Trains for just 2 epochs on small batch to catch any issues
before running full training.
"""

import modal
from pathlib import Path

# Modal setup (same as main# Modal setup
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.0",
        "transformers==4.36.0",
        "accelerate==0.25.0",
        "scikit-learn==1.3.2",
        "numpy==1.26.2",
        "sentencepiece>=0.1.99",  # Required for DeBERTa tokenizer
    )
    .add_local_dir(str(Path(__file__).parent), remote_path="/training")
)

app = modal.App("writeo-feedback-test", image=image)


@app.function(
    gpu="A10G",
    timeout=600,  # 10 minutes max for test
    # No secrets needed - DeBERTa-v3 is public
)
def test_training():
    """Quick test run to verify pipeline."""
    import sys

    sys.path.append("/training")

    import torch
    from torch.optim import AdamW
    from transformers import AutoTokenizer
    from pathlib import Path

    from feedback_model import FeedbackModel, MultiTaskLoss
    from feedback_dataset import create_dataloaders

    print("=" * 80)
    print("T-AES-FEEDBACK TEST RUN")
    print("=" * 80)

    # Test configuration (minimal)
    config = {
        "model_name": "microsoft/deberta-v3-base",
        "max_length": 512,
        "batch_size": 8,  # Smaller batch
        "learning_rate": 2e-5,
        "num_epochs": 2,  # Just 2 epochs
        "cefr_weight": 1.0,
        "span_weight": 0.5,
        "error_type_weight": 0.3,
    }

    print("\nTest Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # Data loaders
    print("\nCreating data loaders...")
    train_loader, dev_loader = create_dataloaders(
        train_file=Path("/training/data-enhanced/train-enhanced.jsonl"),
        dev_file=Path("/training/data-enhanced/dev-enhanced.jsonl"),
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        max_length=config["max_length"],
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Dev batches: {len(dev_loader)}")

    # Model
    print("\nCreating model...")
    model = FeedbackModel(model_name=config["model_name"])
    model = model.to(device)

    # Loss & optimizer
    loss_fn = MultiTaskLoss(
        cefr_weight=config["cefr_weight"],
        span_weight=config["span_weight"],
        error_type_weight=config["error_type_weight"],
    )
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

    print("\n" + "=" * 80)
    print("STARTING TEST TRAINING")
    print("=" * 80)

    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

        # Train on first 10 batches only
        model.train()
        train_losses = []

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 10:  # Just 10 batches for test
                break

            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            targets = {
                "cefr_score": batch["cefr_score"].to(device),
                "span_labels": batch["span_labels"].to(device),
                "error_type_labels": batch["error_type_labels"].to(device),
            }

            # Forward
            outputs = model(input_ids, attention_mask)
            loss, metrics = loss_fn(outputs, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(metrics["total"])
            print(f"  Batch {batch_idx + 1}/10: Loss = {metrics['total']:.4f}")

        avg_loss = sum(train_losses) / len(train_losses)
        print(f"  Average loss: {avg_loss:.4f}")

        # Quick validation
        model.eval()
        dev_losses = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dev_loader):
                if batch_idx >= 5:  # Just 5 batches
                    break

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                targets = {
                    "cefr_score": batch["cefr_score"].to(device),
                    "span_labels": batch["span_labels"].to(device),
                    "error_type_labels": batch["error_type_labels"].to(device),
                }

                outputs = model(input_ids, attention_mask)
                loss, metrics = loss_fn(outputs, targets)

                dev_losses.append(metrics["total"])

        avg_dev_loss = sum(dev_losses) / len(dev_losses)
        print(f"  Dev loss: {avg_dev_loss:.4f}")

    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE - Pipeline works!")
    print("=" * 80)

    return {
        "status": "success",
        "final_train_loss": avg_loss,
        "final_dev_loss": avg_dev_loss,
    }


@app.local_entrypoint()
def main():
    """Run test training."""
    print("Starting test training on Modal...")
    print("This will take ~5-10 minutes")
    print("")

    result = test_training.remote()

    print("\n" + "=" * 80)
    print("TEST RESULTS:")
    print("=" * 80)
    print(result)
    print("\nIf successful, ready to run full training!")
