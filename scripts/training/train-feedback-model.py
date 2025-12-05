"""
Train T-AES-FEEDBACK model on Modal GPU.

Multi-task training for CEFR scoring + error detection.
"""

import modal
from pathlib import Path

# Modal setup
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

app = modal.App("writeo-feedback-training", image=image)


@app.function(
    gpu="A10G",  # NVIDIA A10G (24GB VRAM)
    timeout=7200,  # 2 hours
    # No secrets needed - DeBERTa-v3 is public
)
def train_feedback_model():
    """Train multi-task feedback model."""
    import sys

    sys.path.append("/training")

    import torch
    from torch.optim import AdamW
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from pathlib import Path

    from feedback_model import FeedbackModel, MultiTaskLoss
    from feedback_dataset import create_dataloaders

    print("=" * 80)
    print("T-AES-FEEDBACK TRAINING")
    print("=" * 80)

    # Configuration
    config = {
        "model_name": "microsoft/deberta-v3-base",
        "max_length": 512,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "num_epochs": 15,
        "warmup_steps": 500,
        "cefr_weight": 1.0,
        "span_weight": 0.5,
        "error_type_weight": 0.3,
        "patience": 3,
    }

    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # Create data loaders
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

    # Create model
    print("\nCreating model...")
    model = FeedbackModel(
        model_name=config["model_name"],
        num_error_types=5,
        dropout=0.1,
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Loss function
    loss_fn = MultiTaskLoss(
        cefr_weight=config["cefr_weight"],
        span_weight=config["span_weight"],
        error_type_weight=config["error_type_weight"],
    )

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

    # Learning rate scheduler
    total_steps = len(train_loader) * config["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=total_steps,
    )

    print(f"\nTotal training steps: {total_steps}")
    print(f"Warmup steps: {config['warmup_steps']}")

    # Training loop
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    best_dev_loss = float("inf")
    patience_counter = 0

    for epoch in range(config["num_epochs"]):
        print(f"\n{'=' * 80}")
        print(f"EPOCH {epoch + 1}/{config['num_epochs']}")
        print(f"{'=' * 80}")

        # Training
        model.train()
        train_losses = {"total": [], "cefr": [], "span": [], "error_type": []}

        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            targets = {
                "cefr_score": batch["cefr_score"].to(device),
                "span_labels": batch["span_labels"].to(device),
                "error_type_labels": batch["error_type_labels"].to(device),
            }

            # Forward pass
            outputs = model(input_ids, attention_mask)

            # Compute loss
            loss, metrics = loss_fn(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Track losses
            for key in train_losses:
                train_losses[key].append(metrics[key])

            # Log progress
            if (batch_idx + 1) % 50 == 0:
                avg_loss = sum(train_losses["total"][-50:]) / min(
                    50, len(train_losses["total"])
                )
                print(
                    f"  Batch {batch_idx + 1}/{len(train_loader)}: Loss = {avg_loss:.4f}"
                )

        # Epoch training summary
        avg_train_losses = {k: sum(v) / len(v) for k, v in train_losses.items()}
        print("\nTraining metrics:")
        for key, value in avg_train_losses.items():
            print(f"  {key}_loss: {value:.4f}")

        # Validation
        print("\nValidating...")
        model.eval()
        dev_losses = {"total": [], "cefr": [], "span": [], "error_type": []}

        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                targets = {
                    "cefr_score": batch["cefr_score"].to(device),
                    "span_labels": batch["span_labels"].to(device),
                    "error_type_labels": batch["error_type_labels"].to(device),
                }

                outputs = model(input_ids, attention_mask)
                loss, metrics = loss_fn(outputs, targets)

                for key in dev_losses:
                    dev_losses[key].append(metrics[key])

        avg_dev_losses = {k: sum(v) / len(v) for k, v in dev_losses.items()}
        print("Validation metrics:")
        for key, value in avg_dev_losses.items():
            print(f"  {key}_loss: {value:.4f}")

        # Early stopping
        if avg_dev_losses["total"] < best_dev_loss:
            best_dev_loss = avg_dev_losses["total"]
            patience_counter = 0

            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_dev_loss,
                "config": config,
            }
            torch.save(checkpoint, "/training/feedback_model_best.pt")
            print(f"✅ Saved checkpoint (dev_loss: {best_dev_loss:.4f})")
        else:
            patience_counter += 1
            print(f"⚠️  No improvement ({patience_counter}/{config['patience']})")

            if patience_counter >= config["patience"]:
                print("\n❌ Early stopping triggered!")
                break

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best dev loss: {best_dev_loss:.4f}")

    return {
        "best_dev_loss": best_dev_loss,
        "epochs_trained": epoch + 1,
        "config": config,
    }


@app.local_entrypoint()
def main():
    """Run training."""
    result = train_feedback_model.remote()
    print("\n" + "=" * 80)
    print("TRAINING RESULTS:")
    print("=" * 80)
    print(result)
