#!/usr/bin/env python3
"""
Train overall score model on Modal.

Supports both test runs (quick validation) and full training runs.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import modal

# Import config (will need to be available in Modal)
# In Modal, files are at /training/, locally they're in the same directory
# Add /training to path for Modal, and current directory for local
sys.path.insert(0, "/training")
sys.path.insert(0, str(Path(__file__).parent))

try:
    from config import DEFAULT_CONFIG  # type: ignore[import-untyped]
except ImportError as e:
    raise ImportError(
        f"Could not import config. Tried paths: {sys.path[:2]}. Error: {e}"
    ) from e

# Modal setup
app = modal.App("writeo-training")

# Training image with dependencies
training_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "transformers>=4.40.0",
        "torch==2.2.0",
        "datasets>=2.14.0",
        "numpy>=1.24.0,<2.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "accelerate>=0.26.0",  # Required for Trainer with PyTorch
    )
    .add_local_dir(str(Path(__file__).parent), remote_path="/training")
)

# Modal volume for model storage
volume = modal.Volume.from_name("writeo-models", create_if_missing=True)


def load_jsonl_dataset(file_path: str) -> list[dict]:
    """Load dataset from JSONL file."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def prepare_dataset(data: list[dict], tokenizer: Any, max_length: int = 512):
    """Prepare dataset for training."""
    from datasets import Dataset

    inputs = [item["input"] for item in data]
    targets = [float(item["target"]) for item in data]

    # Tokenize
    encodings = tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    # Add labels
    encodings["labels"] = targets

    return Dataset.from_dict(encodings)


def create_regression_trainer_class():
    """Create RegressionTrainer class (needs to be inside function to access torch)."""
    import torch
    from transformers import Trainer

    class RegressionTrainer(Trainer):
        """Custom trainer for regression task."""

        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            """Compute MSE loss for regression."""
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits.squeeze()

            # Ensure logits and labels are on same device
            if isinstance(logits, torch.Tensor):
                logits = logits.to(labels.device)

            # Don't clamp during training - let model learn the full range
            # Clamping is only for inference/display
            loss = torch.nn.functional.mse_loss(logits, labels.float())
            return (loss, outputs) if return_outputs else loss

    return RegressionTrainer


@app.function(
    image=training_image,
    volumes={"/vol": volume},
    gpu="A10G",  # Use A10G for training (faster and similar cost to T4)
    timeout=3600 * 4,  # 4 hours max for full training
    # No secrets needed - HuggingFace models are public
)
def train_model(
    test_run: bool = False,
    data_dir: str = "/training/data",
):
    """Train the overall score model."""
    # Import here so they're only needed on Modal
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        TrainingArguments,
    )

    # Load config (use default for now, can be customized later)
    config = DEFAULT_CONFIG

    print("=" * 80)
    print("TRAINING OVERALL SCORE MODEL")
    print("=" * 80)
    print(f"Base model: {config.base_model}")
    print(f"Test run: {test_run}")
    print(f"Data dir: {data_dir}")
    print("=" * 80)

    # Load data
    train_path = Path(data_dir) / config.train_file
    dev_path = Path(data_dir) / config.dev_file

    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError(
            f"Data files not found. Train: {train_path.exists()}, Dev: {dev_path.exists()}"
        )

    print(f"Loading training data from {train_path}...")
    train_data = load_jsonl_dataset(str(train_path))
    print(f"Loading dev data from {dev_path}...")
    dev_data = load_jsonl_dataset(str(dev_path))

    # Limit data for test runs
    if test_run:
        print(f"TEST RUN: Limiting to {config.test_run_max_samples} samples")
        train_data = train_data[: config.test_run_max_samples]
        dev_data = dev_data[: min(config.test_run_max_samples // 5, len(dev_data))]

    print(f"Training samples: {len(train_data)}")
    print(f"Dev samples: {len(dev_data)}")

    # Load tokenizer and model
    print(f"Loading tokenizer and model: {config.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=1,  # Single output for regression
        problem_type="regression",  # Explicitly set as regression
    )

    # Initialize regression head to predict near the mean target value (~5.86)
    # This helps the model start closer to reasonable predictions
    import torch.nn as nn

    if hasattr(model, "classifier") and hasattr(model.classifier, "out_proj"):
        # Initialize bias to predict mean (targets are ~5.86)
        if (
            hasattr(model.classifier.out_proj, "bias")
            and model.classifier.out_proj.bias is not None
        ):
            nn.init.constant_(model.classifier.out_proj.bias, 5.86)
        # Initialize weights to small values
        if hasattr(model.classifier.out_proj, "weight"):
            nn.init.normal_(model.classifier.out_proj.weight, mean=0.0, std=0.02)

    # Prepare datasets
    print("Preparing datasets...")
    train_dataset = prepare_dataset(
        train_data, tokenizer, max_length=config.max_seq_length
    )
    dev_dataset = prepare_dataset(dev_data, tokenizer, max_length=config.max_seq_length)

    # Training arguments
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs if not test_run else 1,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",  # Changed from evaluation_strategy
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        report_to=None,  # Disable wandb/tensorboard for now
        max_steps=config.test_run_max_steps if test_run else -1,
    )

    # Create trainer class
    RegressionTrainer = create_regression_trainer_class()

    # Trainer
    trainer = RegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_threshold=config.early_stopping_threshold,
            )
        ]
        if not test_run
        else [],
    )

    # Train
    print("\nStarting training...")
    train_result = trainer.train()

    # Save final model
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # Evaluate
    print("\nEvaluating on dev set...")
    eval_results = trainer.evaluate()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Final train loss: {train_result.training_loss:.4f}")
    print(f"Final eval loss: {eval_results['eval_loss']:.4f}")
    print(f"Model saved to: {output_dir}")
    print("=" * 80)

    # Commit volume
    volume.commit()

    return {
        "train_loss": train_result.training_loss,
        "eval_loss": eval_results["eval_loss"],
        "model_path": output_dir,
    }


@app.local_entrypoint()
def main(test_run: bool = True):
    """Local entrypoint for training."""
    # For local runs, we'd need to prepare data first
    # This is mainly for Modal deployment
    result = train_model.remote(test_run=test_run)
    print(f"Training result: {result}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", action="store_true", help="Run a quick test")
    parser.add_argument("--full", action="store_true", help="Run full training")
    args = parser.parse_args()

    # Default to full training if no flag specified
    is_test_run = args.test_run and not args.full

    with app.run():
        train_model.remote(test_run=is_test_run)
