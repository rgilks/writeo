#!/usr/bin/env python3
"""
Evaluate trained model on test set using Modal.

Run: modal run scripts/training/evaluate-model-modal.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import modal

app = modal.App("writeo-evaluate-model")

# Modal volume for model storage
volume = modal.Volume.from_name("writeo-models", create_if_missing=True)

# Training image with dependencies
training_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "transformers>=4.40.0",
        "torch==2.2.0",
        "numpy>=1.24.0,<2.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "accelerate>=0.26.0",
        "datasets>=2.14.0",
    )
    .add_local_dir("scripts/training", remote_path="/training")
)


def load_jsonl_dataset(file_path: str) -> list[dict]:
    """Load dataset from JSONL file."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def prepare_dataset(data: list[dict], tokenizer: Any, max_length: int = 512):
    """Prepare dataset for evaluation."""
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

    return Dataset.from_dict(encodings), targets


@app.function(
    image=training_image,
    volumes={"/vol": volume},
    gpu="T4",  # Use T4 for evaluation
    timeout=600,  # 10 minutes max
)
def evaluate_model_on_modal():
    """Evaluate model on test set."""
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # Add training directory to path
    sys.path.insert(0, "/training")
    from config import DEFAULT_CONFIG  # type: ignore[import-untyped]

    config = DEFAULT_CONFIG

    model_path = "/vol/models/corpus-trained-roberta"
    test_data_path = "/training/data/test.jsonl"

    print("=" * 80)
    print("EVALUATING MODEL")
    print("=" * 80)
    print(f"Model path: {model_path}")
    print(f"Test data: {test_data_path}")
    print("=" * 80)

    # Load test data
    print("Loading test data...")
    test_data = load_jsonl_dataset(test_data_path)
    print(f"Test samples: {len(test_data)}")

    # Load model and tokenizer
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"Using device: {device}")

    # Prepare dataset
    print("Preparing dataset...")
    test_dataset, true_labels = prepare_dataset(
        test_data, tokenizer, max_length=config.max_seq_length
    )

    # Evaluate
    print("Running evaluation...")
    predictions = []
    true_scores = []

    with torch.no_grad():
        for i in range(len(test_dataset)):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(test_dataset)} samples...")
            item = test_dataset[i]
            input_ids = torch.tensor([item["input_ids"]]).to(device)
            attention_mask = torch.tensor([item["attention_mask"]]).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_score = outputs.logits.squeeze().cpu().item()
            pred_score = np.clip(
                pred_score, config.target_score_min, config.target_score_max
            )

            predictions.append(pred_score)
            true_scores.append(true_labels[i])

    # Calculate metrics
    predictions = np.array(predictions)
    true_scores = np.array(true_scores)

    mae = np.mean(np.abs(predictions - true_scores))
    rmse = np.sqrt(np.mean((predictions - true_scores) ** 2))
    correlation = np.corrcoef(predictions, true_scores)[0, 1]

    # CEFR classification accuracy
    def score_to_cefr(score: float) -> str:
        if score >= 8.5:
            return "C2"
        elif score >= 8.0:
            return "C1+"
        elif score >= 7.0:
            return "C1"
        elif score >= 6.5:
            return "B2+"
        elif score >= 5.5:
            return "B2"
        elif score >= 5.0:
            return "B1+"
        elif score >= 4.5:
            return "B1"
        elif score >= 4.0:
            return "A2+"
        else:
            return "A2"

    # Get true CEFR labels from test data
    true_cefrs = [item["cefr"] for item in test_data]
    pred_cefrs = [score_to_cefr(p) for p in predictions]

    # Calculate accuracy (exact match)
    cefr_accuracy = sum(p == t for p, t in zip(pred_cefrs, true_cefrs)) / len(
        true_cefrs
    )

    # Per-CEFR level analysis
    cefr_stats = {}
    for cefr in set(true_cefrs):
        indices = [i for i, t in enumerate(true_cefrs) if t == cefr]
        if indices:
            cefr_predictions = predictions[indices]
            cefr_true = true_scores[indices]
            cefr_mae = np.mean(np.abs(cefr_predictions - cefr_true))
            cefr_stats[cefr] = {
                "count": len(indices),
                "mae": float(cefr_mae),
                "mean_pred": float(np.mean(cefr_predictions)),
                "mean_true": float(np.mean(cefr_true)),
            }

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Pearson Correlation: {correlation:.4f}")
    print(f"CEFR Classification Accuracy: {cefr_accuracy:.2%}")
    print("\nPer-CEFR Level Analysis:")
    for cefr, stats in sorted(cefr_stats.items()):
        print(f"  {cefr}: {stats['count']} samples, MAE: {stats['mae']:.4f}")
        print(
            f"    Mean predicted: {stats['mean_pred']:.2f}, Mean true: {stats['mean_true']:.2f}"
        )

    print("=" * 80)

    # Save results
    results = {
        "mae": float(mae),
        "rmse": float(rmse),
        "correlation": float(correlation),
        "cefr_accuracy": float(cefr_accuracy),
        "cefr_stats": cefr_stats,
    }

    results_path = f"{model_path}/evaluation_results.json"
    os.makedirs(model_path, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Commit volume
    volume.commit()

    return results


@app.local_entrypoint()
def main():
    """Local entrypoint for evaluation."""
    results = evaluate_model_on_modal.remote()
    print(f"\n✅ Evaluation complete!")
    print(f"MAE: {results['mae']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"Correlation: {results['correlation']:.4f}")
    print(f"CEFR Accuracy: {results['cefr_accuracy']:.2%}")
