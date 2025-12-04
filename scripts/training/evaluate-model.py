#!/usr/bin/env python3
"""
Evaluate trained model on test set.

Can run locally or on Modal.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from config import TrainingConfig, DEFAULT_CONFIG
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from config import TrainingConfig, DEFAULT_CONFIG


def load_jsonl_dataset(file_path: str) -> list[dict]:
    """Load dataset from JSONL file."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def prepare_dataset(data: list[dict], tokenizer: Any, max_length: int = 512) -> Dataset:
    """Prepare dataset for evaluation."""
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


def evaluate_model(
    model_path: str,
    test_data_path: str,
    config: TrainingConfig | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Evaluate model on test set."""
    if config is None:
        config = DEFAULT_CONFIG

    print("=" * 80)
    print("EVALUATING MODEL")
    print("=" * 80)
    print(f"Model path: {model_path}")
    print(f"Test data: {test_data_path}")
    print(f"Device: {device}")
    print("=" * 80)

    # Load test data
    print("Loading test data...")
    test_data = load_jsonl_dataset(test_data_path)
    print(f"Test samples: {len(test_data)}")

    # Load model and tokenizer
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

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

    # CEFR classification accuracy (updated to match corrected IELTS-aligned mapping)
    def score_to_cefr(score: float) -> str:
        """Convert numeric score to CEFR level (IELTS-aligned)"""
        if score >= 8.25:
            return "C2"
        elif score >= 7.75:
            return "C1+"
        elif score >= 7.0:
            return "C1"
        elif score >= 6.25:
            return "B2+"
        elif score >= 5.5:
            return "B2"
        elif score >= 4.75:
            return "B1+"
        elif score >= 4.0:
            return "B1"
        elif score >= 3.25:
            return "A2+"
        elif score >= 2.75:
            return "A2"
        elif score >= 2.25:
            return "A1+"
        else:
            return "A1"

    def cefr_to_score(cefr: str) -> float:
        """Convert CEFR level to numeric score (IELTS-aligned)"""
        mapping = {
            "A1": 2.0,
            "A1+": 2.5,
            "A2": 3.0,
            "A2+": 3.5,
            "B1": 4.5,
            "B1+": 5.0,
            "B2": 6.0,
            "B2+": 6.5,
            "C1": 7.5,
            "C1+": 8.0,
            "C2": 8.5,
        }
        return mapping.get(cefr, 4.5)

    # Get true CEFR labels from test data
    true_cefrs = [item["cefr"] for item in test_data]
    pred_cefrs = [score_to_cefr(p) for p in predictions]

    # Calculate CEFR classification metrics
    cefr_accuracy = sum(p == t for p, t in zip(pred_cefrs, true_cefrs)) / len(
        true_cefrs
    )

    # Quadratic Weighted Kappa (QWK) - gold standard for AES evaluation
    # QWK accounts for ordinal nature and severity of disagreement
    cefr_to_idx = {
        cefr: i
        for i, cefr in enumerate(
            ["A1", "A1+", "A2", "A2+", "B1", "B1+", "B2", "B2+", "C1", "C1+", "C2"]
        )
    }
    true_cefr_indices = [cefr_to_idx.get(c, 4) for c in true_cefrs]  # Default to B1
    pred_cefr_indices = [cefr_to_idx.get(c, 4) for c in pred_cefrs]

    qwk = cohen_kappa_score(true_cefr_indices, pred_cefr_indices, weights="quadratic")

    # Adjacent accuracy - predictions within ±1 CEFR level
    adjacent_accuracy = sum(
        abs(cefr_to_idx.get(p, 4) - cefr_to_idx.get(t, 4)) <= 1
        for p, t in zip(pred_cefrs, true_cefrs)
    ) / len(true_cefrs)

    # Confusion matrix for detailed analysis
    conf_matrix = confusion_matrix(
        true_cefr_indices, pred_cefr_indices, labels=list(range(len(cefr_to_idx)))
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
    print(f"\n📊 PRIMARY METRIC (Gold Standard for AES):")
    print(f"   Quadratic Weighted Kappa (QWK): {qwk:.4f}")
    if qwk >= 0.75:
        print(f"   ✅ EXCELLENT - Approaching human-level agreement")
    elif qwk >= 0.60:
        print(f"   ✅ GOOD - Strong agreement")
    elif qwk >= 0.40:
        print(f"   ⚠️  MODERATE - Acceptable agreement")
    else:
        print(f"   ❌ POOR - Needs improvement")

    print(f"\n📈 REGRESSION METRICS:")
    print(f"   Mean Absolute Error (MAE): {mae:.4f}")
    print(f"   Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"   Pearson Correlation: {correlation:.4f}")

    print(f"\n🎯 CLASSIFICATION METRICS:")
    print(f"   CEFR Exact Match Accuracy: {cefr_accuracy:.2%}")
    print(f"   Adjacent Accuracy (±1 level): {adjacent_accuracy:.2%}")

    print("\n📋 Per-CEFR Level Analysis:")
    for cefr, stats in sorted(cefr_stats.items()):
        print(f"  {cefr}: {stats['count']} samples, MAE: {stats['mae']:.4f}")
        print(
            f"    Mean predicted: {stats['mean_pred']:.2f}, Mean true: {stats['mean_true']:.2f}"
        )

    print("\n" + "=" * 80)

    # Save results
    results = {
        "qwk": float(qwk),  # Primary metric
        "mae": float(mae),
        "rmse": float(rmse),
        "correlation": float(correlation),
        "cefr_accuracy": float(cefr_accuracy),
        "adjacent_accuracy": float(adjacent_accuracy),
        "cefr_stats": cefr_stats,
        "predictions": predictions.tolist(),
        "true_scores": true_scores.tolist(),
        "confusion_matrix": conf_matrix.tolist(),
    }

    results_path = Path(model_path) / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path to test data JSONL file (default: from config)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="scripts/training/data",
        help="Data directory (if test-data not specified)",
    )

    args = parser.parse_args()

    # Determine test data path
    if args.test_data:
        test_data_path = args.test_data
    else:
        test_data_path = Path(args.data_dir) / DEFAULT_CONFIG.test_file

    if not os.path.exists(test_data_path):
        print(f"ERROR: Test data file not found: {test_data_path}")
        sys.exit(1)

    if not os.path.exists(args.model_path):
        print(f"ERROR: Model path not found: {args.model_path}")
        sys.exit(1)

    evaluate_model(args.model_path, str(test_data_path))


if __name__ == "__main__":
    main()
