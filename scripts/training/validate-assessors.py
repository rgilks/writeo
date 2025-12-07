#!/usr/bin/env python3
"""
Validate AES-ESSAY and AES-CORPUS assessors against Write & Improve corpus test set.

Computes standard automated essay scoring (AES) metrics:
- QWK (Quadratic Weighted Kappa) - primary metric for AES
- MAE/RMSE - prediction error metrics
- CEFR accuracy - exact and adjacent (±1 level) accuracy

The test set contains 481 held-out essays from Write & Improve corpus that were
never used in training, ensuring unbiased evaluation.

Examples:
    # Quick validation (10 essays)
    export API_KEY="your-api-key"
    python scripts/training/validate-assessors.py --limit 10

    # Full validation (481 test essays)
    python scripts/training/validate-assessors.py

    # Test only one assessor
    python scripts/training/validate-assessors.py --skip-corpus --limit 10
"""

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import requests
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)

# Load environment variables from .env.local if it exists
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent.parent / ".env.local"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment from {env_path}")
except ImportError:
    print("⚠️  python-dotenv not installed, trying os.getenv() only")
    print("   Install with: pip install python-dotenv")


# CEFR mapping (IELTS-aligned)
CEFR_TO_SCORE = {
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

CEFR_LEVELS = ["A1", "A1+", "A2", "A2+", "B1", "B1+", "B2", "B2+", "C1", "C1+", "C2"]
CEFR_TO_IDX = {cefr: i for i, cefr in enumerate(CEFR_LEVELS)}


def load_test_data(test_file: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Load test essays from JSONL file."""
    data = []
    with open(test_file, "r") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            data.append(json.loads(line.strip()))
    return data


def score_to_cefr(score: float) -> str:
    """Convert numeric score to CEFR level."""
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


def score_via_modal_essay(
    text: str, prompt: str, model: str = "engessay", api_key: str | None = None
) -> dict[str, Any] | None:
    """Score essay via Modal essay service."""
    url = "https://rob-gilks--writeo-essay-fastapi-app.modal.run/grade"

    # Generate simple IDs for validation
    submission_id = str(uuid.uuid4())
    question_id = str(uuid.uuid4())
    answer_id = str(uuid.uuid4())

    # Match the ModalRequest schema exactly
    payload = {
        "submission_id": submission_id,
        "parts": [
            {
                "part": 1,
                "answers": [
                    {
                        "id": answer_id,
                        "question_id": question_id,
                        "question_text": prompt or "Write an essay.",
                        "answer_text": text,
                    }
                ],
            }
        ],
    }

    headers = {}
    if api_key:
        headers["Authorization"] = f"Token {api_key}"

    try:
        response = requests.post(
            url,
            json=payload,
            params={"model_key": model} if model else None,
            headers=headers,
            timeout=90,  # Modal has ~11-13s cold start time
        )
        response.raise_for_status()
        result = response.json()

        # Extract assessor result
        parts = result.get("results", {}).get("parts", [])
        if parts and parts[0].get("answers"):
            answer = parts[0]["answers"][0]
            assessors = answer.get("assessorResults", [])
            if assessors:
                return assessors[0]
        return None
    except Exception as e:
        print(f"Error scoring via modal-essay ({model}): {e}")
        return None


def score_via_modal_corpus(
    text: str, api_key: str | None = None
) -> dict[str, Any] | None:
    """Score essay via Modal corpus service."""
    url = "https://rob-gilks--writeo-corpus-fastapi-app.modal.run/score"

    payload = {"text": text, "max_length": 512}

    headers = {}
    if api_key:
        headers["Authorization"] = f"Token {api_key}"

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()

        # Convert to assessor result format
        return {
            "id": "AES-CORPUS",
            "name": "Corpus RoBERTa",
            "type": "grader",
            "overall": result.get("score"),
            "label": result.get("cefr_level"),
        }
    except Exception as e:
        print(f"Error scoring via modal-corpus: {e}")
        return None


def compute_metrics(
    predictions: np.ndarray, true_scores: np.ndarray, true_cefrs: list[str], name: str
) -> dict[str, Any]:
    """Compute standard AES metrics."""
    # Clip predictions to valid range
    predictions_clipped = np.clip(predictions, 2.0, 8.5)

    # Regression metrics
    mae = mean_absolute_error(true_scores, predictions_clipped)
    rmse = np.sqrt(mean_squared_error(true_scores, predictions_clipped))
    correlation = np.corrcoef(predictions_clipped, true_scores)[0, 1]

    # Convert to CEFR for classification
    pred_cefrs = [score_to_cefr(p) for p in predictions_clipped]

    # CEFR classification metrics
    exact_accuracy = sum(p == t for p, t in zip(pred_cefrs, true_cefrs)) / len(
        true_cefrs
    )

    # QWK - primary metric for AES
    true_indices = [CEFR_TO_IDX.get(c, 4) for c in true_cefrs]
    pred_indices = [CEFR_TO_IDX.get(c, 4) for c in pred_cefrs]
    qwk = cohen_kappa_score(true_indices, pred_indices, weights="quadratic")

    # Adjacent accuracy (±1 level)
    adjacent_accuracy = sum(
        abs(CEFR_TO_IDX.get(p, 4) - CEFR_TO_IDX.get(t, 4)) <= 1
        for p, t in zip(pred_cefrs, true_cefrs)
    ) / len(true_cefrs)

    # Confusion matrix
    conf_matrix = confusion_matrix(
        true_indices, pred_indices, labels=list(range(len(CEFR_LEVELS)))
    )

    # Per-CEFR level stats
    cefr_stats = {}
    for cefr in set(true_cefrs):
        indices = [i for i, t in enumerate(true_cefrs) if t == cefr]
        if indices:
            cefr_preds = predictions_clipped[indices]
            cefr_true = true_scores[indices]
            cefr_mae = np.mean(np.abs(cefr_preds - cefr_true))
            cefr_stats[cefr] = {
                "count": len(indices),
                "mae": float(cefr_mae),
                "mean_pred": float(np.mean(cefr_preds)),
                "mean_true": float(np.mean(cefr_true)),
            }

    print(f"\n{'=' * 80}")
    print(f"{name} RESULTS")
    print(f"{'=' * 80}")
    print("\n📊 PRIMARY METRIC:")
    print(f"   QWK: {qwk:.4f}", end="")
    if qwk >= 0.75:
        print(" ✅ EXCELLENT")
    elif qwk >= 0.60:
        print(" ✅ GOOD")
    elif qwk >= 0.40:
        print(" ⚠️  MODERATE")
    else:
        print(" ❌ NEEDS IMPROVEMENT")

    print("\n📈 REGRESSION METRICS:")
    print(f"   MAE: {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   Correlation: {correlation:.4f}")

    print("\n🎯 CLASSIFICATION METRICS:")
    print(f"   CEFR Exact Accuracy: {exact_accuracy:.2%}")
    print(f"   Adjacent Accuracy (±1): {adjacent_accuracy:.2%}")

    return {
        "name": name,
        "qwk": float(qwk),
        "mae": float(mae),
        "rmse": float(rmse),
        "correlation": float(correlation),
        "exact_accuracy": float(exact_accuracy),
        "adjacent_accuracy": float(adjacent_accuracy),
        "cefr_stats": cefr_stats,
        "confusion_matrix": conf_matrix.tolist(),
        "predictions": predictions_clipped.tolist(),
    }


def generate_markdown_report(
    results_essay: dict, results_corpus: dict, output_file: str
):
    """Generate markdown comparison report."""
    report = f"""# Assessor Validation Report

Comparison of AES-ESSAY and AES-CORPUS assessors against corpus test set.

## Summary

| Metric | AES-ESSAY | AES-CORPUS | Winner |
|--------|-------------|--------------|---------|
| **QWK** (primary) | {results_essay["qwk"]:.4f} | {results_corpus["qwk"]:.4f} | {"🏆 ESSAY" if results_essay["qwk"] > results_corpus["qwk"] else "🏆 CORPUS"} |
| **MAE** | {results_essay["mae"]:.4f} | {results_corpus["mae"]:.4f} | {"🏆 ESSAY" if results_essay["mae"] < results_corpus["mae"] else "🏆 CORPUS"} |
| **RMSE** | {results_essay["rmse"]:.4f} | {results_corpus["rmse"]:.4f} | {"🏆 ESSAY" if results_essay["rmse"] < results_corpus["rmse"] else "🏆 CORPUS"} |
| **Correlation** | {results_essay["correlation"]:.4f} | {results_corpus["correlation"]:.4f} | {"🏆 ESSAY" if results_essay["correlation"] > results_corpus["correlation"] else "🏆 CORPUS"} |
| **Exact Accuracy** | {results_essay["exact_accuracy"]:.2%} | {results_corpus["exact_accuracy"]:.2%} | {"🏆 ESSAY" if results_essay["exact_accuracy"] > results_corpus["exact_accuracy"] else "🏆 CORPUS"} |
| **Adjacent Accuracy** | {results_essay["adjacent_accuracy"]:.2%} | {results_corpus["adjacent_accuracy"]:.2%} | {"🏆 ESSAY" if results_essay["adjacent_accuracy"] > results_corpus["adjacent_accuracy"] else "🏆 CORPUS"} |

## Interpretation

### QWK (Quadratic Weighted Kappa)
The gold standard metric for automated essay scoring. Measures agreement between predicted and true CEFR levels, accounting for ordinal nature and severity of disagreement.

- **≥0.75**: Excellent (approaching human-level agreement)
- **0.60-0.74**: Good (strong agreement)
- **0.40-0.59**: Moderate (acceptable)
- **<0.40**: Needs improvement

### Per-CEFR Level Analysis

#### AES-ESSAY
"""

    for cefr in sorted(results_essay["cefr_stats"].keys()):
        stats = results_essay["cefr_stats"][cefr]
        report += f"\n- **{cefr}**: {stats['count']} samples, MAE: {stats['mae']:.4f}"

    report += "\n\n#### AES-CORPUS\n"

    for cefr in sorted(results_corpus["cefr_stats"].keys()):
        stats = results_corpus["cefr_stats"][cefr]
        report += f"\n- **{cefr}**: {stats['count']} samples, MAE: {stats['mae']:.4f}"

    report += "\n\n## Recommendations\n\n"

    if results_corpus["qwk"] > results_essay["qwk"]:
        report += "✅ **AES-CORPUS shows better performance** - Consider using corpus model as primary assessor.\n"
    else:
        report += "✅ **AES-ESSAY shows better performance** - Current essay model is performing well.\n"

    report += "\n## Next Steps\n\n"
    report += "1. Review per-CEFR level performance to identify weaknesses\n"
    report += "2. Examine confusion matrices to understand error patterns\n"
    report += "3. Consider ensemble approach combining both models\n"
    report += "4. Identify essays where models disagree for manual review\n"

    with open(output_file, "w") as f:
        f.write(report)

    print(f"\n📄 Report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate assessors against corpus test set"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="scripts/training/data/test.jsonl",
        help="Path to test data JSONL file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of essays to score (for quick testing)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="scripts/training",
        help="Output directory for results",
    )
    parser.add_argument(
        "--skip-essay",
        action="store_true",
        help="Skip AES-ESSAY validation",
    )
    parser.add_argument(
        "--skip-corpus",
        action="store_true",
        help="Skip AES-CORPUS validation",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for Modal services (or set API_KEY env var)",
    )

    args = parser.parse_args()

    # Get API key from args or environment
    api_key = args.api_key or os.getenv("API_KEY")
    if not api_key and not args.skip_essay:
        print("⚠️  WARNING: No API key provided for Modal services")
        print("   Set API_KEY environment variable or use --api-key argument")
        print("   Continuing without authentication (may fail)...\n")

    # Load test data
    print(f"Loading test data from {args.test_data}...")
    test_data = load_test_data(args.test_data, limit=args.limit)
    print(f"Loaded {len(test_data)} essays")

    # Extract ground truth
    true_scores = np.array([item["target"] for item in test_data])
    true_cefrs = [item["cefr"] for item in test_data]

    # Score via AES-ESSAY
    essay_predictions = []
    if not args.skip_essay:
        print(f"\n{'=' * 80}")
        print("Scoring via AES-ESSAY (engessay model)...")
        print(f"{'=' * 80}")

        for i, item in enumerate(test_data):
            print(f"Progress: {i + 1}/{len(test_data)}", end="\r")

            # Extract text from input (remove prompt if present)
            text = item["input"]
            # Try to extract just the essay (after "Essay:")
            if "Essay:" in text:
                text = text.split("Essay:", 1)[1].strip()

            result = score_via_modal_essay(text, "", model="engessay", api_key=api_key)
            if result:
                essay_predictions.append(result.get("overall", 4.5))
            else:
                print(
                    f"\nWarning: Failed to score essay {i + 1}, using default score 4.5"
                )
                essay_predictions.append(4.5)

        print()  # New line after progress
        essay_predictions = np.array(essay_predictions)

    # Score via AES-CORPUS
    corpus_predictions = []
    if not args.skip_corpus:
        print(f"\n{'=' * 80}")
        print("Scoring via AES-CORPUS (corpus-roberta model)...")
        print(f"{'=' * 80}")

        for i, item in enumerate(test_data):
            print(f"Progress: {i + 1}/{len(test_data)}", end="\r")

            # Use full input for corpus model
            text = item["input"]

            result = score_via_modal_corpus(text, api_key=api_key)
            if result:
                corpus_predictions.append(result.get("overall", 4.5))
            else:
                print(
                    f"\nWarning: Failed to score essay {i + 1}, using default score 4.5"
                )
                corpus_predictions.append(4.5)

        print()  # New line after progress
        corpus_predictions = np.array(corpus_predictions)

    # Compute metrics
    results = {}

    if not args.skip_essay:
        results["essay"] = compute_metrics(
            essay_predictions, true_scores, true_cefrs, "AES-ESSAY"
        )

    if not args.skip_corpus:
        results["corpus"] = compute_metrics(
            corpus_predictions, true_scores, true_cefrs, "AES-CORPUS"
        )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "validation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {results_file}")

    # Generate markdown report
    if not args.skip_essay and not args.skip_corpus:
        report_file = output_dir / "validation_report.md"
        generate_markdown_report(results["essay"], results["corpus"], str(report_file))

    print(f"\n{'=' * 80}")
    print("VALIDATION COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
