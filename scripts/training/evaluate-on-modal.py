#!/usr/bin/env python3
"""
Evaluate CEFR model on Modal (where the model is stored).
"""

import json
from pathlib import Path
import sys

import modal

# Add training to path
sys.path.insert(0, "/training")

app = modal.App("writeo-eval-v3")  # Fresh name for new transformers version

# Same image as training
eval_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "transformers>=4.46.0",  # Upgraded for better local_files_only support
        "torch==2.2.0",
        "datasets>=2.14.0",
        "numpy>=1.24.0,<2.0",
        "scikit-learn>=1.3.0",
        "safetensors>=0.4.0",
    )
    .add_local_dir(str(Path(__file__).parent), remote_path="/training")
)

volume = modal.Volume.from_name("writeo-models", create_if_missing=True)


@app.function(
    image=eval_image,
    gpu="A10G",
    timeout=600,
    volumes={"/vol/models": volume},
)
def evaluate_model():
    """Evaluate the trained model."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import numpy as np
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        cohen_kappa_score,
    )

    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    # Load model
    model_path = "/vol/models/corpus-trained-roberta"
    print(f"\nLoading model from {model_path}...")

    # Load tokenizer from roberta-base (same as training, avoids path validation)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # Load trained model - from_pretrained handles safetensors automatically
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ Model loaded successfully on {device}")

    # Load test data
    test_data = []
    with open("/training/data/test.jsonl") as f:
        for line in f:
            test_data.append(json.loads(line))

    print(f"\nEvaluating on {len(test_data)} test samples...")

    # Run predictions
    predictions = []
    true_scores = []
    true_cefr = []

    for item in test_data:
        inputs = tokenizer(
            item["input"],
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            pred_score = outputs.logits.squeeze().item()
            predictions.append(pred_score)
            true_scores.append(item["target"])
            true_cefr.append(item["cefr"])

    predictions = np.array(predictions)
    true_scores = np.array(true_scores)

    # Clip predictions to valid range
    predictions_clipped = np.clip(predictions, 2.0, 8.5)

    # Convert to CEFR for classification metrics
    def score_to_cefr_index(score):
        """Convert score to CEFR class index."""
        score_to_class = {
            2.0: 0,
            2.5: 1,
            3.0: 2,
            3.5: 3,
            4.5: 4,
            5.0: 5,
            5.5: 5,
            6.0: 6,
            6.5: 7,
            7.5: 8,
            8.0: 9,
            8.5: 10,
        }
        # Find closest score
        closest = min(score_to_class.keys(), key=lambda x: abs(x - score))
        return score_to_class[closest]

    pred_classes = np.array([score_to_cefr_index(p) for p in predictions_clipped])
    true_classes = np.array([score_to_cefr_index(t) for t in true_scores])

    # Calculate metrics
    mae = mean_absolute_error(true_scores, predictions_clipped)
    rmse = np.sqrt(mean_squared_error(true_scores, predictions_clipped))

    # QWK - primary metric
    qwk = cohen_kappa_score(true_classes, pred_classes, weights="quadratic")

    # Adjacent accuracy
    adjacent_correct = sum(
        abs(pred_classes[i] - true_classes[i]) <= 1 for i in range(len(pred_classes))
    )
    adjacent_acc = adjacent_correct / len(pred_classes) * 100

    exact_acc = (pred_classes == true_classes).mean() * 100

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    print("\n📊 PRIMARY METRIC (Gold Standard for AES):")
    print(f"   Quadratic Weighted Kappa (QWK): {qwk:.4f}")

    if qwk >= 0.75:
        print("   ✅ EXCELLENT - Approaching human-level agreement!")
    elif qwk >= 0.60:
        print("   ✅ GOOD - Strong performance")
    elif qwk >= 0.40:
        print("   ⚠️  MODERATE - Room for improvement")
    else:
        print("   ❌ NEEDS IMPROVEMENT")

    print("\n📈 REGRESSION METRICS:")
    print(f"   Mean Absolute Error (MAE): {mae:.4f}")
    print(f"   Root Mean Squared Error (RMSE): {rmse:.4f}")

    print("\n🎯 CLASSIFICATION METRICS:")
    print(f"   CEFR Exact Match Accuracy: {exact_acc:.2f}%")
    print(f"   Adjacent Accuracy (±1 level): {adjacent_acc:.2f}%")

    print("\n" + "=" * 80)

    return {
        "qwk": float(qwk),
        "mae": float(mae),
        "rmse": float(rmse),
        "exact_accuracy": float(exact_acc),
        "adjacent_accuracy": float(adjacent_acc),
    }


@app.local_entrypoint()
def main():
    """Run evaluation."""
    result = evaluate_model.remote()
    print(f"\nFinal QWK: {result['qwk']:.4f}")
    print("Target was: 0.65-0.70 (baseline)")

    if result["qwk"] >= 0.70:
        print("✅ TARGET EXCEEDED!")
    elif result["qwk"] >= 0.65:
        print("✅ TARGET MET!")
    else:
        print(f"⚠️  Below target (gap: {0.65 - result['qwk']:.4f})")
