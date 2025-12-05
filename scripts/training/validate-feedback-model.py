"""
Validate T-AES-FEEDBACK model on Modal.

Loads trained checkpoint and evaluates on test set with full metrics.
"""

import modal
from pathlib import Path

# Use same image as training
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.0",
        "transformers==4.36.0",
        "accelerate==0.25.0",
        "scikit-learn==1.3.2",
        "numpy==1.26.2",
        "sentencepiece>=0.1.99",
    )
    .add_local_dir(str(Path(__file__).parent), remote_path="/training")
)

app = modal.App("writeo-feedback-validation", image=image)

# Same volume as training
volume = modal.Volume.from_name("writeo-feedback-models", create_if_missing=True)


@app.function(
    gpu="T4",  # Smaller GPU for inference
    timeout=1800,  # 30 minutes
    volumes={"/checkpoints": volume},  # Mount volume
)
def validate_feedback_model():
    """Validate trained feedback model on test set."""
    import sys

    sys.path.append("/training")

    import torch
    import numpy as np
    from transformers import AutoTokenizer
    from sklearn.metrics import (
        cohen_kappa_score,
        mean_absolute_error,
        f1_score,
        precision_score,
        recall_score,
    )
    from pathlib import Path

    from feedback_model import FeedbackModel
    from feedback_dataset import FeedbackDataset

    print("=" * 80)
    print("T-AES-FEEDBACK VALIDATION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load checkpoint
    checkpoint_path = "/checkpoints/feedback_model_best.pt"
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Dev loss: {checkpoint['loss']:.4f}")

    # Create model
    print("\nCreating model...")
    model = FeedbackModel(
        model_name=checkpoint["config"]["model_name"],
        num_error_types=5,
        dropout=0.1,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load tokenizer and test data
    print("\nLoading tokenizer and test data...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint["config"]["model_name"])

    test_dataset = FeedbackDataset(
        data_file=Path(
            "/training/data-enhanced/dev-enhanced.jsonl"
        ),  # Using dev as test
        tokenizer=tokenizer,
        max_length=512,
    )

    print(f"Test samples: {len(test_dataset)}")

    # Run inference
    print("\nRunning inference...")
    predictions = {
        "cefr_scores": [],
        "true_cefr_scores": [],
        "span_predictions": [],
        "span_labels": [],
        "error_type_predictions": [],
        "error_type_labels": [],
    }

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(test_dataset)}")

            sample = test_dataset[idx]

            # Move to device
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)

            # Predict
            outputs = model(input_ids, attention_mask)

            # CEFR scores
            predictions["cefr_scores"].append(outputs["cefr_score"].item())
            predictions["true_cefr_scores"].append(sample["cefr_score"].item())

            # Span predictions (argmax of logits)
            span_preds = torch.argmax(outputs["span_logits"], dim=-1).cpu().numpy()[0]
            predictions["span_predictions"].extend(span_preds)
            predictions["span_labels"].extend(sample["span_labels"].numpy())

            # Error type predictions (sigmoid > 0.5)
            error_type_preds = (
                (torch.sigmoid(outputs["error_type_logits"]) > 0.5).cpu().numpy()[0]
            )
            predictions["error_type_predictions"].append(error_type_preds)
            predictions["error_type_labels"].append(sample["error_type_labels"].numpy())

    print("\n" + "=" * 80)
    print("VALIDATION METRICS")
    print("=" * 80)

    # CEFR metrics
    cefr_pred = np.array(predictions["cefr_scores"])
    cefr_true = np.array(predictions["true_cefr_scores"])

    # Round to nearest 0.5 for QWK (ordinal) and convert to integers
    cefr_pred_rounded = (np.round(cefr_pred * 2) / 2 * 2).astype(
        int
    )  # Scale to integers
    cefr_true_rounded = (np.round(cefr_true * 2) / 2 * 2).astype(int)

    qwk = cohen_kappa_score(cefr_true_rounded, cefr_pred_rounded, weights="quadratic")
    mae = mean_absolute_error(cefr_true, cefr_pred)

    print("\nCEFR Performance:")
    print(f"  QWK: {qwk:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Mean prediction: {cefr_pred.mean():.2f}")
    print(f"  Mean true: {cefr_true.mean():.2f}")

    # Adjacent accuracy (within 0.5)
    adjacent = np.abs(cefr_pred - cefr_true) <= 0.5
    adjacent_acc = adjacent.mean()
    print(f"  Adjacent accuracy: {adjacent_acc:.1%}")

    # Error span detection metrics
    span_pred = np.array(predictions["span_predictions"])
    span_true = np.array(predictions["span_labels"])

    # Filter out padding (-100)
    mask = span_true != -100
    span_pred_filtered = span_pred[mask]
    span_true_filtered = span_true[mask]

    # Binary: error (1,2) vs no-error (0)
    span_pred_binary = (span_pred_filtered > 0).astype(int)
    span_true_binary = (span_true_filtered > 0).astype(int)

    span_f1 = f1_score(span_true_binary, span_pred_binary, average="binary")
    span_precision = precision_score(
        span_true_binary, span_pred_binary, average="binary"
    )
    span_recall = recall_score(span_true_binary, span_pred_binary, average="binary")

    print("\nError Span Detection:")
    print(f"  F1: {span_f1:.4f}")
    print(f"  Precision: {span_precision:.4f}")
    print(f"  Recall: {span_recall:.4f}")

    # Error type classification
    error_type_pred = np.array(predictions["error_type_predictions"])
    error_type_true = np.array(predictions["error_type_labels"])

    error_categories = ["grammar", "vocabulary", "mechanics", "fluency", "other"]

    print("\nError Type Classification:")
    for idx, category in enumerate(error_categories):
        cat_f1 = f1_score(
            error_type_true[:, idx], error_type_pred[:, idx], average="binary"
        )
        print(f"  {category}: F1 = {cat_f1:.4f}")

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

    return {
        "cefr": {
            "qwk": float(qwk),
            "mae": float(mae),
            "adjacent_accuracy": float(adjacent_acc),
        },
        "span_detection": {
            "f1": float(span_f1),
            "precision": float(span_precision),
            "recall": float(span_recall),
        },
        "error_types": {
            cat: float(
                f1_score(
                    error_type_true[:, idx], error_type_pred[:, idx], average="binary"
                )
            )
            for idx, cat in enumerate(error_categories)
        },
    }


@app.local_entrypoint()
def main():
    """Run validation."""
    print("Starting validation on Modal...")
    result = validate_feedback_model.remote()

    print("\n" + "=" * 80)
    print("FINAL RESULTS:")
    print("=" * 80)
    import json

    print(json.dumps(result, indent=2))
