"""
Evaluate AES-DEBERTA model on held-out test set.

Calculates:
- Quadratic Weighted Kappa (QWK) for CEFR
- Mean Absolute Error (MAE) for dimensional scores
- Error distribution and correlation metrics
"""

import modal
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, accuracy_score
from scipy.stats import pearsonr

# Configuration
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 384
BATCH_SIZE = 16  # Larger batch for inference
MODEL_PATH = "/vol/models/deberta-v3-aes"

# Modal setup
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.6.0",
        "transformers>=4.40.0",
        "datasets>=2.19.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.1.0",
        "numpy>=1.24.0,<2.0",
        "tqdm>=4.66.0",
        "sentencepiece>=0.1.99",
        "safetensors>=0.4.2",
        "scipy>=1.10.0",
    )
    .add_local_dir(str(Path(__file__).parent), remote_path="/training")
)

app = modal.App("writeo-deberta-eval", image=image)
volume = modal.Volume.from_name("writeo-deberta-models")
data_volume = modal.Volume.from_name("writeo-training-data")


def scale_dress_to_ielts(score: float) -> float:
    # 1.0 -> 2.0, 5.0 -> 9.0
    return 2.0 + (score - 1.0) * (7.0 / 4.0)


def score_to_cefr(score: float) -> str:
    if score < 2.5:
        return "A1"
    if score < 3.5:
        return "A2"
    if score < 4.5:
        return "B1"
    if score < 5.5:
        return "B2"
    if score < 7.5:
        return "C1"
    return "C2"


@app.function(
    gpu="A10G",
    volumes={"/vol": volume, "/data": data_volume},
    timeout=1800,  # 30 minutes
)
def evaluate_model():
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer

    # Needs to import model class locally or from mounted file
    # We'll assume model.py is in the python path (mounted /training)
    import sys

    sys.path.append("/training")
    from model import DeBERTaAESModel

    class AESDataset(Dataset):
        def __init__(self, data, tokenizer, max_length):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    print("============================================================")
    print("📊 Evaluating AES-DEBERTA on Held-Out Test Set")
    print("============================================================")

    # Load Model
    print(f"📦 Loading model from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = DeBERTaAESModel(MODEL_NAME)

    # Load weights
    state_dict = torch.load(os.path.join(MODEL_PATH, "pytorch_model.bin"))
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()

    # Load DREsS_New (Held Out)
    dress_path = "/data/dress/DREsS_New.tsv"
    print(f"📥 Loading test data from {dress_path}")

    df = pd.read_csv(dress_path, sep="\t")
    # Columns: id, prompt, essay, content, organization, language, total
    # Normalize
    df.columns = [c.lower().strip() for c in df.columns]

    test_data = []
    skipped = 0

    for idx, row in df.iterrows():
        try:
            content = float(row["content"])
            org = float(row["organization"])
            lang = float(row["language"])

            test_data.append(
                {
                    "text": str(row["essay"]),
                    "prompt": str(row["prompt"]) if pd.notna(row.get("prompt")) else "",
                    "ta": scale_dress_to_ielts(content),
                    "cc": scale_dress_to_ielts(org),
                    "vocab": scale_dress_to_ielts(lang),
                    "grammar": scale_dress_to_ielts(lang),
                    "source": "dress_new",
                }
            )
        except (ValueError, KeyError):
            skipped += 1

    print(f"✅ Loaded {len(test_data)} test essays (skipped {skipped})")

    # Create DataLoader
    # We need to replicate the collate_fn from train script
    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        prompts = [item["prompt"] for item in batch]

        # Combine prompt + text if prompt exists
        inputs = [p + " " + t if p else t for p, t in zip(prompts, texts)]

        encoding = tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        batch_out = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }

        # Add targets
        batch_out["ta"] = torch.tensor(
            [item["ta"] for item in batch], dtype=torch.float
        )
        batch_out["cc"] = torch.tensor(
            [item["cc"] for item in batch], dtype=torch.float
        )
        batch_out["vocab"] = torch.tensor(
            [item["vocab"] for item in batch], dtype=torch.float
        )
        batch_out["grammar"] = torch.tensor(
            [item["grammar"] for item in batch], dtype=torch.float
        )

        # CEFR target (overall score -> ordinal label)
        # Using overall average of dimensions as proxy for true label if unavailable
        overall_scores = [
            (item["ta"] + item["cc"] + item["vocab"] + item["grammar"]) / 4.0
            for item in batch
        ]
        batch_out["cefr"] = torch.tensor(
            overall_scores, dtype=torch.float
        )  # Use float for regression

        return batch_out

    dataset = AESDataset(test_data, tokenizer, MAX_LENGTH)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # Run Inference
    print(f"🚀 Running inference on {len(test_data)} essays...")

    true_scores = {"ta": [], "cc": [], "vocab": [], "grammar": [], "overall": []}
    pred_scores = {
        "ta": [],
        "cc": [],
        "vocab": [],
        "grammar": [],
        "overall": [],
        "cefr_score": [],
    }

    from tqdm import tqdm

    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()

            outputs = model(input_ids, attention_mask)

            # Store predictions
            pred_scores["ta"].extend(outputs["ta"].cpu().numpy())
            pred_scores["cc"].extend(outputs["cc"].cpu().numpy())
            pred_scores["vocab"].extend(outputs["vocab"].cpu().numpy())
            pred_scores["grammar"].extend(outputs["grammar"].cpu().numpy())
            pred_scores["overall"].extend(outputs["overall"].cpu().numpy())
            pred_scores["cefr_score"].extend(outputs["cefr_score"].cpu().numpy())

            # Store truths
            true_scores["ta"].extend(batch["ta"].numpy())
            true_scores["cc"].extend(batch["cc"].numpy())
            true_scores["vocab"].extend(batch["vocab"].numpy())
            true_scores["grammar"].extend(batch["grammar"].numpy())

            # Calculate true overall
            ta = batch["ta"].numpy()
            cc = batch["cc"].numpy()
            vocab = batch["vocab"].numpy()
            gram = batch["grammar"].numpy()
            true_scores["overall"].extend((ta + cc + vocab + gram) / 4.0)

    # Compute Metrics
    metrics = {}
    print("\n📊 Results:")
    print(f"{'Dimension':<10} {'MAE':<10} {'Pearson':<10} {'Within 1.0':<10}")
    print("-" * 50)

    for dim in ["ta", "cc", "vocab", "grammar", "overall"]:
        true = np.array(true_scores[dim])
        pred = np.array(pred_scores[dim])

        mae = mean_absolute_error(true, pred)
        pearson = pearsonr(true, pred)[0]
        within_1 = np.mean(np.abs(true - pred) <= 1.0) * 100

        metrics[dim] = {"mae": mae, "pearson": pearson, "within_1": within_1}
        print(f"{dim.upper():<10} {mae:.4f}     {pearson:.4f}      {within_1:.1f}%")

    # CEFR QWK
    # Map true overall score to CEFR level (A1=0, A2=1, B1=2, B2=3, C1=4, C2=5)
    def to_ordinal(scores):
        res = []
        for s in scores:
            if s < 2.5:
                res.append(0)  # A1
            elif s < 3.5:
                res.append(1)  # A2
            elif s < 4.5:
                res.append(2)  # B1
            elif s < 5.5:
                res.append(3)  # B2
            elif s < 7.5:
                res.append(4)  # C1
            else:
                res.append(5)  # C2
        return np.array(res)

    true_ord = to_ordinal(true_scores["overall"])
    pred_ord = to_ordinal(pred_scores["cefr_score"])

    qwk = cohen_kappa_score(true_ord, pred_ord, weights="quadratic")
    accuracy = accuracy_score(true_ord, pred_ord)

    print("-" * 50)
    print(f"CEFR QWK:      {qwk:.4f}")
    print(f"CEFR Accuracy: {accuracy * 100:.1f}%")

    metrics["cefr"] = {"qwk": qwk, "accuracy": accuracy}

    return metrics


if __name__ == "__main__":
    # Needed for local import simulation
    import sys

    sys.path.append(os.path.dirname(__file__))

    # Needs a mock class if running locally without Modal function context
    class MockDataset:
        def __init__(self, data, *args):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    # Inject into global namespace for pickle
    global AESDataset
    AESDataset = MockDataset

    # This runs on Modal
    pass
