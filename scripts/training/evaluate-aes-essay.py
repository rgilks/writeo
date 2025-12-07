import modal
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
from scipy.stats import pearsonr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define App
app = modal.App("evaluate-aes-essay")

# Image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "scikit-learn",
        "pandas",
        "numpy",
        "scipy",
        "textstat",  # Likely needed for calibration
        "nltk",  # Likely needed for calibration
        "pydantic",
    )
    .add_local_dir(
        "/Users/robertgilks/Source/writeo/services/modal-essay", remote_path="/app"
    )
)

volume = modal.Volume.from_name("writeo-training-data")

MODEL_NAME = "KevSun/Engessay_grading_ML"


def scale_dress_to_ielts(score: float) -> float:
    # 1.0 -> 2.0, 5.0 -> 9.0 (Same as other script)
    return 2.0 + (score - 1.0) * (7.0 / 4.0)


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


@app.function(image=image, volumes={"/data": volume}, gpu="T4", timeout=1800)
@modal.enter()
def setup():
    # Download NLTK data if needed (calibration might use it?)
    import nltk

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


@app.function(image=image, volumes={"/data": volume}, gpu="T4", timeout=1800)
def evaluate_remote():
    sys.path.append("/app")

    # Import scoring logic
    from scoring import process_engessay_scoring

    print(f"📦 Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).cuda().eval()

    print("Inbox DREsS_New...")
    df = pd.read_csv("/data/dress/DREsS_New.tsv", sep="\t")
    df.columns = [c.lower().strip() for c in df.columns]

    print(f"🚀 Evaluating on {len(df)} essays...")

    true_scores = {"ta": [], "cc": [], "vocab": [], "grammar": [], "overall": []}
    pred_scores = {"ta": [], "cc": [], "vocab": [], "grammar": [], "overall": []}

    from tqdm import tqdm

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            essay = str(row["essay"])

            # Ground Truth
            content = float(row["content"])
            org = float(row["organization"])
            lang = float(row["language"])

            true_ta = scale_dress_to_ielts(content)
            true_cc = scale_dress_to_ielts(org)
            true_vocab = scale_dress_to_ielts(lang)
            true_gram = scale_dress_to_ielts(lang)
            true_overall = (true_ta + true_cc + true_vocab + true_gram) / 4.0

            # Inference
            inputs = tokenizer(
                essay, return_tensors="pt", truncation=True, max_length=512
            ).to("cuda")
            with torch.no_grad():
                logits = model(**inputs).logits.cpu().numpy()[0]

            # Use service logic to process logits -> dimensions
            scores = process_engessay_scoring(logits, essay)

            # Collect results
            true_scores["ta"].append(true_ta)
            true_scores["cc"].append(true_cc)
            true_scores["vocab"].append(true_vocab)
            true_scores["grammar"].append(true_gram)
            true_scores["overall"].append(true_overall)

            pred_scores["ta"].append(scores["TA"])
            pred_scores["cc"].append(scores["CC"])
            pred_scores["vocab"].append(scores["Vocab"])
            pred_scores["grammar"].append(scores["Grammar"])
            pred_scores["overall"].append(scores["Overall"])

        except Exception as e:
            print(f"Error on row {idx}: {e}")
            continue

    # Metrics
    print("\n📊 AES-ESSAY (Baseline) Results on DREsS_New:")
    print(f"{'Dimension':<10} {'MAE':<10} {'Pearson':<10}")
    print("-" * 40)

    for dim in ["ta", "cc", "vocab", "grammar", "overall"]:
        true = np.array(true_scores[dim])
        pred = np.array(pred_scores[dim])
        mae = mean_absolute_error(true, pred)
        pearson = pearsonr(true, pred)[0]
        print(f"{dim.upper():<10} {mae:.4f}     {pearson:.4f}")

    # CEFR QWK
    true_ord = to_ordinal(true_scores["overall"])
    pred_ord = to_ordinal(
        pred_scores["overall"]
    )  # Using Overall score for CEFR proxy as model doesn't output distinct CEFR

    qwk = cohen_kappa_score(true_ord, pred_ord, weights="quadratic")
    print("-" * 40)
    print(f"QWK (Overall): {qwk:.4f}")


@app.local_entrypoint()
def main():
    evaluate_remote.remote()
