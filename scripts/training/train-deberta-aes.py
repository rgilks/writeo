"""
Train AES-DEBERTA model on Modal GPU.

3-Stage Training:
1. Dimensional pre-training on IELTS + DREsS for TA, CC, Vocab, Grammar
2. CEFR calibration on W&I corpus
3. End-to-end fine-tuning with combined loss
"""

import modal
from pathlib import Path

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
    )
    .add_local_dir(str(Path(__file__).parent), remote_path="/training")
)

app = modal.App("writeo-deberta-training", image=image)

# Volume for model checkpoints
volume = modal.Volume.from_name("writeo-deberta-models", create_if_missing=True)

# Volume for training data (DREsS)
data_volume = modal.Volume.from_name("writeo-training-data")

# Configuration
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 384  # Reduced from 512 to fit A10G with DeBERTa-v3-large
CHECKPOINT_PATH = "/vol/models/deberta-v3-aes"


def load_ielts_dataset():
    """Load IELTS-WT2-LLaMa3-1k from HuggingFace."""
    from datasets import load_dataset

    print("📥 Loading IELTS-WT2 dataset from HuggingFace...")
    ds = load_dataset("123Harr/IELTS-WT2-LLaMa3-1k")

    # Normalize column names and extract scores
    data = []
    for row in ds["train"]:
        data.append(
            {
                "text": row.get("essay", row.get("text", "")),
                "prompt": row.get("prompt", ""),
                "ta": float(
                    row.get("task_achievement_score", row.get("Task Achievement", 0))
                ),
                "cc": float(
                    row.get("coherence_score", row.get("Coherence and Cohesion", 0))
                ),
                "vocab": float(
                    row.get("lexical_resource_score", row.get("Lexical Resource", 0))
                ),
                "grammar": float(
                    row.get(
                        "grammar_score", row.get("Grammatical Range and Accuracy", 0)
                    )
                ),
                "source": "ielts",
            }
        )

    print(f"✅ Loaded {len(data)} IELTS essays")
    return data


def load_dress_dataset(dress_path: str):
    """Load DREsS_Std dataset from TSV file."""
    import pandas as pd

    print(f"📥 Loading DREsS dataset from {dress_path}...")

    # DREsS_Std has header row: id, source, prompt, essay, content, organization, language, total
    # Try reading with header first, fallback to no header if needed
    try:
        df = pd.read_csv(dress_path, sep="\t")
        print(f"   Loaded with header. Columns: {list(df.columns)}")
        # Normalize column names to lowercase
        df.columns = [c.lower().strip() for c in df.columns]
    except Exception as e:
        print(f"   Header read failed, trying without: {e}")
        df = pd.read_csv(
            dress_path,
            sep="\t",
            header=None,
            names=[
                "id",
                "source",
                "prompt",
                "essay",
                "content",
                "organization",
                "language",
                "total",
            ],
        )

    # Map DREsS scores (1-5) to IELTS scale (0-9)
    # Linear mapping: 1 → 2, 5 → 9 => score = 2 + (dress_score - 1) * 7/4
    def scale_dress_to_ielts(score: float) -> float:
        return 2.0 + (score - 1.0) * (7.0 / 4.0)

    data = []
    skipped = 0
    for idx, row in df.iterrows():
        try:
            content = float(row["content"])
            org = float(row["organization"])
            lang = float(row["language"])

            data.append(
                {
                    "text": str(row["essay"]),
                    "prompt": str(row["prompt"]) if pd.notna(row.get("prompt")) else "",
                    "ta": scale_dress_to_ielts(content),  # Content → TA
                    "cc": scale_dress_to_ielts(org),  # Organization → CC
                    "vocab": scale_dress_to_ielts(lang),
                    "grammar": scale_dress_to_ielts(lang),
                    "source": "dress",
                }
            )
        except (ValueError, KeyError) as e:
            skipped += 1
            if skipped <= 5:
                print(f"   Skipping row {idx}: {e}")

    if skipped > 5:
        print(f"   ... and {skipped - 5} more skipped rows")

    print(f"✅ Loaded {len(data)} DREsS essays")
    return data


def load_wi_corpus(wi_path: str):
    """Load Write & Improve corpus for CEFR calibration."""
    import json

    print(f"📥 Loading W&I corpus from {wi_path}...")

    data = []
    with open(wi_path, "r") as f:
        for line in f:
            row = json.loads(line)
            data.append(
                {
                    "text": row.get("text", row.get("essay", "")),
                    "prompt": row.get("prompt", ""),
                    "cefr": float(row.get("target", row.get("cefr_score", 0))),
                    "source": "wi",
                }
            )

    print(f"✅ Loaded {len(data)} W&I essays")
    return data


class AESDataset:
    """Dataset for essay scoring with multi-task targets."""

    def __init__(self, data: list, tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        import torch

        item = self.data[idx]

        # Tokenize
        text = item.get("prompt", "") + "\n\n" + item["text"]
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        result = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

        # Add dimension targets (NaN if not available)
        for dim in ["ta", "cc", "vocab", "grammar", "cefr"]:
            if dim in item:
                result[dim] = torch.tensor(item[dim], dtype=torch.float32)
            else:
                result[dim] = torch.tensor(float("nan"), dtype=torch.float32)

        return result


def collate_fn(batch):
    """Collate function for DataLoader."""
    import torch

    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "ta": torch.stack([x["ta"] for x in batch]),
        "cc": torch.stack([x["cc"] for x in batch]),
        "vocab": torch.stack([x["vocab"] for x in batch]),
        "grammar": torch.stack([x["grammar"] for x in batch]),
        "cefr": torch.stack([x["cefr"] for x in batch]),
    }


@app.function(
    gpu="A10G",
    timeout=14400,  # 4 hours
    volumes={"/vol": volume, "/data": data_volume},
)
def train_deberta_aes(
    debug: bool = False,
    stage1_epochs: int = 5,
    stage2_epochs: int = 3,
    stage3_epochs: int = 2,
):
    """Train multi-task DeBERTa-v3 AES model."""
    import os
    import sys
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from tqdm import tqdm

    # Add training directory to path
    sys.path.insert(0, "/training")

    # Import model (we need to copy model.py to training dir)
    # For now, define inline or import from mounted volume

    print("=" * 60)
    print("🚀 Starting AES-DEBERTA Training")
    print("=" * 60)

    # Configuration - reduced batch size for A10G with DeBERTa-v3-large
    batch_size = 1  # Smallest batch for memory
    gradient_accumulation = 32  # Large accumulation to maintain effective batch size
    learning_rate = 1e-5

    if debug:
        stage1_epochs = 1
        stage2_epochs = 1
        stage3_epochs = 1

    # Load tokenizer
    print(f"\n📦 Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load datasets
    print("\n📊 Loading datasets...")

    # IELTS dataset from HuggingFace
    ielts_data = load_ielts_dataset()

    # DREsS_Std dataset (from Modal volume)
    dress_path = "/data/dress/DREsS_Std.tsv"
    dress_data = []
    if os.path.exists(dress_path):
        dress_data = load_dress_dataset(dress_path)
    else:
        print("⚠️ DREsS_Std dataset not found, using IELTS only for Stage 1")

    # DREsS_New dataset (held out entirely for testing)
    dress_new_path = "/data/dress/DREsS_New.tsv"
    dress_new_data = []
    if os.path.exists(dress_new_path):
        dress_new_data = load_dress_dataset(dress_new_path)
        print(f"📌 DREsS_New held out for testing: {len(dress_new_data)} essays")

    # W&I corpus for CEFR
    wi_path = "/training/data/train.jsonl"
    wi_data = []
    if os.path.exists(wi_path):
        wi_data = load_wi_corpus(wi_path)
    else:
        print("⚠️ W&I corpus not found, skipping Stage 2")

    # Debug mode: limit data
    if debug:
        ielts_data = ielts_data[:50]
        dress_data = dress_data[:100] if dress_data else []
        dress_new_data = dress_new_data[:50] if dress_new_data else []
        wi_data = wi_data[:50] if wi_data else []

    # ========================================
    # TRAIN/VAL/TEST SPLITS (80/10/10)
    # ========================================
    from sklearn.model_selection import train_test_split

    print("\n📊 Splitting datasets (80/10/10)...")

    # IELTS splits
    if len(ielts_data) > 10:
        ielts_train, ielts_temp = train_test_split(
            ielts_data, test_size=0.2, random_state=42
        )
        ielts_val, ielts_test = train_test_split(
            ielts_temp, test_size=0.5, random_state=42
        )
    else:
        ielts_train, ielts_val, ielts_test = ielts_data, [], []

    # DREsS splits
    if len(dress_data) > 10:
        dress_train, dress_temp = train_test_split(
            dress_data, test_size=0.2, random_state=42
        )
        dress_val, dress_test = train_test_split(
            dress_temp, test_size=0.5, random_state=42
        )
    else:
        dress_train, dress_val, dress_test = dress_data, [], []

    # W&I splits
    if len(wi_data) > 10:
        wi_train, wi_temp = train_test_split(wi_data, test_size=0.2, random_state=42)
        wi_val, wi_test = train_test_split(wi_temp, test_size=0.5, random_state=42)
    else:
        wi_train, wi_val, wi_test = wi_data, [], []

    # Print split sizes
    print(
        f"   IELTS:  train={len(ielts_train)}, val={len(ielts_val)}, test={len(ielts_test)}"
    )
    print(
        f"   DREsS:  train={len(dress_train)}, val={len(dress_val)}, test={len(dress_test)}"
    )
    print(f"   DREsS_New (held out): test={len(dress_new_data)}")
    print(f"   W&I:    train={len(wi_train)}, val={len(wi_val)}, test={len(wi_test)}")

    # Combine datasets for each stage
    stage1_train = ielts_train + dress_train
    stage1_val = ielts_val + dress_val
    stage1_test = ielts_test + dress_test + dress_new_data  # Include DREsS_New in test

    print(
        f"\n   Stage 1 totals: train={len(stage1_train)}, val={len(stage1_val)}, test={len(stage1_test)}"
    )

    # Create data loaders
    stage1_train_dataset = AESDataset(stage1_train, tokenizer, MAX_LENGTH)
    stage1_val_dataset = AESDataset(stage1_val, tokenizer, MAX_LENGTH)

    stage1_train_loader = DataLoader(
        stage1_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    stage1_val_loader = DataLoader(
        stage1_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Initialize model
    print(f"\n🧠 Initializing model: {MODEL_NAME}")

    # Import model components
    from model import DeBERTaAESModel, MultiTaskLoss

    model = DeBERTaAESModel(model_name=MODEL_NAME)
    model = model.cuda()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )

    # Loss function
    loss_fn = MultiTaskLoss(dimension_weight=1.0, cefr_weight=0.5)

    # ==== STAGE 1: Dimensional Training ====
    print(f"\n{'=' * 60}")
    print(f"📈 STAGE 1: Dimensional Training ({stage1_epochs} epochs)")
    print(f"   Train: {len(stage1_train)} essays, Val: {len(stage1_val)} essays")
    print(f"{'=' * 60}")

    total_steps = len(stage1_train_loader) * stage1_epochs // gradient_accumulation
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    # Early stopping variables
    best_val_loss = float("inf")
    patience = 2
    patience_counter = 0
    best_model_state = None

    for epoch in range(stage1_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        progress = tqdm(
            stage1_train_loader, desc=f"Epoch {epoch + 1}/{stage1_epochs} [Train]"
        )

        optimizer.zero_grad()
        for step, batch in enumerate(progress):
            # Move to GPU
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            targets = {
                k: v.cuda()
                for k, v in batch.items()
                if k not in ["input_ids", "attention_mask"]
            }

            # Forward
            outputs = model(input_ids, attention_mask)
            loss, losses = loss_fn(outputs, targets)
            loss = loss / gradient_accumulation

            # Backward
            loss.backward()

            if (step + 1) % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * gradient_accumulation
            progress.set_postfix({"loss": f"{loss.item() * gradient_accumulation:.4f}"})

        avg_train_loss = epoch_loss / len(stage1_train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = {"ta": 0.0, "cc": 0.0, "vocab": 0.0, "grammar": 0.0}
        val_count = 0

        with torch.no_grad():
            for batch in tqdm(
                stage1_val_loader, desc=f"Epoch {epoch + 1}/{stage1_epochs} [Val]"
            ):
                input_ids = batch["input_ids"].cuda()
                attention_mask = batch["attention_mask"].cuda()
                targets = {
                    k: v.cuda()
                    for k, v in batch.items()
                    if k not in ["input_ids", "attention_mask"]
                }

                outputs = model(input_ids, attention_mask)
                loss, _ = loss_fn(outputs, targets)
                val_loss += loss.item()

                # Calculate MAE for each dimension
                for dim in ["ta", "cc", "vocab", "grammar"]:
                    if dim in outputs and dim in targets:
                        mask = ~torch.isnan(targets[dim])
                        if mask.sum() > 0:
                            mae = (
                                torch.abs(outputs[dim][mask] - targets[dim][mask])
                                .mean()
                                .item()
                            )
                            val_mae[dim] += mae * mask.sum().item()
                            val_count += mask.sum().item()

        avg_val_loss = val_loss / max(len(stage1_val_loader), 1)
        avg_val_mae = {k: v / max(val_count / 4, 1) for k, v in val_mae.items()}

        print(
            f"   Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}"
        )
        print(
            f"            MAE: TA={avg_val_mae['ta']:.3f}, CC={avg_val_mae['cc']:.3f}, "
            f"Vocab={avg_val_mae['vocab']:.3f}, Grammar={avg_val_mae['grammar']:.3f}"
        )

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            print(f"   ✅ New best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"   ⚠️ No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"   🛑 Early stopping triggered at epoch {epoch + 1}")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict({k: v.cuda() for k, v in best_model_state.items()})
        print("   ✅ Restored best model from validation")

    # ==== STAGE 2: CEFR Calibration ====
    if len(wi_train) > 0 and stage2_epochs > 0:
        print(f"\n{'=' * 60}")
        print(f"📈 STAGE 2: CEFR Calibration ({stage2_epochs} epochs)")
        print(f"   Train: {len(wi_train)} essays (W&I)")
        print(f"{'=' * 60}")

        stage2_train_dataset = AESDataset(wi_train, tokenizer, MAX_LENGTH)
        stage2_train_loader = DataLoader(
            stage2_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        # Freeze dimension heads, train CEFR head
        for param in model.ta_head.parameters():
            param.requires_grad = False
        for param in model.cc_head.parameters():
            param.requires_grad = False
        for param in model.vocab_head.parameters():
            param.requires_grad = False
        for param in model.grammar_head.parameters():
            param.requires_grad = False

        cefr_loss_fn = MultiTaskLoss(dimension_weight=0.0, cefr_weight=1.0)

        for epoch in range(stage2_epochs):
            epoch_loss = 0.0
            progress = tqdm(
                stage2_train_loader, desc=f"CEFR Epoch {epoch + 1}/{stage2_epochs}"
            )

            optimizer.zero_grad()
            for step, batch in enumerate(progress):
                input_ids = batch["input_ids"].cuda()
                attention_mask = batch["attention_mask"].cuda()
                targets = {
                    k: v.cuda()
                    for k, v in batch.items()
                    if k not in ["input_ids", "attention_mask"]
                }

                outputs = model(input_ids, attention_mask)
                loss, _ = cefr_loss_fn(outputs, targets)
                loss = loss / gradient_accumulation
                loss.backward()

                if (step + 1) % gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * gradient_accumulation
                progress.set_postfix(
                    {"loss": f"{loss.item() * gradient_accumulation:.4f}"}
                )

            print(
                f"   Epoch {epoch + 1} CEFR loss: {epoch_loss / len(stage2_train_loader):.4f}"
            )

        # Unfreeze all
        for param in model.parameters():
            param.requires_grad = True

    # ==== STAGE 3: End-to-end Fine-tuning ====
    if stage3_epochs > 0:
        print(f"\n{'=' * 60}")
        print(f"📈 STAGE 3: End-to-end Fine-tuning ({stage3_epochs} epochs)")
        print(f"{'=' * 60}")

        # Combine all TRAINING data
        all_train_data = stage1_train + wi_train
        all_train_dataset = AESDataset(all_train_data, tokenizer, MAX_LENGTH)
        all_train_loader = DataLoader(
            all_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        combined_loss_fn = MultiTaskLoss(dimension_weight=0.6, cefr_weight=0.4)

        for epoch in range(stage3_epochs):
            epoch_loss = 0.0
            progress = tqdm(
                all_train_loader, desc=f"Fine-tune Epoch {epoch + 1}/{stage3_epochs}"
            )

            optimizer.zero_grad()
            for step, batch in enumerate(progress):
                input_ids = batch["input_ids"].cuda()
                attention_mask = batch["attention_mask"].cuda()
                targets = {
                    k: v.cuda()
                    for k, v in batch.items()
                    if k not in ["input_ids", "attention_mask"]
                }

                outputs = model(input_ids, attention_mask)
                loss, _ = combined_loss_fn(outputs, targets)
                loss = loss / gradient_accumulation
                loss.backward()

                if (step + 1) % gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * gradient_accumulation
                progress.set_postfix(
                    {"loss": f"{loss.item() * gradient_accumulation:.4f}"}
                )

            print(
                f"   Epoch {epoch + 1} combined loss: {epoch_loss / len(all_train_loader):.4f}"
            )

    # Save model
    print(f"\n💾 Saving model to {CHECKPOINT_PATH}")
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
    tokenizer.save_pretrained(CHECKPOINT_PATH)

    # Commit volume
    volume.commit()

    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print(f"   Model saved to: {CHECKPOINT_PATH}")
    print("=" * 60)

    return {"status": "success", "checkpoint": CHECKPOINT_PATH}


@app.local_entrypoint()
def main(debug: bool = False):
    """Run training."""
    result = train_deberta_aes.remote(debug=debug)
    print(f"Training result: {result}")


if __name__ == "__main__":
    main()
