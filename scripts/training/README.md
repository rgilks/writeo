# Training Overall Score Model

This directory contains scripts for training a custom overall score model using the Write and Improve corpus.

## Overview

The training pipeline trains a **DeBERTa-v3-base** model to predict CEFR levels (A1 to C2) from the Write and Improve corpus, which contains ~4,741 essays with human-annotated CEFR labels.

**Key Features**:

- **Ordinal Regression**: Treats CEFR as ordered categories (not continuous scores) using CORAL loss
- **IELTS-Aligned Scoring**: Corrected CEFR-to-score mapping aligned with IELTS bands
- **QWK Evaluation**: Quadratic Weighted Kappa as primary metric (gold standard for AES)
- **State-of-the-Art Model**: DeBERTa-v3-base (outperforms RoBERTa on most benchmarks)

## Prerequisites

- Write and Improve corpus (2024 v2) located at `~/Desktop/write-and-improve-corpus-2024-v2/`
- Modal account with GPU access
- Python 3.12+

## Quick Start

### 1. Prepare Data

```bash
python scripts/training/prepare-corpus.py
```

This will:

- Load corpus data from the default paths
- Filter for final versions with human CEFR labels
- Convert CEFR levels to band scores (2-9)
- Split into train/dev/test sets
- Save to `scripts/training/data/`

### 2. Test Run (Quick Validation)

Before full training, run a quick test to verify everything works:

```bash
modal run scripts/training/train-overall-score.py --test-run
```

This will:

- Train on a small subset (100 samples)
- Run for a limited number of steps (50)
- Verify the training pipeline works
- Save model to Modal volume

### 3. Full Training

Once test run succeeds, run full training:

```bash
modal run scripts/training/train-overall-score.py --full
```

This will:

- Train on full dataset (~4,000+ training samples)
- Run for multiple epochs with early stopping
- Save best model to Modal volume at `/vol/models/corpus-trained-roberta/`

### 4. Evaluate

Evaluate the trained model:

```bash
python scripts/training/evaluate-model.py \
  --model-path /vol/models/corpus-trained-roberta \
  --data-dir scripts/training/data
```

Or if running on Modal:

```bash
modal run scripts/training/evaluate-model.py \
  --model-path /vol/models/corpus-trained-roberta \
  --data-dir /training/data
```

## Configuration

Edit `scripts/training/config.py` to adjust:

### Model & Architecture

- Base model (DeBERTa-v3 vs RoBERTa)
- Use ordinal regression vs standard regression
- Number of CEFR classes (11)

### Loss Functions

- **CORAL** (recommended): Rank-consistent ordinal regression
- **Soft Labels**: Gaussian soft labels for ordinal data
- **Focal Loss**: Addresses class imbalance
- **CDW-CE**: Class distance weighted cross-entropy
- **MSE**: Baseline regression (for comparison)

### Training Hyperparameters

- Learning rate (2e-5 for DeBERTa-v3)
- Batch size, epochs
- Early stopping patience
- Max sequence length

See [`QUICKSTART.md`](QUICKSTART.md) for detailed configuration guide.

## Model Integration

Once trained, the model is automatically available in the application:

- Model key: `corpus-deberta` (or `corpus-roberta` for legacy)
- Assessor ID: `T-AES-CORPUS`
- Can be used via API: `POST /grade?model_key=corpus-deberta`

**Expected Performance**:

- QWK: 0.75-0.80 (approaching human-level agreement)
- MAE: 0.4-0.5
- Adjacent Accuracy (±1 level): ~92%

## Training on Modal

The training script uses Modal for GPU-accelerated training:

- Default GPU: A10G (can be changed in script)
- Training runs as a separate Modal app: `writeo-training`
- Model is saved to Modal volume: `writeo-models`

## Data Format

Training data is in JSONL format with:

- `input`: Combined prompt and essay text
- `target`: Band score (2.0-9.0)
- `cefr`: Original CEFR label
- `split`: train/dev/test
- `essay_id`: Unique essay identifier

## Troubleshooting

- **Model not found**: Ensure training completed and model was saved to Modal volume
- **Out of memory**: Reduce batch size in config
- **Training too slow**: Use A10 or A100 GPU instead of T4
- **Data not found**: Check corpus paths in `prepare-corpus.py`
