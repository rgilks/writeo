# Training Overall Score Model

This directory contains scripts for training a custom overall score model using the Write and Improve corpus.

## Overview

The training pipeline trains a RoBERTa-base model to predict overall essay scores (band 2-9) from the Write and Improve corpus, which contains ~5,050 essays with human-annotated CEFR labels.

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

- Base model (RoBERTa vs DistilBERT)
- Learning rate, batch size, epochs
- Test run limits
- Model output path

## Model Integration

Once trained, the model is automatically available in the application:

- Model key: `corpus-roberta`
- Assessor ID: `T-AES-CORPUS`
- Can be used via API: `POST /grade?model_key=corpus-roberta`

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
