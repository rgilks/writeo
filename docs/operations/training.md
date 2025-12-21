# CEFR Training Pipeline

Complete training pipeline for custom CEFR scoring models using the Write & Improve corpus.

## Quick Start

### 1. Prepare Data

```bash
python scripts/training/prepare-corpus.py
```

Processes ~23K essays from Write & Improve corpus → 4,741 filtered essays with CEFR labels.

### 2. Run Training on Modal

```bash
# Test run (5-10 minutes, 100 samples)
modal run scripts/training/train-overall-score.py

# Production training disabled by default - see config.py
```

### 3. Validate Assessors

Compare AES-ESSAY and AES-DEBERTA performance against the corpus test set:

```bash
# Quick test (10 essays)
python scripts/training/validate-assessors.py --limit 10

# Full validation
python scripts/training/validate-assessors.py
```

**Current Performance:**

- **AES-DEBERTA**: Primary scorer with dimensional breakdown
- **AES-ESSAY**: QWK 0.58 (moderate), MAE 0.55, 90% adjacent accuracy

Results are saved to `validation_results.json` and `validation_report.md`.

### 4. Deployed Models

**AES-DEBERTA** (Primary):

- **Service**: `services/modal-deberta/`
- **Endpoint**: `POST /score`
- **Guide**: See [DeBERTa Model Guide](../models/deberta.md) for details.

**AES-ESSAY** (Legacy):

- **URL**: `https://rob-gilks--writeo-essay-fastapi-app.modal.run`
- **Endpoint**: `POST /grade`
- **Performance**: QWK 0.58 (moderate), MAE 0.55, 90% adjacent accuracy (post-calibration)

## Architecture

### Training Pipeline

1. **Data Preparation** (`prepare-corpus.py`)
   - Loads Write & Improve corpus
   - Filters for final versions with CEFR labels
   - Maps CEFR to IELTS-aligned scores (A1/2.0 → C2/8.5)
   - Creates train/dev/test splits (80/10/10)

2. **Model Training** (`train-overall-score.py`)
   - Fine-tunes transformer models (RoBERTa or DeBERTa-v3)
   - Supports MSE regression or ordinal regression (CORAL)
   - Runs on Modal with GPU acceleration (T4/A10G)
   - Saves to Modal volume: `/vol/models/corpus-trained-roberta`

3. **Evaluation** (`evaluate-on-modal.py`)
   - Computes QWK (Quadratic Weighted Kappa)
   - Calculates MAE, RMSE, adjacent accuracy
   - Per-CEFR-level analysis

### Deployment

**Modal Service** (`services/modal-deberta/`)

- FastAPI REST API
- Loads trained model from Modal volume
- GPU-accelerated inference
- Automatic CEFR level conversion

## Configuration

Edit `config.py` for:

| Setting                  | Default        | Options                     |
| ------------------------ | -------------- | --------------------------- |
| `base_model`             | `roberta-base` | `microsoft/deberta-v3-base` |
| `use_ordinal_regression` | `False`        | `True` (CORAL loss)         |
| `learning_rate`          | `3e-5`         | `2e-5` for DeBERTa-v3       |
| `batch_size`             | `16`           | Reduce to 8 if OOM          |
| `max_seq_length`         | `512`          | 99.9% essays fit            |

### Loss Functions

| Loss              | Use Case        | Pros                     | Cons                   |
| ----------------- | --------------- | ------------------------ | ---------------------- |
| **MSE** (default) | Baseline        | Simple, stable           | Ignores ordinality     |
| **CORAL**         | Ordinal data    | Rank-consistent          | Complex implementation |
| **Soft Labels**   | Ordinal data    | Smooth predictions       | Requires tuning        |
| **Focal**         | Class imbalance | Focuses on hard examples | Can be unstable        |

## Data

### CEFR→Score Mapping (IELTS-aligned)

```python
A1 → 2.0    B1  → 4.5    C1  → 7.5
A1+  → 2.5    B1+ → 5.0    C1+ → 8.0
A2  → 3.0    B2  → 6.0    C2  → 8.5
A2+ → 3.5    B2+ → 6.5
```

### Dataset Statistics

- **Total**: 4,741 essays
- **Train**: 3,784 (80%)
- **Dev**: 476 (10%)
- **Test**: 481 (10%)
- **Class imbalance**: 119:1 (B1+ peak, A1+ minority)
- **Sequence lengths**: Median 245 tokens, p99 448 tokens

### Format (JSONL)

```json
{
  "input": "Combined prompt + essay text",
  "target": 5.0,
  "cefr": "B1+",
  "essay_id": "unique-id"
}
```

## Performance

### Current (RoBERTa + MSE)

- **Eval Loss**: 0.43 (excellent)
- **Expected QWK**: 0.65-0.70
- **Expected MAE**: 0.6-0.7

### With Ordinal Regression (Goal)

- **Target QWK**: 0.75-0.80
- **Target MAE**: 0.4-0.5
- **Adjacent Accuracy**: ~92%

## Project Files

### Core Scripts

- `prepare-corpus.py` - Data preparation pipeline
- `train-overall-score.py` - Training script (Modal GPU)
- `evaluate-on-modal.py` - Model evaluation on Modal
- `validate-assessors.py` - **Validate assessors against corpus** (production validation)
- `config.py` - Training configuration
- `analyze-data.py` - Dataset statistics

### Model Architecture

- `models.py` - CORAL & soft label models
- `losses.py` - Ordinal regression losses

### Data Protection

- `.gitignore` - Prevents committing corpus data (terms of use)

## Troubleshooting

**Model not found**

```bash
# Check if training completed
modal volume ls writeo-models
```

**Out of memory**

```python
# config.py
batch_size = 8  # Reduce from 16
max_seq_length = 256  # Reduce from 512
```

**Slow training**

- Use test run first to validate
- Switch to smaller model: `microsoft/deberta-v3-small`
- Increase GPU type in Modal

**Low QWK (\u003c0.60)**

- Enable ordinal regression: `use_ordinal_regression = True`
- Try CORAL loss: `loss_type = "coral"`
- Check CEFR mapping is correct

## Next Steps

After successful baseline training:

1. **Enable Ordinal Regression**: Set `use_ordinal_regression = True`
2. **Data Augmentation**: For minority classes (A1+, C2)
3. **Hyperparameter Tuning**: Grid search learning rate
4. **DeBERTa-v3**: Switch to stronger model (after fixing Modal issues)
5. **Ensemble**: Train multiple models, average predictions

## Integration

The trained model is integrated as a Modal service:

```python
# Service endpoint (example)
POST https://rob-gilks--writeo-deberta-fastapi-app.modal.run/score

# Request
{"text": "Essay text here", "max_length": 512}

# Response
{"score": 3.74, "cefr_level": "A2+", "model": "deberta-v3-large"}
```

See `services/modal-deberta/` for deployment code.

## References

- **Write & Improve Corpus**: Cambridge English
- **CORAL**: Rank Consistent Ordinal Regression (Cao et al., 2020)
- **QWK**: Standard metric for automated essay scoring
- **DeBERTa-v3**: Enhanced BERT with disentangled attention
