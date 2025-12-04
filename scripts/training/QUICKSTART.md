# Quick Reference: Training with Ordinal Regression

## Configuration Options

Edit `config.py` to customize your training approach:

### Model Selection

```python
base_model = "microsoft/deberta-v3-base"  # DeBERTa-v3 (recommended)
# Or:  "roberta-base", "microsoft/deberta-v3-small"
```

### Choose Training Approach

**Option 1: Ordinal Regression (RECOMMENDED)**

```python
use_ordinal_regression = True
loss_type = "coral"  # Best for ordinal data
num_classes = 11  # 11 CEFR levels
```

**Option 2: Baseline (MSE Regression)**

```python
use_ordinal_regression = False
```

### Loss Functions for Ordinal Regression

| Loss Type       | When to Use        | Pros                                           | Cons                        |
| --------------- | ------------------ | ---------------------------------------------- | --------------------------- |
| `"coral"`       | **Default choice** | Ensures rank-monotonicity, theoretically sound | Requires special model      |
| `"soft_labels"` | General purpose    | Simple, works well                             | Less theoretically grounded |
| `"focal"`       | Class imbalance    | Focuses on hard examples                       | Can be unstable             |
| `"cdw_ce"`      | Distance-aware     | Penalizes far-off predictions more             | Computationally expensive   |

### Loss Function Parameters

```python
# For soft_labels loss
soft_label_sigma = 1.0  # Lower = sharper distribution (0.5-2.0)

# For focal loss
focal_alpha = 0.25  # Class weighting (0.1-0.5)
focal_gamma = 2.0   # Focusing parameter (1.0-5.0)
```

## Training Commands

### Test Run (Quick Validation)

```bash
cd /Users/robertgilks/Source/writeo
modal run scripts/training/train-overall-score.py --test-run
```

This will:

- Train on 100 samples
- Run for 50 steps
- Verify everything works
- Take ~5-10 minutes

### Full Training

```bash
modal run scripts/training/train-overall-score.py --full
```

This will:

- Train on full dataset (~3,784 samples)
- Run for multiple epochs with early stopping
- Take ~2-4 hours
- Save model to `/vol/models/corpus-trained-deberta/`

## Evaluation

### Local Evaluation

```bash
python scripts/training/evaluate-model.py \
  --model-path /path/to/saved/model \
  --data-dir scripts/training/data
```

### Modal Evaluation

```bash
modal run scripts/training/evaluate-model.py \
  --model-path /vol/models/corpus-trained-deberta \
  --data-dir /training/data
```

## Expected Performance

### Baseline (MSE Regression)

- QWK: ~0.65-0.70
- MAE: ~0.6-0.7
- Adjacent Accuracy: ~85%

### With Ordinal Regression (CORAL)

- QWK: ~0.75-0.80 ✅
- MAE: ~0.4-0.5
- Adjacent Accuracy: ~92%

### Performance Targets

- 🎯 QWK ≥ 0.75 = Excellent
- ✅ QWK 0.60-0.75 = Good
- ⚠️ QWK 0.40-0.60 = Moderate
- ❌ QWK < 0.40 = Needs improvement

## What Changed (Summary)

### 1. CEFR Mapping - CORRECTED ✅

- Added missing `A1+` level
- Reduced all scores by ~1.0-1.5 to align with IELTS
- Example: B1 was 5.0, now 4.5

### 2. DeBERTa-v3 - UPGRADED ✅

- Switched from RoBERTa to DeBERTa-v3-base
- Better performance on most NLP benchmarks
- Same size, similar speed

### 3. Ordinal Regression - NEW ✅

- Treats CEFR as ordered categories (not continuous scores)
- CORAL loss ensures rank-monotonicity
- Soft labels assign probability to neighboring classes

### 4. QWK Metric - ADDED ✅

- Gold standard for AES evaluation
- Penalizes errors based on distance
- Now primary evaluation metric

## Troubleshooting

### "Model not found" during evaluation

- Ensure training completed successfully
- Check model was saved to Modal volume
- Verify path: `/vol/models/corpus-trained-deberta/`

### Out of memory during training

- Reduce `batch_size` from 16 to 8 or 4
- Reduce `max_seq_length` from 512 to 256
- Use `microsoft/deberta-v3-small` instead

### Low QWK score (<0.60)

- Try different loss function (`coral`, `soft_labels`)
- Increase training epochs
- Check for data quality issues

### Training too slow

- Use `--test-run` first to verify
- Consider smaller model (`deberta-v3-small`)
- Check GPU allocation in Modal

## Next Steps

After successful training with ordinal regression:

1. **Compare approaches**: Train both MSE and CORAL, compare QWK
2. **Hyperparameter tuning**: Try different learning rates, batch sizes
3. **Data augmentation**: Add for minority classes (A1+, C2)
4. **Ensemble**: Train 3-5 models, average predictions

## Files Modified

- ✅ `config.py` - Added ordinal regression options
- ✅ `prepare-corpus.py` - Fixed CEFR mapping, added ordinal class conversion
- ✅ `evaluate-model.py` - Added QWK metric, fixed CEFR thresholds
- ✅ `train-overall-score.py` - Integrated ordinal regression
- ✅ `models.py` - NEW: CORAL and soft label models
- ✅ `losses.py` - NEW: Ordinal regression loss functions
