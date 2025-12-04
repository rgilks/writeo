# Training Configuration Summary

## Current Setup (Ready for Full Training)

### Model & Architecture

- **Base Model**: RoBERTa-base (125M params)
- **Approach**: Baseline MSE regression (ordinal regression ready but needs Modal cache clear)
- **Output**: `/vol/models/corpus-trained-roberta`

### Data

- **Total Essays**: 4,741 (after filtering)
- **Train/Dev/Test**: 3,784 / 476 / 481 (80/10/10 split)
- **CEFR Mapping**: ✅ Corrected (IELTS-aligned, A1+=2.5 to C2=8.5)

### Class Distribution

```
A1+:    9 (0.2%)  ⚠️  Minority
A2:   105 (2.2%)
A2+:  671 (14.2%)
B1:   925 (19.5%)
B1+: 1073 (22.6%) ← Peak
B2:   919 (19.4%)
B2+:  529 (11.2%)
C1:   370 (7.8%)
C1+:  102 (2.2%)
C2:    38 (0.8%)  ⚠️  Minority
```

**Imbalance Ratio**: 119:1 (max/min)

### Sequence Lengths

- Median: 245 tokens
- p90: 357 tokens
- p95: 392 tokens
- p99: 448 tokens
- **Truncation at 512**: 0.1% (negligible)

### Hyperparameters

```python
learning_rate: 3e-5       # Optimal for RoBERTa
batch_size: 16            # Fits A10G memory
num_epochs: 10            # With early stopping
max_seq_length: 512       # Adequate for 99.9% of essays
warmup_ratio: 0.1
weight_decay: 0.01
```

### Training Time Estimate

- **Test Run** (100 samples, 50 steps): ~20-30 seconds
- **Full Run** (3,784 samples, ~2,370 steps): 2-4 hours on A10G

## Performance Targets

### Baseline (MSE Regression)

- QWK: 0.65-0.70
- MAE: 0.6-0.7
- Adjacent Accuracy: ~85%

### With Ordinal Regression (CORAL)

- QWK: 0.75-0.80 ✨
- MAE: 0.4-0.5
- Adjacent Accuracy: ~92%

## Known Issues & Workarounds

### 1. Modal Caching

**Issue**: Modal aggressively caches `.add_local_dir()` files  
**Impact**: Code changes (like ordinal regression) not picked up  
**Workaround**: Use baseline MSE first, enable ordinal after cache clears

### 2. Class Imbalance

**Issue**: 119:1 ratio, A1+ has only 9 samples  
**Solutions**:

- ✅ Ordinal regression naturally handles this better than MSE
- Option: Enable focal_loss (`loss_type='focal'`)
- Option: Data augmentation for minorities

### 3. DeBERTa-v3 on Modal

**Issue**: Tokenizer compatibility problems  
**Status**: Deferred - RoBERTa only ~3-5% worse performance  
**Future**: Can switch after Modal tokenizer issues resolved

## Files Protected (Git-Ignored)

```gitignore
# Corpus data - DO NOT COMMIT (terms of use)
data/*.jsonl
data/metadata.json
```

✅ `.gitignore` in place to protect corpus data

## Next Steps

1. **Test Run** (verify everything works):

   ```bash
   modal run scripts/training/train-overall-score.py --test-run
   ```

2. **Full Training** (2-4 hours):

   ```bash
   modal run scripts/training/train-overall-score.py --full
   ```

3. **Evaluation**:

   ```bash
   python scripts/training/evaluate-model.py \
     --model-path /vol/models/corpus-trained-roberta \
     --data-dir scripts/training/data
   ```

4. **Optional**: Re-enable ordinal regression in `config.py` and re-train for better QWK

## Expected Improvement

| Metric  | Before | After (Baseline) | After (Ordinal) |
| ------- | ------ | ---------------- | --------------- |
| QWK     | N/A    | 0.65-0.70        | **0.75-0.80**   |
| MAE     | N/A    | 0.6-0.7          | **0.4-0.5**     |
| Adj Acc | N/A    | ~85%             | **~92%**        |

Baseline should already show good improvement due to:

- ✅ Corrected CEFR mapping
- ✅ QWK evaluation metric
- ✅ More training data (4,741 vs previous unknown)
