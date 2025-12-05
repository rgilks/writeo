# T-AES-FEEDBACK Service

Multi-task CEFR scoring and error detection service.

## Model

- **Architecture:** DeBERTa-v3-base (184M parameters)
- **Tasks:**
  1. CEFR scoring (regression)
  2. Error span detection (token classification)
  3. Error type classification (multi-label)
- **Checkpoint:** Epoch 3
- **Performance:**
  - CEFR QWK: 0.835
  - Error Span F1: 0.125 (improving)
  - Error Types F1: 0.061 (learning)

## Deployment

```bash
# Deploy service
modal deploy services/modal-feedback/app.py

# Test locally
modal run services/modal-feedback/app.py
```

## API

**Endpoint:** `/score`

**Request:**

```json
{
  "text": "Your essay text here..."
}
```

**Response:**

```json
{
  "cefr_score": 5.5,
  "cefr_level": "B1+",
  "error_spans": [
    {
      "start": 15,
      "tokens": ["is"]
    }
  ],
  "error_types": {
    "grammar": 0.75,
    "vocabulary": 0.1,
    "mechanics": 0.05,
    "fluency": 0.05,
    "other": 0.05
  }
}
```

## Files

- `app.py` - Modal application setup
- `api.py` - FastAPI routes
- `model_loader.py` - Model checkpoint loading
- `feedback_model.py` - Model architecture

## Notes

- Uses Modal Volume: `writeo-feedback-models`
- Checkpoint: `/checkpoints/feedback_model_best.pt`
- GPU: T4 (cheaper for inference)
- Same architecture as training
