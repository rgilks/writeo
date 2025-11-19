# Modal Essay Scoring Service

FastAPI service for essay scoring using ML models. See [MODELS.md](MODELS.md) for model details.

## Quick Start

```bash
# Install dependencies
uv sync  # or: pip install -e .

# Deploy
modal deploy app.py
```

## Endpoints

- `POST /grade` - Score an essay
- `GET /health` - Health check
- `GET /models` - List available models

## Configuration

- **Default Model**: `engessay` (KevSun/Engessay_grading_ML)
- **Model Selection**: Set `MODEL_NAME` env var or use `?model_key=` query param
- **Model Caching**: Weights cached in Modal Volume for faster cold starts

## References

- [MODELS.md](MODELS.md) - Model documentation and comparison
- [docs/DEPLOYMENT.md](../../docs/DEPLOYMENT.md) - Deployment guide
