# Modal LanguageTool Service

FastAPI service for grammar checking using LanguageTool.

## Quick Start

```bash
# Install dependencies
uv sync  # or: pip install -e .

# Deploy
modal deploy app.py
```

## Endpoints

- `POST /check` - Check text for grammar errors
- `GET /health` - Health check

## Configuration

- **LanguageTool Version**: 6.4
- **N-grams**: ✅ Enabled (improves precision for confusable words)
- **Storage**: JAR and n-gram data cached in Modal Volumes
- **Cold Start**: ~2-3s (JAR cached), ~10-15min first time (n-gram download)
- **Warm Check**: ~100-500ms

## References

- [docs/DEPLOYMENT.md](../../docs/DEPLOYMENT.md) - Deployment guide
