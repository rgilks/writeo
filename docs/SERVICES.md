# Services Documentation

Documentation for individual services in the Writeo project.

## Modal Essay Scoring Service

FastAPI service for essay scoring using ML models.

**Location:** `services/modal-essay/`

### Quick Start

```bash
cd services/modal-essay
uv sync  # or: pip install -e .
modal deploy app.py
```

### Endpoints

- `POST /grade` - Score an essay
- `GET /health` - Health check
- `GET /models` - List available models

### Configuration

- **Default Model**: `engessay` (KevSun/Engessay_grading_ML)
- **Model Selection**: Set `MODEL_NAME` env var or use `?model_key=` query param
- **Model Caching**: Weights cached in Modal Volume for faster cold starts

### Model Details

See `services/modal-essay/MODELS.md` for model documentation and comparison.

## Modal LanguageTool Service

FastAPI service for grammar checking using LanguageTool.

**Location:** `services/modal-lt/`

### Quick Start

```bash
cd services/modal-lt
uv sync  # or: pip install -e .
modal deploy app.py
```

### Endpoints

- `POST /check` - Check text for grammar errors
- `GET /health` - Health check

### Configuration

- **LanguageTool Version**: 6.4
- **N-grams**: ✅ Enabled (improves precision for confusable words)
- **Storage**: JAR and n-gram data cached in Modal Volumes
- **Cold Start**: ~2-3s (JAR cached), ~10-15min first time (n-gram download)
- **Warm Check**: ~100-500ms

## Shared Package

Shared TypeScript types, schemas, and utilities.

**Location:** `packages/shared/`

### TypeScript Package

**Building:**

```bash
cd packages/shared
npm run build
```

**Usage:**

```typescript
import {
  CreateQuestionRequest,
  CreateSubmissionRequest,
  AssessmentResults,
  isValidUUID,
  mapScoreToCEFR,
} from "@writeo/shared";
```

### Python Package

**Installation:**

```bash
cd packages/shared/py
pip install -e .
```

**Usage:**

```python
from schemas import (
    ModalRequest,
    AssessmentResults,
    map_score_to_cefr,
)
```

### Notes

- TypeScript package must be built before other packages can use it
- Python package uses Pydantic v2 for schema validation
- Both packages maintain the same data structures for consistency

## References

- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guide
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
