# Services Documentation

Documentation for individual services in the Writeo project.

## API Worker (Cloudflare Workers)

Edge worker that exposes the public API (`/v1/text/submissions`, `/v1/text/.../feedback`, etc.) and orchestrates assessment services.

**Location:** `apps/api-worker/`

### Responsibilities

- Accept submission, feedback, and teacher-feedback requests
- Enforce rate limiting, API-key auth, and payload validation
- Fan out to modal essay/LanguageTool services and merge results
- Serve SSE streaming responses for incremental AI feedback

### Request Validation

- All feedback endpoints now share Zod schemas in `src/routes/feedback/validation.ts`
- Schemas are used by:
  - `handlers-teacher.ts` for teacher feedback requests
  - `handlers-streaming.ts` for AI streaming requests
  - `storage.ts` when merging request-supplied assessment data with stored assessor results
- Adding/changing request fields only requires updating the schema file (and consumers automatically inherit stricter validation errors)

### Local Testing

- `scripts/hooks/pre-push` spins up the worker + web app with mocked LLM responses and runs both Vitest + Playwright suites
- To run separately:
  ```bash
  npm test          # Vitest API suite
  npm run test:e2e  # Playwright suite (requires dev servers or run via pre-push hook)
  ```

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

See [Model Documentation](../models/overview.md) for model documentation and comparison.

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

## Modal GEC Service (Seq2Seq)

FastAPI service for Grammatical Error Correction using Seq2Seq models.

**Location:** `services/modal_gec/`

### Quick Start

```bash
cd services/modal_gec
modal deploy main.py
```

### Endpoints

- `POST /gec_endpoint` - Correct text and return edits
- `GET /health` - Health check

### Configuration

- **Model**: `google/flan-t5-base` (Fine-tuned)
- **Method**: Seq2Seq generation + Diff-based extraction
- **GPU**: A10G
- **Keep-Warm**: 60s
- **Speed**: ~12-16s per request (high quality, slow)

## Modal GECToR Service (Fast)

FastAPI service for fast Grammatical Error Correction using GECToR (Tag, Not Rewrite).

**Location:** `services/modal-gector/`

### Quick Start

```bash
cd services/modal-gector
modal deploy main.py
```

### Endpoints

- `POST /gector_endpoint` - Correct text and return edits
- `GET /health` - Health check

### Configuration

- **Model**: `gotutiyan/gector-roberta-base-5k`
- **Method**: Token-level tagging (encoder-only, all tokens in parallel)
- **GPU**: T4 (cheaper than A10G)
- **Keep-Warm**: 60s
- **Speed**: ~1-2s per request (~10x faster than Seq2Seq)

### Performance Comparison

| Service    | Speed  | Quality | Cost   |
| ---------- | ------ | ------- | ------ |
| Seq2Seq    | 12-16s | High    | Higher |
| **GECToR** | 1-2s   | Good    | Lower  |

Both services run in parallel and results appear as separate assessors (`T-GEC-SEQ2SEQ` and `T-GEC-GECTOR`).

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

- [Deployment Guide](../operations/deployment.md) - Deployment guide
- [System Architecture](../architecture/overview.md) - System architecture
- [Model Documentation](../models/overview.md) - Model details
