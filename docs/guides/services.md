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

---

## Modal Essay Services (Scoring)

### 1. Corpus Scoring Service (`modal-corpus`)

**Location:** `services/modal-corpus/`

Primary scorer trained on the Write & Improve corpus. Provides the most accurate CEFR alignment (0.96 correlation).

- **Model:** `roberta-base` fine-tuned on W&I data
- **Output:** Overall Score + CEFR Level
- **Performance:** Very fast (~200ms warm)

**Quick Start:**

```bash
cd services/modal-corpus && modal deploy main.py
```

### 2. DeBERTa Scoring Service (`modal-deberta`)

**Location:** `services/modal-deberta/`

Advanced scorer providing multi-dimensional breakdown.

- **Model:** `microsoft/deberta-v3-large`
- **Output:** Dimensions: Task Achievement, Coherence & Cohesion, Vocabulary, Grammar
- **Performance:** Slower (~400-800ms warm), requires GPU

**Quick Start:**

```bash
cd services/modal-deberta && modal deploy main.py
```

### 3. Legacy Essay Service (`modal-essay`)

**Location:** `services/modal-essay/`

Original baseline scorer.

- **Model:** `engessay` (KevSun/Engessay_grading_ML)
- **Status:** Legacy / Fallback

---

## Modal Grammar Services (GEC)

### 1. GEC Seq2Seq (`modal-gec`)

**Location:** `services/modal-gec/`

High-precision correction using Sequence-to-Sequence generation.

- **Model:** `google/flan-t5-base` (Fine-tuned)
- **Method:** Generates corrected text, then calculates diffs
- **Pros:** Highest correction quality, handles complex rewrites
- **Cons:** Slow (~12-16s), expensive

**Quick Start:**

```bash
cd services/modal-gec && modal deploy main.py
```

### 2. GECToR Fast (`modal-gector`)

**Location:** `services/modal-gector/`

High-speed correction using iterative tagging.

- **Model:** `gotutiyan/gector-roberta-base-5k`
- **Method:** Token classification (Keep, Delete, Replace, etc.)
- **Pros:** Fast (~1-2s), cheaper
- **Cons:** Less capable of structural rewrites

**Quick Start:**

```bash
cd services/modal-gector && modal deploy main.py
```

### 3. LanguageTool (`modal-lt`)

**Location:** `services/modal-lt/`

Rule-based grammar and mechanics checking.

- **Engine:** LanguageTool 6.4 (Java)
- **Pros:** CPU-only (cheap), explains errors well, catches mechanical issues
- **Cons:** Misses complex phrasing issues

**Quick Start:**

```bash
cd services/modal-lt && modal deploy app.py
```

---

## Shared Package

Shared code library used by both the Frontend/API (TypeScript) and Python Services.

**Location:** `packages/shared/`

### Structure

- `ts/`: TypeScript source (types, validation schemas, utilities)
- `py/`: Python source (Pydantic models corresponding to TS types)

### TypeScript

**Building:**

```bash
cd packages/shared
npm run build
```

**Usage:**

```typescript
import { AssessmentResults, mapScoreToCEFR } from "@writeo/shared";
```

### Python

**Installation:**

```bash
cd packages/shared/py
pip install -e .
```

**Usage:**

```python
from schemas import AssessmentResults, map_score_to_cefr
```

**Note:** When changing a data structure, you must update both the TypeScript `zod` schemas/interfaces AND the Python `pydantic` models to ensure compatibility across the system.

---

## References

- [Deployment Guide](../operations/deployment.md)
- [System Architecture](../architecture/overview.md)
- [Add New Service](adding-services.md)
