# Adding New Assessor Services

This guide explains how to add new Modal-based assessor services to Writeo.

---

## Quick Reference

| Task                    | Files to Modify                    |
| ----------------------- | ---------------------------------- |
| Add new Modal service   | 3 files (see below)                |
| Modify existing service | Usually just `service-registry.ts` |

---

## Adding a New Modal Service

### Step 1: Add to Assessor Registry

**File:** `apps/api-worker/src/services/submission/service-registry.ts`

Add an entry to `ASSESSOR_REGISTRY`:

```typescript
{
  assessorId: "T-NEW-SERVICE",           // Unique ID shown to frontend
  id: "newservice",                       // Short ID for internal use
  displayName: "My New Service",          // Human-readable name
  type: "grader" | "feedback",            // grader = scoring, feedback = errors
  configPath: "features.assessors.xxx",   // Path in AppConfig
  timingKey: "5x_newservice_fetch",       // For performance tracking
  model: "model-name",                    // ML model identifier
  createRequest: (text, modal) => modal.newServiceMethod(text),
  parseResponse: (json) => json as NewServiceResult,
  createAssessor: (data) => {
    const d = data as NewServiceResult;
    return {
      id: "T-NEW-SERVICE",
      name: "My New Service",
      type: "grader",
      overall: d.score,          // For graders
      // meta: { edits: d.edits }  // For feedback services
    };
  },
}
```

### Step 2: Add ModalClient Method

**File:** `apps/api-worker/src/services/modal/client.ts`

```typescript
async newServiceMethod(text: string): Promise<Response> {
  return this.postJson(`${this.config.modal.newServiceUrl}/endpoint`, { text });
}
```

Also add to `modal/types.ts` interface and `modal/mock.ts`.

### Step 3: Add Config

**File:** `apps/api-worker/src/config/assessors.json`

```json
{
  "_defaults": {
    "newService": "T-NEW-SERVICE - Description"
  },
  "newService": true
}
```

**File:** `apps/api-worker/src/services/config.ts`

Add URL config and assessor flag.

---

## Service Types

### Grader Services

Return a score (e.g., CEFR level):

```typescript
type: "grader",
createAssessor: (data) => ({
  id: "T-AES-XXX",
  type: "grader",
  overall: data.score,    // Numeric score
  label: data.cefr_level, // Display label
})
```

**Examples:** T-AES-CORPUS, T-AES-FEEDBACK

### Feedback Services

Return errors/edits to highlight:

```typescript
type: "feedback",
createAssessor: (data) => ({
  id: "T-GEC-XXX",
  type: "feedback",
  meta: {
    edits: data.edits,           // Array of edits
    correctedText: data.corrected,
  }
})
```

**Examples:** T-GEC-SEQ2SEQ, T-GEC-GECTOR

---

## Response Type Definitions

Add your response type at the top of `service-registry.ts`:

```typescript
export interface NewServiceResult {
  score: number;
  // ... other fields
}
```

---

## Testing Your Service

1. **Type check:** `npm run type-check`
2. **Unit tests:** `npm run test:unit`
3. **Integration test:**

   ```bash
   # Start local dev
   npm run dev

   # Submit an essay and check response includes your assessor
   ```

---

## Python Modal Service Template

Create a new service in `services/modal-xxx/`:

```python
# main.py
import modal
app = modal.App("writeo-xxx")

image = modal.Image.debian_slim().pip_install(
    "fastapi[standard]", "torch", "transformers"
)

@app.function(image=image, gpu="T4", scaledown_window=60)
@modal.asgi_app()
def fastapi_app():
    from api import create_fastapi_app
    return create_fastapi_app()
```

```python
# api.py
from fastapi import FastAPI

def create_fastapi_app():
    app = FastAPI()

    @app.post("/endpoint")
    async def score(request: Request):
        # Your ML logic here
        return {"score": 0.5}

    return app
```

Deploy: `modal deploy main.py`
