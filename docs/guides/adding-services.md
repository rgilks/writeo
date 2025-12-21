# Adding New Assessor Services

This guide explains how to add new Modal-based assessor services to Writeo.

---

## Quick Reference

| Task                    | Files to Modify                                                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Add new Modal service   | 1. `service-registry.ts`<br>2. `modal/client.ts` & `modal/types.ts`<br>3. `config/assessors.json` & `services/config.ts` |
| Modify existing service | Usually just `service-registry.ts`                                                                                       |

---

## Adding a New Modal Service

### Step 1: Add to Assessor Registry

**File:** `apps/api-worker/src/services/submission/service-registry.ts`

Add an entry to `ASSESSOR_REGISTRY`:

```typescript
{
  assessorId: "NEW-SERVICE",              // Unique ID shown to frontend (matches ASSESSOR_IDS constant)
  id: "newservice",                       // Short ID for internal use
  displayName: "My New Service",          // Human-readable name
  type: "grader" | "feedback",            // grader = scoring, feedback = errors
  configPath: "features.assessors.xxx",   // Path in AppConfig
  timingKey: "5x_newservice_fetch",       // For performance tracking
  model: "model-name",                    // ML model identifier for metadata

  // createRequest: Creates the Promise<Response> for the service call
  createRequest: (text, modal, answerId, config) => {
      // access config if needed, e.g. config.features.languageTool.language
      return modal.newServiceMethod(text);
  },

  parseResponse: (json) => json as NewServiceResult,

  createAssessor: (data, text) => {
    const d = data as NewServiceResult;
    return {
      id: "NEW-SERVICE",
      name: "My New Service",
      type: "grader",
      overall: d.score,          // For graders
      // meta: { edits: d.edits }  // For feedback services
    };
  },
}
```

### Step 2: Add ModalClient Method

**File:** `apps/api-worker/src/services/modal/types.ts`
Add the method definition to the `ModalService` interface:

```typescript
export interface ModalService {
  // ... existing methods
  newServiceMethod(text: string): Promise<Response>;
}
```

**File:** `apps/api-worker/src/services/modal/client.ts`
Implement the method in `ModalClient`:

```typescript
async newServiceMethod(text: string): Promise<Response> {
  return this.postJson(`${this.config.modal.newServiceUrl}/endpoint`, { text });
}
```

**File:** `apps/api-worker/src/services/modal/mock.ts`
Implement the mock method in `MockModalClient` for testing:

```typescript
async newServiceMethod(text: string): Promise<Response> {
  return new Response(JSON.stringify({ score: 5.0 }), { status: 200 });
}
```

### Step 3: Add Configuration

**File:** `apps/api-worker/src/config/assessors.json`
Add the feature flag default:

```json
{
  "_defaults": {
    "newService": "NEW-SERVICE - Description"
  },
  "newService": true
}
```

**File:** `apps/api-worker/src/services/config.ts`

1. Add the URL to `AppConfig["modal"]`.
2. Map the environment variable in `buildConfig`.
3. Update `AppConfig["features"]["assessors"]` to read from `assessors.json`.

---

## Service Types

### Grader Services

Return a score (e.g., CEFR level):

```typescript
type: "grader",
createAssessor: (data) => ({
  id: "AES-XXX",
  type: "grader",
  overall: data.score,    // Numeric score
  label: data.cefr_level, // Display label
})
```

**Examples:** AES-FEEDBACK, AES-DEBERTA

### Feedback Services

Return errors/edits to highlight:

```typescript
type: "feedback",
createAssessor: (data) => ({
  id: "GEC-XXX",
  type: "feedback",
  meta: {
    edits: data.edits,           // Array of edits for UI highlighting
    correctedText: data.corrected,
  }
})
```

**Examples:** GEC-SEQ2SEQ, GEC-GECTOR, GEC-LT

---

## Response Type Definitions

Add your response type at the top of `service-registry.ts`:

```typescript
export interface NewServiceResult {
  score: number;
  // ... other fields matching the JSON response from Python
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
# app.py
import os
import modal

app = modal.App("writeo-xxx")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi[standard]", "torch", "transformers"
)
image = image.add_local_dir(os.path.dirname(__file__), remote_path="/app", copy=True)

volume = modal.Volume.from_name("writeo-xxx-models", create_if_missing=True)

@app.function(image=image, volumes={"/vol": volume}, gpu="T4", scaledown_window=60)
@modal.asgi_app()
def fastapi_app():
    import sys
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")
    from api import create_fastapi_app
    return create_fastapi_app()
```

```python
# api.py
from fastapi import FastAPI, Request

def create_fastapi_app():
    app = FastAPI()

    @app.post("/endpoint")
    async def score(request: Request):
        data = await request.json()
        # Your ML logic here
        return {"score": 0.5}

    return app
```

Deploy: `modal deploy main.py`
