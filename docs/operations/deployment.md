# Deployment Guide

Complete guide for deploying Writeo to production.

## Pre-Deployment Checklist

- [ ] Test locally: `wrangler dev` and `npm run dev`
- [ ] Choose operational mode coverage: **Production** (Recommended) or **Legacy/Minimal**
- [ ] Verify all secrets are set (see Step 3)
- [ ] Run tests: `npm run test:all` (or let pre-push hook handle it)
- [ ] Code is formatted: `npm run format:check`
- [ ] Type checking passes: `npm run type-check`

## Operational Modes

Writeo's architecture has evolved to use specialized models for different tasks.

### 🚀 Production Mode (Recommended)

Uses the full suite of high-performance models for maximum accuracy:

- **AES-DEBERTA:** Primary assessor (Dimensional scoring, DeBERTa-v3-large)
- **AES-CORPUS:** Secondary assessor (Verification, RoBERTa-base)
- **GEC-SEQ2SEQ:** High-precision grammar correction (Flan-T5-base)
- **GEC-GECTOR:** Low-latency grammar correction (RoBERTa-base)

**Cost:** Variable based on usage (Modal GPU time + LLM tokens).
**Performance:** High accuracy, parallel execution of scoring and correction.

### 🪙 Legacy / Minimal Mode

Uses legacy or fewer models to save costs:

- **AES-ESSAY:** Legacy assessor (Single model, deprecated but functional)
- **LanguageTool:** Rule-based grammar checking (CPU-only)

**Note:** The application is configured to prefer the Production models (`AES-DEBERTA`) by default. Running in minimal mode requires adjusting `apps/api-worker/src/config/assessors.json`.

## Quick Deployment

> [!WARNING]
> The automated scripts (`./scripts/deploy-all.sh`, `./scripts/deploy-modal.sh`) primarily support the legacy configuration. For a full production deployment, follow the **Step-by-Step Deployment** below.

## Step-by-Step Deployment

### Step 1: Setup Cloudflare Resources

Run the setup script (creates R2 bucket and KV namespace):

```bash
./scripts/setup.sh
```

Or manually:

```bash
wrangler r2 bucket create writeo-data-1
wrangler kv:namespace create "WRITEO_RESULTS" # Copy ID to wrangler.toml
wrangler kv:namespace create "WRITEO_RESULTS" --preview # Copy preview_id
```

### Step 2: Deploy Modal Services

Deploy the services required for your chosen operational mode.

**Prerequisites:**

- Install Modal: `pip install modal`
- Authenticate: `modal token new`

#### 2.1 Primary Scoring (AES-DEBERTA) - _Required for Production_

Dimensional scoring model (DeBERTa-v3-large).

```bash
cd services/modal-deberta
modal deploy app.py
# Copy the endpoint URL
```

#### 2.2 Secondary Scoring (AES-CORPUS) - _Recommended_

Verification model trained on corpus data.

```bash
cd services/modal-corpus
modal deploy app.py
# Copy the endpoint URL
```

#### 2.3 Grammar Correction (GEC-SEQ2SEQ) - _Recommended_

High-precision grammar correction (Flan-T5).

```bash
cd services/modal-gec
modal deploy main.py
# Copy the endpoint URL
```

#### 2.4 Fast Grammar Correction (GEC-GECTOR) - _Recommended_

Fast, token-classification based correction.

```bash
cd services/modal-gector
modal deploy main.py
# Copy the endpoint URL
```

#### 2.5 Feedback Model (AES-FEEDBACK) - _Optional_

Experimental model for span-level feedback.

```bash
cd services/modal-feedback
modal deploy app.py
# Copy the endpoint URL
```

#### 2.6 Legacy Scoring (AES-ESSAY) - _Legacy/Fallback_

Required by the API worker configuration unless explicitly disabled, or a placeholder URL is provided.

```bash
cd services/modal-essay
modal deploy app.py
# Copy the endpoint URL
```

#### 2.7 LanguageTool (GEC-LT) - _Optional_

Rule-based grammar checking.

```bash
cd services/modal-lt
modal deploy app.py
# Copy the endpoint URL
```

### Step 3: Configure Secrets

Set the secrets in Cloudflare Workers to connect your API to the deployed services.

```bash
cd apps/api-worker

# --- Service URLs ---

# Primary Scorer (DeBERTa)
wrangler secret put MODAL_DEBERTA_URL
# Paste modal-deberta URL

# Secondary Scorer (Corpus)
wrangler secret put MODAL_CORPUS_URL
# Paste modal-corpus URL

# Grammar Correction (Seq2Seq)
wrangler secret put MODAL_GEC_URL
# Paste modal-gec URL

# Fast Grammar Correction (GECToR)
wrangler secret put MODAL_GECTOR_URL
# Paste modal-gector URL

# Feedback Model (Optional)
wrangler secret put MODAL_FEEDBACK_URL
# Paste modal-feedback URL

# Legacy Scorer (Required by config, can be dummy if unused)
wrangler secret put MODAL_GRADE_URL
# Paste modal-essay URL

# LanguageTool (Optional)
wrangler secret put MODAL_LT_URL
# Paste modal-lt URL

# --- API Configuration ---

# Set API authentication key
wrangler secret put API_KEY
# Paste your API key (generate a secure random string)

# Test API Key (Optional, for higher rate limits)
wrangler secret put TEST_API_KEY
# Paste a secondary key

# --- LLM Provider ---

# Helper: Set provider ("openai" or "groq")
wrangler secret put LLM_PROVIDER

# If using OpenAI:
wrangler secret put OPENAI_API_KEY
# Paste OpenAI Key

# If using Groq:
wrangler secret put GROQ_API_KEY
# Paste Groq Key
```

### Step 4: Verify Configuration

1.  Check `apps/api-worker/wrangler.toml`:
    - R2 bucket: `writeo-data-1`
    - KV namespace IDs present
2.  Check `apps/api-worker/src/config/assessors.json`:
    - Ensure the flags (e.g., `"deberta": true`) match the services you deployed. Disable any services you didn't deploy to prevent errors.

### Step 5: Deploy API Worker

```bash
cd apps/api-worker
wrangler deploy
```

Test health: `curl https://your-worker.workers.dev/health`

### Step 6: Deploy Frontend

```bash
cd apps/web
npm run deploy
```

## Automated Deployment (GitHub Actions)

The repository includes `.github/workflows/deploy-and-test.yml`.
Ensure you add the following secrets to your GitHub repository for it to work:

- `CLOUDFLARE_API_TOKEN`
- `CLOUDFLARE_ACCOUNT_ID`
- `API_KEY`

**Note:** The GitHub Action currently deploys the Worker and Web App. It does **not** automatically deploy Modal services. You must deploy those manually using the steps above.

## Troubleshooting

### "Service Unavailable" or 500 Errors

- Check `apps/api-worker` logs: `wrangler tail`
- Verify the specific `MODAL_*_URL` secret is correct.
- Ensure the service is running in Modal dashboard.
- If a service is missing, disable it in `assessors.json` and redeploy the worker.

### "Authentication Failed" on Modal

- Ensure you set the `MODAL_API_KEY` secret within Modal if your services require it (check individual service `app.py` / `main.py`). Not all services enforce this at the Modal layer, but it's best practice.

### Cost Management

- Use `assessors.json` to disable expensive models (e.g., `deberta`, `gecSeq2seq`) if not needed for development.
- Adjust `SCALEDOWN_WINDOW_SECONDS` in each Modal app file to control keep-warm costs.
