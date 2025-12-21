# Operations Guide

Essential operations information for running Writeo in production.

**Quick Mode Switching:** See [MODES.md](modes.md) for easy mode switching guide.

## Environment Variables

### File Structure

You need **different files for different parts**:

- **API Worker**: `apps/api-worker/.dev.vars` (Cloudflare Workers standard)
- **Web App**: `apps/web/.env.local` (Next.js standard)
- **Tests**: `.env.local` in project root

### API Worker

**Production:** Set via `wrangler secret put`:

- **LLM Provider** (choose one):
  - `LLM_PROVIDER=openai` + `OPENAI_API_KEY` - Cost-effective option (GPT-4o-mini)
  - `LLM_PROVIDER=groq` + `GROQ_API_KEY` - Ultra-fast option (Llama 3.3 70B)

- **Modal Service URLs** (Set via `wrangler secret put`):
  - `MODAL_DEBERTA_URL` - AES-DEBERTA Service (Primary Scorer)
  - `MODAL_GEC_URL` - GEC-SEQ2SEQ Service (Grammar)
  - `MODAL_GECTOR_URL` - GEC-GECTOR Service (Fast Grammar)
  - `MODAL_FEEDBACK_URL` - AES-FEEDBACK Service (Experimental)

  - `MODAL_LT_URL` - LanguageTool Service (Optional)

- **API Worker Config**:
  - `API_KEY` - API authentication key (must match web app and Modal services)
  - `TEST_API_KEY` (optional) - Test key with higher rate limits
  - `ALLOWED_ORIGINS` (optional) - CORS origins (default: all)
  - `LT_LANGUAGE` (optional) - Default language code (default: `"en-GB"`)
  - `AI_MODEL` (optional) - Override default model name

**Modal Services:** Set via `modal secret create`:

- `MODAL_API_KEY` - API key for Modal services (should match Cloudflare Worker `API_KEY`)

**Local Development:** Copy `apps/api-worker/.dev.vars.example` to `.dev.vars` and fill in your values:

```bash
cp apps/api-worker/.dev.vars.example apps/api-worker/.dev.vars
# Then edit .dev.vars with your values
```

### Web App

**Local Development:** Copy `apps/web/.env.example` to `.env.local` and fill in your values:

```bash
cp apps/web/.env.example apps/web/.env.local
# Then edit .env.local with your values
```

Example `.env.local`:

```bash
API_BASE_URL=http://localhost:8787
API_KEY=your-key
```

**Production:** Set via Cloudflare Dashboard or `wrangler.toml`:

```toml
[vars]
API_BASE_URL = "https://your-api-worker.workers.dev"
# API_KEY should be set as secret
```

### Tests

**Copy `.env.example` to `.env.local` in project root:**

```bash
cp .env.example .env.local
# Then edit .env.local with your values
```

**Note:** `API_KEY` must match the API worker key (or use `TEST_API_KEY` for higher rate limits).

## Observability

### Logging

**Cloudflare Workers Logging:**

Logging is **enabled by default**. View logs via:

1. **Cloudflare Dashboard (Recommended)**: Workers & Pages → Your Worker → Observability → Logs
2. **Command Line**:
   ```bash
   ./scripts/check-logs.sh api-worker "error" 20
   ```

### Observability Dashboard

**Location:** Cloudflare Dashboard → Workers & Pages → Your Worker → Observability

**Features:**

- Real-time logs
- Traces (5% sampling rate)
- Metrics (CPU, requests, errors)

### Monitoring Modal Services

Use the `modal` CLI to view logs for specific services.

**Primary Scoring:**

```bash
modal app logs writeo-deberta  # AES-DEBERTA (Primary)
```

**Grammar Correction:**

```bash
modal app logs writeo-gec-service     # GEC-SEQ2SEQ (High Precision)
modal app logs writeo-gector-service  # GEC-GECTOR (Low Latency)
```

**Other Services:**

```bash
modal app logs writeo-feedback # AES-FEEDBACK (Experimental)

modal app logs writeo-lt       # GEC-LT (Legacy)
```

## Performance

**Typical Response Times:**

- **Full Assessment (Warm)**: 3-5s (using Parallel execution)
- **Full Assessment (Cold)**: 15-20s (if multiple services cold start)

**Service Latency Breakdown:**

- **AES-DEBERTA**: ~300-500ms (Warm)
- **GEC-GECTOR**: ~1-2s (Warm)
- **GEC-SEQ2SEQ**: ~12-16s (Warm) - _Slowest component_
- **LLM Feedback**: 2-5s (OpenAI) vs <1s (Groq)

**Bottlenecks:**

- **GEC-SEQ2SEQ**: High precision but significantly slower than GECToR. Use only when precision is critical.
- **Cold Starts**: Modal services take 10-20s to boot. Use "Turbo Mode" (Keep-Warm) for production.

**Optimizations:**

- **Parellel Execution**: The API Worker requests all assessors in parallel.
- **GEC Selection**: Prefer GECToR (`gecGector`) for real-time feedback; use Seq2Seq (`gecSeq2seq`) for final checks.
- **LLM Streaming**: Feedback is streamed to the user to reduce perceived latency.

## Monitoring Metrics

**Key Metrics to Watch:**

- **Worker**: Request count, Error rate (5xx), CPU Time per request.
- **Modal**: Active Containers, Backlog, GPU Memory usage.
- **Storage**: R2 Class A/B operations, KV Read/Write units.

## Troubleshooting

**Modal "Service Unavailable" / 500 Errors:**

- Check specific service logs: `modal app logs writeo-deberta`
- Verify `MODAL_*_URL` secrets in `wrangler secret list`.

**Slow Response Times:**

- Check `X-Timing-Slowest` header in response.
- If `GEC-SEQ2SEQ` is slow, verify if user needs high-precision (disable in `assessors.json` if not).
- If Cold Starts are frequent, switch to Turbo Mode (`./scripts/set-mode.sh turbo`).

## References

- [System Architecture](../architecture/overview.md)
- [DEPLOYMENT.md](deployment.md) - Deployment guide
- [Interactive API Docs](https://writeo-api-worker.rob-gilks.workers.dev/docs) - API specification (Swagger UI)
