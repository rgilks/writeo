# Operations Guide

Essential operations information for running Writeo in production.

**Quick Mode Switching:** See [MODES.md](MODES.md) for easy mode switching guide.

## Environment Variables

### File Structure

You need **different files for different parts**:

- **API Worker**: `apps/api-worker/.dev.vars` (Cloudflare Workers standard)
- **Web App**: `apps/web/.env.local` (Next.js standard)
- **Tests**: `.env.local` in project root

**You don't need `.env` files** - `.env.local` is sufficient. Some configs load `.env` as a fallback, but `.env.local` takes precedence.

### API Worker

**Production:** Set via `wrangler secret put`:

- `MODAL_GRADE_URL` - Essay Scoring Modal endpoint
- `API_KEY` - API authentication key (must match web app)
- **LLM Provider** (choose one):
  - `LLM_PROVIDER=openai` + `OPENAI_API_KEY` - Cost-effective option (GPT-4o-mini)
  - `LLM_PROVIDER=groq` + `GROQ_API_KEY` - Ultra-fast option (Llama 3.3 70B)
- `MODAL_LT_URL` (optional) - LanguageTool Modal endpoint
- `LT_LANGUAGE` (optional) - Default language code (default: `"en-GB"`)
- `AI_MODEL` (optional) - Model name (default: `"gpt-4o-mini"` for OpenAI, `"llama-3.3-70b-versatile"` for Groq)
- `TEST_API_KEY` (optional) - Test key with higher rate limits
- `ALLOWED_ORIGINS` (optional) - CORS origins (default: all)

**Local Development:** Copy `apps/api-worker/.dev.vars.example` to `.dev.vars` and fill in your values:

```bash
cp apps/api-worker/.dev.vars.example apps/api-worker/.dev.vars
# Then edit .dev.vars with your values
```

Example `.dev.vars`:

```bash
API_KEY=your-key
MODAL_GRADE_URL=https://your-endpoint.modal.run
# Choose your LLM provider:
LLM_PROVIDER=openai  # or "groq"
OPENAI_API_KEY=your-openai-key  # Required if LLM_PROVIDER=openai
GROQ_API_KEY=your-groq-key  # Required if LLM_PROVIDER=groq
MODAL_LT_URL=https://your-lt-endpoint.modal.run  # optional
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

Example `.env.local`:

```bash
API_KEY=your-key
API_BASE=http://localhost:8787
PLAYWRIGHT_BASE_URL=http://localhost:3000
```

**Note:** `API_KEY` must match the API worker key (or use `TEST_API_KEY` for higher rate limits).

## Logging

**Cloudflare Workers:**

```bash
# Use helper script (recommended)
./scripts/check-logs.sh api-worker "error" 20

# Or Cloudflare Dashboard (best for production)
# https://dash.cloudflare.com → Workers & Pages → Logs
```

**Modal Services:**

```bash
modal app logs writeo-essay
modal app logs writeo-lt
```

⚠️ **Never use `wrangler tail` without timeout** - it blocks indefinitely.

## Performance

**Typical Response Times:**

- Warm: 3-10s (full processing)
- Cold: 8-15s (Modal cold start)

**Bottlenecks:**

- AI Feedback: 3-8s per answer
- Modal cold starts: 8-15s (Essay Scoring), 2-5s (LanguageTool)

**Optimizations:**

- Parallel processing (Essay Scoring + LanguageTool + Relevance)
- Combined AI feedback calls
- Model caching via Modal Volumes

## Monitoring

**Key Metrics:**

- Request count/errors
- Response times
- KV/R2 usage
- Error rates

**Timing Headers:**

- `X-Timing-Total`: Total processing time
- `X-Timing-Slowest`: Top 5 slowest operations
- `X-Timing-Data`: Full timing breakdown (JSON)

## Troubleshooting

**Modal errors:**

- Verify secrets: `wrangler secret list`
- Check Modal logs: `modal app logs writeo-essay`
- Test endpoint: `curl https://your-endpoint/health`

**KV/R2 errors:**

- Verify namespace IDs in `wrangler.toml`
- Check bucket exists: `wrangler r2 bucket list`
- Check namespace: `wrangler kv:namespace list`

**Results not appearing:**

- Check API logs: `./scripts/check-logs.sh api-worker`
- Verify Modal service accessible
- Check KV permissions

## Operational Modes

Writeo supports two operational modes optimized for different use cases:

### 🪙 Cheap Mode (Cost-Optimized)

**Configuration:**

- **LLM Provider:** OpenAI (GPT-4o-mini)
- **Modal Services:** Scale-to-zero (30-second scaledown window)
- **Best For:** Cost-conscious deployments, variable traffic, development/testing

**Performance:**

- **First request after inactivity:** 8-15s (Modal cold start)
- **Subsequent requests:** 3-10s (warm)
- **LLM latency:** ~1-3s per request

**Cost Breakdown:**

- **LLM API:** ~$0.0025 per submission
- **Modal Services:** Pay-per-use, scales to zero when idle
- **Monthly cost (100 submissions/day):** ~$7.50/month (LLM) + ~$0.10-1.00/month (Modal) = **~$7.60-8.50/month**

**Setup:**

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your-key
# Modal services automatically scale to zero (default behavior)
```

### ⚡ Turbo Mode (Performance-Optimized)

**Configuration:**

- **LLM Provider:** Groq (Llama 3.3 70B Versatile)
- **Modal Services:** Keep warm (reduced scaledown window or always-on)
- **Best For:** Production deployments requiring low latency, consistent performance

**Performance:**

- **First request:** 2-5s (Modal warm, Groq ultra-fast)
- **Subsequent requests:** 1-3s (all services warm)
- **LLM latency:** ~100-500ms per request (ultra-fast)

**Cost Breakdown:**

- **LLM API:** ~$0.02 per submission
- **Modal Services:** ~$5-20/month (keeping services warm, varies by traffic)
- **Monthly cost (100 submissions/day):** ~$60/month (LLM) + ~$5-20/month (Modal) = **~$65-80/month**

**Setup:**

```bash
LLM_PROVIDER=groq
GROQ_API_KEY=your-key
# Configure Modal services with reduced scaledown_window or keep warm
# See services/modal-essay/app.py and services/modal-lt/app.py
```

### Mode Comparison

| Feature                    | Cheap Mode            | Turbo Mode               |
| -------------------------- | --------------------- | ------------------------ |
| **LLM Provider**           | OpenAI (GPT-4o-mini)  | Groq (Llama 3.3 70B)     |
| **LLM Cost/Submission**    | ~$0.0025              | ~$0.02                   |
| **Modal Scaling**          | Scale-to-zero (30s)   | Keep warm                |
| **Cold Start**             | 8-15s (first request) | 2-5s (first request)     |
| **Warm Latency**           | 3-10s                 | 1-3s                     |
| **Monthly Cost (100/day)** | ~$7.60-8.50           | ~$65-80                  |
| **Best For**               | Cost optimization     | Performance optimization |

**Recommendation:**

- Use **Cheap Mode** for development, testing, or cost-sensitive production deployments
- Use **Turbo Mode** for production deployments requiring consistent low latency

## Cost Optimization

**Free Tier Limits:**

- Cloudflare Workers: 100k requests/day
- Cloudflare Workers AI: 10k requests/day
- R2/KV: 10GB/100MB free

**Cost Per Submission:**

**OpenAI (GPT-4o-mini)** - Cost-effective:

- **Base:** ~$0.0025 (2 required API calls)
- **With teacher feedback:** ~$0.003-0.004
- **Average:** ~$0.0027

**Groq (Llama 3.3 70B)** - Ultra-fast:

- **Base:** ~$0.015-0.02 (2 required API calls)
- **With teacher feedback:** ~$0.02-0.03
- **Average:** ~$0.016-0.022

**Monthly Costs:**

**Infrastructure (Free Tier):**

- Cloudflare: $0
- Modal: ~$0.10-1.00/month
- **Total Infrastructure**: ~$0.12-1.15/month

**LLM API (Pay-Per-Use) - OpenAI:**

- Low usage (10/day): ~$0.75/month
- Moderate (100/day): ~$7.50/month
- High (1,000/day): ~$75/month
- Maximum (14,400/day, rate limited): ~$1,080/month

**LLM API (Pay-Per-Use) - Groq:**

- Low usage (10/day): ~$6/month
- Moderate (100/day): ~$60/month
- High (1,000/day): ~$600/month
- Maximum (14,400/day, rate limited): ~$8,640/month

**Cost Controls:**

- Rate limiting: 10 submissions/minute per IP (prevents runaway costs)
- Word limits: 250-500 words per essay
- Text truncation: 15,000 chars max for AI processing
- Token limits: Reduced to minimize costs

**Scale-to-Zero:** No idle costs - services scale to zero when not in use.

See [COST_REVIEW.md](COST_REVIEW.md) for detailed cost analysis and guardrails.

## References

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guide
- [SPEC.md](SPEC.md) - API specification
