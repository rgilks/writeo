# Operations Guide

Essential operations information for running Writeo in production.

**Quick Mode Switching:** See [MODES.md](MODES.md) for easy mode switching guide.

## Environment Variables

### File Structure

You need **different files for different parts**:

- **API Worker**: `apps/api-worker/.dev.vars` (Cloudflare Workers standard)
- **Web App**: `apps/web/.env.local` (Next.js standard)
- **Tests**: `.env.local` in project root

### API Worker

**Production:** Set via `wrangler secret put`:

- `MODAL_GRADE_URL` - Essay Scoring Modal endpoint
- `API_KEY` - API authentication key (must match web app and Modal services)
- **LLM Provider** (choose one):
  - `LLM_PROVIDER=openai` + `OPENAI_API_KEY` - Cost-effective option (GPT-4o-mini)
  - `LLM_PROVIDER=groq` + `GROQ_API_KEY` - Ultra-fast option (Llama 3.3 70B)
- `MODAL_LT_URL` (optional) - LanguageTool Modal endpoint
- `LT_LANGUAGE` (optional) - Default language code (default: `"en-GB"`)
- `AI_MODEL` (optional) - Model name (default: `"gpt-4o-mini"` for OpenAI, `"llama-3.3-70b-versatile"` for Groq)
- `TEST_API_KEY` (optional) - Test key with higher rate limits
- `ALLOWED_ORIGINS` (optional) - CORS origins (default: all)

**Modal Services:** Set via `modal secret create`:

- `MODAL_API_KEY` - API key for Modal services (should match Cloudflare Worker `API_KEY`)

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

**Cloudflare Workers Logging:**

Logging is **enabled by default** for all Workers. All `console.log()`, `console.error()`, and `console.warn()` statements are automatically captured.

**View Logs:**

1. **Cloudflare Dashboard (Recommended for Production):**
   - Go to: https://dash.cloudflare.com → Workers & Pages → Your Worker → Logs
   - Real-time logs with filtering and search
   - No timeout issues, best for production monitoring

2. **Command Line (for quick checks):**

   ```bash
   # Use helper script (safe, with timeout)
   ./scripts/check-logs.sh api-worker "error" 20
   ./scripts/check-logs.sh api-worker "LLM Assessment" 50
   ./scripts/check-logs.sh api-worker "" 30  # Recent logs
   ```

3. **Direct wrangler command (with timeout):**
   ```bash
   timeout 10s npx wrangler tail --format json --search "error" | head -20
   ```

**Log Levels:**

- `console.log()` - Info/debug messages
- `console.warn()` - Warnings (via `safeLogWarn()`)
- `console.error()` - Errors (via `safeLogError()`)

**Log Format:**
All logs are automatically sanitized to remove sensitive data (API keys, tokens, etc.) via `safeLogError()`, `safeLogWarn()`, and `safeLogInfo()` utilities.

**Modal Services:**

```bash
modal app logs writeo-essay
modal app logs writeo-lt
```

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

Writeo supports two operational modes optimized for different use cases. See [MODES.md](MODES.md) for detailed mode switching guide.

**Quick Summary:**

- **🪙 Cheap Mode**: OpenAI GPT-4o-mini, scale-to-zero → ~$7.60-8.50/month (100/day)
- **⚡ Turbo Mode**: Groq Llama 3.3 70B, keep warm → ~$25-40/month (100/day)

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

- **Base:** ~$0.0048-0.006 (2 required API calls)
- **With teacher feedback:** ~$0.006-0.007
- **Average:** ~$0.006

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

- Low usage (10/day): ~$1.80/month
- Moderate (100/day): ~$18/month
- High (1,000/day): ~$180/month
- Maximum (14,400/day, rate limited): ~$2,592/month

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
