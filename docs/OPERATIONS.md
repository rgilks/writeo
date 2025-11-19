# Operations Guide

Essential operations information for running Writeo in production.

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
- `GROQ_API_KEY` - Groq API key for AI feedback
- `MODAL_LT_URL` (optional) - LanguageTool Modal endpoint
- `LT_LANGUAGE` (optional) - Default language code (default: `"en-GB"`)
- `AI_MODEL` (optional) - Groq model (default: `"llama-3.3-70b-versatile"`)
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
GROQ_API_KEY=your-groq-key
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

## Cost Optimization

**Free Tier Limits:**

- Cloudflare Workers: 100k requests/day
- Cloudflare Workers AI: 10k requests/day
- R2/KV: 10GB/100MB free

**Cost Per Submission:**

- **Base:** ~$0.015-0.02 (2 required Groq API calls)
- **With teacher feedback:** ~$0.02-0.03
- **Average:** ~$0.016-0.022

**Monthly Costs:**

**Infrastructure (Free Tier):**

- Cloudflare: $0
- Modal: ~$0.10-1.00/month
- **Total Infrastructure**: ~$0.12-1.15/month

**Groq API (Pay-Per-Use):**

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
