# Deployment Guide

Complete guide for deploying Writeo to production.

## Pre-Deployment Checklist

- [ ] Test locally: `wrangler dev` and `npm run dev`
- [ ] Choose operational mode: **Cheap Mode** (cost-optimized) or **Turbo Mode** (performance-optimized)
- [ ] Verify all secrets are set (see Step 3)
- [ ] Run tests: `npm run test:all` (or let pre-push hook handle it)
- [ ] Code is formatted: `npm run format:check`
- [ ] Type checking passes: `npm run type-check`

## Operational Modes

Writeo supports two operational modes:

### 🪙 Cheap Mode (Cost-Optimized)

**Setup:**

- Set `LLM_PROVIDER=openai` and `OPENAI_API_KEY`
- Modal services automatically scale-to-zero (default 30-second scaledown)
- **Cost:** ~$7.60-8.50/month (100 submissions/day)
- **Performance:** 8-15s cold start, 3-10s warm

### ⚡ Turbo Mode (Performance-Optimized)

**Setup:**

- Set `LLM_PROVIDER=groq` and `GROQ_API_KEY`
- Configure Modal services with reduced `scaledown_window` or keep warm
- **Cost:** ~$65-80/month (100 submissions/day)
- **Performance:** 2-5s first request, 1-3s warm

See [MODES.md](MODES.md) for quick mode switching guide.  
See [OPERATIONS.md](OPERATIONS.md) for detailed mode comparison and configuration.

## Quick Deployment

**For most deployments, use the automated script:**

```bash
./scripts/deploy-all.sh
```

This script handles all deployment steps automatically:

- Deploys Modal service
- Extracts and configures Modal URL secrets
- Builds shared packages
- Deploys all workers and frontend
- Optionally runs smoke tests

See below for manual step-by-step instructions if needed.

## Step-by-Step Deployment

### Step 1: Setup Cloudflare Resources

Run the setup script (creates R2 bucket and KV namespace):

```bash
./scripts/setup.sh
```

Or manually:

```bash
wrangler r2 bucket create writeo-data
wrangler kv:namespace create "WRITEO_RESULTS"  # Copy ID to wrangler.toml
wrangler kv:namespace create "WRITEO_RESULTS" --preview  # Copy preview_id
```

### Step 2: Deploy Modal Services

#### 2.1 Deploy Essay Scoring Service

```bash
# Recommended: Use deployment script
./scripts/deploy-modal.sh

# Or manually
cd services/modal-essay
modal deploy app.py  # Copy endpoint URL from output
```

#### 2.2 Deploy LanguageTool Service (Optional)

```bash
cd services/modal-lt
modal deploy app.py  # Copy endpoint URL from output
```

**Test endpoints:** `curl https://your-endpoint/health`

### Step 3: Configure Secrets

```bash
cd apps/api-worker

# Set Essay Scoring Modal URL secret
wrangler secret put MODAL_GRADE_URL
# Paste the Essay Scoring Modal endpoint URL when prompted

# Set LanguageTool Modal URL secret (optional)
wrangler secret put MODAL_LT_URL
# Paste the LanguageTool Modal endpoint URL when prompted, or press Enter to skip

# Set LanguageTool language (optional)
wrangler secret put LT_LANGUAGE
# Paste language code (default: "en-GB") or press Enter to skip

# Set API authentication key
wrangler secret put API_KEY
# Paste your API key when prompted (generate a secure random string or JWT)

# Choose your LLM provider (set one):
# Option 1: OpenAI (GPT-4o-mini) - Recommended for cost efficiency
wrangler secret put LLM_PROVIDER
# Enter "openai"
wrangler secret put OPENAI_API_KEY
# Paste your OpenAI API key from https://platform.openai.com/api-keys

# Option 2: Groq (Llama 3.3 70B) - Recommended for speed
# wrangler secret put LLM_PROVIDER
# Enter "groq"
# wrangler secret put GROQ_API_KEY
# Paste your Groq API key from https://console.groq.com

# Verify secrets are set
wrangler secret list
```

**Important**: The `API_KEY` must match the key used in your frontend's `.env.local` file (for local development) or Cloudflare environment variables (for production). See [OPERATIONS.md](OPERATIONS.md) for details.

### Step 4: Verify Configuration

Check `apps/api-worker/wrangler.toml`:

- R2 bucket: `writeo-data`
- KV namespace IDs set (production + preview)

### Step 5: Deploy API Worker

```bash
cd apps/api-worker
wrangler dev  # Test locally first
wrangler deploy  # Deploy to production
```

Test: `curl https://your-worker.workers.dev/health`

### Step 6: Deploy Frontend

```bash
cd apps/web
npm run deploy
```

**Note:** The `deploy` script automatically runs `build:cf` (which builds Next.js and OpenNext for Cloudflare) before deploying. Frontend uses Server Actions, so `API_BASE_URL` should be set in Cloudflare environment (not `.env.local`).

## Automated Deployment (GitHub Actions)

The project includes GitHub Actions workflows for automated deployment:

**Workflow**: `.github/workflows/deploy-and-test.yml`

- **Triggers**: Runs automatically on push to `main` branch
- **Process**:
  1. Builds shared packages
  2. Deploys API worker to Cloudflare
  3. Deploys web app to Cloudflare
  4. Runs API and E2E tests against deployed production site

**Setup Required:**

Configure GitHub secrets in Settings → Secrets and variables → Actions:

- `API_KEY` - API authentication key (must match Cloudflare Workers secret)
- `CLOUDFLARE_API_TOKEN` - Cloudflare API token with Workers edit permissions
- `CLOUDFLARE_ACCOUNT_ID` - Your Cloudflare account ID
- `TEST_API_KEY` (optional) - Test key with higher rate limits

**Note**: Local tests are handled by git pre-push hooks, so the GitHub Action focuses on deployment and production verification.

## Post-Deployment Testing

**Smoke Test:**

```bash
API_BASE=https://your-worker.workers.dev ./scripts/smoke.sh
```

Verifies: Question/answer/submission creation and results retrieval.

**Automated Testing:**

The GitHub Actions workflow automatically runs tests after deployment. You can also run tests manually:

```bash
# Test against production
API_BASE=https://your-worker.workers.dev API_KEY=your-key npm test
PLAYWRIGHT_BASE_URL=https://your-site.com npm run test:e2e
```

### Manual API Testing

```bash
export API_BASE="https://your-worker.workers.dev"
export API_KEY="your-api-key"

# Quick test (creates question, answer, submission, polls for results)
QUESTION_ID=$(uuidgen) && ANSWER_ID=$(uuidgen) && SUBMISSION_ID=$(uuidgen)
# Answers must be sent inline with submissions
# Questions can be sent inline or referenced by ID
curl -X PUT "$API_BASE/text/submissions/$SUBMISSION_ID" -H "Authorization: Token $API_KEY" -H "Content-Type: application/json" -d "{\"submission\":[{\"part\":1,\"answers\":[{\"id\":\"$ANSWER_ID\",\"question-number\":1,\"question-id\":\"$QUESTION_ID\",\"question-text\":\"Describe your weekend.\",\"text\":\"I went to the park.\"}]}],\"template\":{\"name\":\"generic\",\"version\":1}}"

# Or reference an existing question (create question first):
curl -X PUT "$API_BASE/text/questions/$QUESTION_ID" -H "Authorization: Token $API_KEY" -H "Content-Type: application/json" -d '{"text":"Describe your weekend."}'
# Then submit with question reference:
curl -X PUT "$API_BASE/text/submissions/$SUBMISSION_ID" -H "Authorization: Token $API_KEY" -H "Content-Type: application/json" -d "{\"submission\":[{\"part\":1,\"answers\":[{\"id\":\"$ANSWER_ID\",\"question-number\":1,\"question-id\":\"$QUESTION_ID\",\"text\":\"I went to the park.\"}]}],\"template\":{\"name\":\"generic\",\"version\":1}}"
curl -H "Authorization: Token $API_KEY" "$API_BASE/text/submissions/$SUBMISSION_ID"
```

See [docs/SPEC.md](docs/SPEC.md) for complete API examples.

### Frontend Testing

1. Open frontend URL
2. Submit an essay
3. Verify results appear with scores, CEFR level, and grammar errors

## Monitoring

**Cloudflare Workers:**

- Logs: `wrangler tail` (use `./scripts/check-logs.sh api-worker` for safe access)
- Dashboard: https://dash.cloudflare.com → Workers & Pages → Analytics

**Modal Services:**

- Logs: `modal app logs writeo-essay` or `modal app logs writeo-lt`
- Dashboard: https://modal.com → Your Apps

**Key Metrics:** Request count/errors, response times, KV/R2 usage

See [docs/OPERATIONS.md](docs/OPERATIONS.md) for detailed logging and monitoring guidance.

## Troubleshooting

### Common Issues

**Modal service errors:**

- Verify `MODAL_GRADE_URL`, `API_KEY`, and `OPENAI_API_KEY` secrets are set correctly
- Verify `MODAL_LT_URL` is set if grammar checking is needed (optional)
- Check Modal service logs in dashboard: `modal app logs writeo-essay` or `modal app logs writeo-lt`
- Test Modal endpoints directly:
  - Essay Scoring: `curl https://your-essay-scoring-endpoint/health`
  - LanguageTool: `curl https://your-lt-endpoint/health`
- Verify Modal authentication: `modal token show`

**KV not storing results:**

- Verify KV namespace IDs in wrangler.toml
- Check KV namespace exists: `wrangler kv:namespace list`
- Verify write permissions
- Check worker logs for KV errors

**R2 access errors:**

- Verify bucket name matches: `writeo-data`
- Check R2 bucket exists: `wrangler r2 bucket list`
- Verify bucket bindings in wrangler.toml
- Check bucket permissions

**Results not appearing:**

- Check API worker logs: `wrangler tail`
- Verify Modal service is accessible
- Check KV namespace permissions
- Verify submission was created successfully

**CORS errors:**

- Ensure API worker has CORS headers configured
- Check allowed origins in CORS configuration
- Verify frontend is using correct API base URL

### Rollback Plan

If issues occur:

1. **Revert to previous worker version:**
   - Cloudflare dashboard → Workers → Versions
   - Select previous version and promote

2. **Or redeploy previous commit:**

   ```bash
   git checkout <previous-commit>
   cd apps/api-worker
   wrangler deploy
   ```

3. **Disable Modal services:**
   - Remove or update `MODAL_GRADE_URL` secret (disables essay scoring)
   - Remove or update `MODAL_LT_URL` secret (disables grammar checking)
   - API will return errors but won't crash

## Success Criteria

- [ ] All smoke tests pass
- [ ] Frontend can submit and receive results
- [ ] Results format matches specification
- [ ] No errors in worker logs
- [ ] Modal service responds within acceptable time (< 30s)
- [ ] CORS headers work correctly
- [ ] KV storage persists results

## Additional Resources

- [Cloudflare Workers Docs](https://developers.cloudflare.com/workers/)
- [Modal Documentation](https://modal.com/docs)
- [Wrangler CLI Reference](https://developers.cloudflare.com/workers/wrangler/)
