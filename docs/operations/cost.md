# Cost Review & Optimization Guide

**Review Frequency:** Quarterly  
**Last Reviewed:** 2025-12-20

---

## Executive Summary

Writeo supports multiple LLM providers for AI-powered feedback. This document provides:

- Cost breakdown per submission for each provider
- Monthly cost estimates at different usage levels
- Current cost controls and prevention strategies
- Provider comparison

**Supported Providers:**

- **OpenAI (GPT-4o-mini)**: ~$0.0025 per submission - Cost-effective, excellent quality
- **Groq (Llama 3.3 70B Versatile)**: ~$0.006 per submission - Ultra-fast, excellent quality

**Operational Modes:**

- **🪙 Cheap Mode**: GPT-4o-mini + Modal scale-to-zero → ~$7.60-8.50/month (100 submissions/day)
- **⚡ Turbo Mode**: Llama 3.3 70B + Modal keep-warm → ~$25-40/month (100 submissions/day)

**Key Cost Driver:** LLM API calls (varies by provider and mode)

**Lean Mode Configuration:**

Assessors are configured via `apps/api-worker/src/config/assessors.json`. Defaults optimize for cost:

| Assessor     | Default | Cost Impact | Notes                          |
| ------------ | ------- | ----------- | ------------------------------ |
| AES-DEBERTA  | ✅ ON   | ~$0.0003    | Default high-accuracy scorer   |
| AES-ESSAY    | ❌ OFF  | ~$0.0001    | Legacy (deprecated)            |
| GEC-SEQ2SEQ  | ✅ ON   | ~$0.0001    | Best GEC (precise diffs, slow) |
| GEC-GECTOR   | ✅ ON   | ~$0.00008   | Fast GEC (~10x faster)         |
| GEC-LT       | ✅ ON   | ~$0.0001    | Typos, mechanics               |
| AES-FEEDBACK | ❌ OFF  | ~$0.0001    | Experimental                   |
| GEC-LLM      | ❌ OFF  | ~$0.002     | Expensive, redundant           |

**Modal Services Cost Breakdown:**

| Service            | GPU  | Keep-Warm | Cost/Invocation | Notes                         |
| ------------------ | ---- | --------- | --------------- | ----------------------------- |
| **modal-deberta**  | A10G | 30s       | ~$0.00030       | DeBERTa-v3 Multi-Head         |
| **modal-essay**    | T4   | 30s       | ~$0.00008       | Legacy Essay Scoring          |
| **modal-feedback** | T4   | 30s       | ~$0.00008       | Feedback model (experimental) |
| **modal-lt**       | CPU  | 30s       | ~$0.00002       | LanguageTool grammar check    |
| **modal-gec**      | A10G | 30s       | ~$0.00015       | Seq2Seq GEC (Flan-T5, slow)   |
| **modal-gector**   | T4   | 30s       | ~$0.00008       | GECToR fast (~10x faster)     |

**GPU Pricing (Modal, as of Q4 2025):**

- T4: ~$0.59/hour (~$0.00016/second)
- A10G: ~$1.10/hour (~$0.00031/second)
- CPU: ~$0.0001/second

**Cold Start vs Warm:**

- Cold start adds ~2-5s for GPU services (billed)
- Keep-warm reduces cold starts but incurs idle cost
- Default config: T4 services keep-warm 30-60s, CPU keep-warm 300s

**Lean Mode (default)**: ~$0.0005/submission (Modal only)  
**Full Mode** (all enabled): ~$0.0025-0.006/submission (with LLM)

See [Evaluation Report](../models/evaluation.md) for assessor performance data.

---

## Cost Breakdown Per Submission

### API Calls Per Submission

Each essay submission triggers **2 required API calls** to your chosen LLM provider (OpenAI or Groq):

**Pricing Basis (as of Q1 2025):**

- **OpenAI GPT-4o-mini:** ~$0.15/1M input, ~$0.60/1M output
- **Groq Llama 3.3 70B:** ~$0.59/1M input, ~$0.79/1M output

1. **Grammar Error Detection** (`getLLMAssessment`)
   - **Input tokens:** ~2,500 avg (Question + Answer + Prompt)
   - **Output tokens:** ~750 avg (JSON error list)
   - **Cost (OpenAI):** ~$0.0008
   - **Cost (Groq):** ~$0.0021
   - **Frequency:** Once per answer submission

2. **Detailed Feedback** (`getCombinedFeedback`)
   - **Input tokens:** ~4,000 avg (Question + Answer + Context + Errors)
   - **Output tokens:** ~450 avg (Structured feedback)
   - **Cost (OpenAI):** ~$0.0009
   - **Cost (Groq):** ~$0.0027
   - **Frequency:** Once per answer submission

3. **Teacher Feedback** (`getTeacherFeedback` / streaming endpoint) - **OPTIONAL**
   - **Purpose:** Provide concise, teacher-style feedback (initial, clues, or detailed explanation)
   - **Input tokens:** ~1,200-1,500 (average observed: ~1,300)
   - **Output tokens:** ~400-800 (average observed: ~750 for explanation)
   - **Cost (OpenAI):** ~$0.0008
   - **Cost (Groq):** ~$0.0014
   - **Frequency:** On-demand (user requests teacher feedback)
   - **Streaming:** The UI uses Server-Sent Events (SSE) streaming for real-time feedback display

### Total Cost Per Submission (Base)

| Provider                 | Input Cost (6.5k tokens) | Output Cost (1.2k tokens) | **Total Base Cost**   |
| :----------------------- | :----------------------- | :------------------------ | :-------------------- |
| **OpenAI (GPT-4o-mini)** | ~$0.0010                 | ~$0.0007                  | **~$0.0017 - 0.0025** |
| **Groq (Llama 3.3 70B)** | ~$0.0038                 | ~$0.0009                  | **~$0.0048 - 0.0060** |

_Note: Ranges account for varying essay lengths (250-500 words) and error counts._

---

## Monthly Cost Estimates

### Submission Volume Scenarios

Based on **rate limit of 10 submissions/minute per IP address** AND **Daily Limit of 100 submissions/day**:

- **Daily Max Per User:** 100 submissions (enforced hard limit)
- **System-Wide Capacity:** Scales with number of concurrent users (unlimited by architecture)

**OpenAI (GPT-4o-mini) Cost Scenarios (Per User/IP):**

| Scenario               | Submissions/Day | Submissions/Month | Cost/Submission | Monthly Cost     |
| ---------------------- | --------------- | ----------------- | --------------- | ---------------- |
| **Low Usage**          | 10              | ~300              | $0.0025         | **~$0.75/month** |
| **Moderate Usage**     | 50              | ~1,500            | $0.0025         | **~$3.75/month** |
| **Heavy Usage**        | 80              | ~2,400            | $0.0025         | **~$6.00/month** |
| **Maximum (per user)** | 100             | ~3,000            | $0.0025         | **~$7.50/month** |

**Groq (Llama 3.3 70B Versatile) Cost Scenarios (Per User/IP):**

| Scenario               | Submissions/Day | Submissions/Month | Cost/Submission | Monthly Cost      |
| ---------------------- | --------------- | ----------------- | --------------- | ----------------- |
| **Low Usage**          | 10              | ~300              | $0.006          | **~$1.80/month**  |
| **Moderate Usage**     | 50              | ~1,500            | $0.006          | **~$9.00/month**  |
| **Heavy Usage**        | 80              | ~2,400            | $0.006          | **~$14.40/month** |
| **Maximum (per user)** | 100             | ~3,000            | $0.006          | **~$18.00/month** |

### Realistic Usage Estimates

**Educational Context (OpenAI):**

- Small class (20 students): ~20-40 submissions/day = **~$1.50-3/month**
- Medium class (100 students): ~100-200 submissions/day = **~$7.50-15/month**
- Large institution (1,000 students): ~1,000-2,000 submissions/day = **~$75-150/month**

**Educational Context (Groq):**

- Small class (20 students): ~20-40 submissions/day = **~$3.60-7.20/month**
- Medium class (100 students): ~100-200 submissions/day = **~$18-36/month**
- Large institution (1,000 students): ~1,000-2,000 submissions/day = **~$180-360/month**

**Note:** Two-tier rate limiting protects against abuse:

1. **Burst Limit:** 10 submissions/min (per IP) to prevent rapid-fire scripts.
2. **Daily Limit:** 100 submissions/day (per IP) to cap total daily cost.

**Maximum theoretical liability per IP:** ~$0.25/day (OpenAI) or ~$0.60/day (Groq).

---

## Costs Without LLM API (If Disabled)

If LLM API is disabled (both OpenAI and Groq), Writeo can still function with reduced features. Here's the cost breakdown:

### What Works Without LLM API

✅ **Still Available:**

- Essay scoring (Modal service)
- Grammar checking (LanguageTool via Modal)
- Relevance checking (Cloudflare Workers AI)
- Basic assessment results
- CEFR level mapping
- Error detection and highlighting

❌ **Not Available:**

- AI-powered grammar error detection (`getLLMAssessment`)
- Detailed AI feedback (`getCombinedFeedback`)
- Teacher feedback (`getTeacherFeedback`)
- Context-aware feedback and suggestions

### Cost Breakdown (Without LLM API)

**Infrastructure Costs Only:**

| Service                         | Free Tier         | Monthly Cost          |
| ------------------------------- | ----------------- | --------------------- |
| **Cloudflare Workers**          | 100k requests/day | $0.00                 |
| **Cloudflare Workers AI**       | 10k requests/day  | $0.00                 |
| **Cloudflare R2 Storage**       | 10 GB free        | $0.00                 |
| **Cloudflare KV Storage**       | 100 MB free       | $0.00                 |
| **Modal Essay Scoring Service** | Pay-per-use       | ~$0.10-1.00/month     |
| **Modal LanguageTool**          | Pay-per-use       | ~$0.01-0.10/month     |
| **Total Infrastructure**        | -                 | **~$0.11-1.10/month** |

**Monthly Cost Examples (Without LLM API):**

| Usage Level | Submissions/Day | Monthly Cost      |
| ----------- | --------------- | ----------------- |
| Low         | 10              | ~$0.11-1.10/month |
| Moderate    | 50              | ~$0.11-1.10/month |
| High        | 80              | ~$0.11-1.10/month |
| Maximum     | 100             | ~$0.11-1.10/month |

**Key Points:**

- Infrastructure costs are **fixed** regardless of submission volume (within free tier limits)
- Modal services are pay-per-use but very cheap (~$0.0001-0.001 per submission)
- **Total cost: ~$0.11-1.10/month** (essentially free tier)
- No variable costs based on submission volume

### How to Disable LLM API

**Option 1: Environment Variable (Recommended)**

Add to your Cloudflare Worker secrets:

```bash
DISABLE_AI_FEEDBACK=true
```

Then modify `apps/api-worker/src/services/submission-processor.ts` to check this flag:

```typescript
if (c.env.DISABLE_AI_FEEDBACK === "true") {
  // Skip AI feedback calls
  // Return results without AI feedback
}
```

**Option 2: Remove API Keys**

Simply don't set `OPENAI_API_KEY` or `GROQ_API_KEY` secrets. The system will fail gracefully (or you can add error handling to skip AI features).

**Option 3: Set Empty/Mock Key**

Set `OPENAI_API_KEY=MOCK` or `GROQ_API_KEY=MOCK` to use mock responses (no real API calls, but also no real AI feedback).

### Feature Comparison

| Feature                         | With LLM API (OpenAI or Groq) | Without LLM API |
| ------------------------------- | ----------------------------- | --------------- |
| Essay Scoring                   | ✅                            | ✅              |
| Grammar Checking (LanguageTool) | ✅                            | ✅              |
| Relevance Checking              | ✅                            | ✅              |
| AI Grammar Detection            | ✅                            | ❌              |
| Detailed AI Feedback            | ✅                            | ❌              |
| Teacher Feedback                | ✅                            | ❌              |
| Context-aware Suggestions       | ✅                            | ❌              |
| **Cost per submission**         | ~$0.006                       | ~$0.0001-0.001  |
| **Monthly cost (100/day)**      | ~$18                          | ~$0.11-1.10     |

**Recommendation:** For production use, both OpenAI and Groq provide excellent value for comprehensive AI feedback. Choose OpenAI for cost efficiency (~$0.0025/submission) or Groq for ultra-fast responses (~$0.006/submission). If cost is a concern, the system can operate without LLM API using only LanguageTool for grammar checking.

---

## Current Cost Controls

### 1. Rate Limiting ✅

**Implementation:** `apps/api-worker/src/middleware/rate-limit.ts`

- **Burst Limit:** 10 submissions/minute per IP address (prevents rapid-fire abuse)
- **Daily Limit:** 100 submissions/day per IP address (hard cost cap)
- **General requests:** 30/minute per IP address
- **Results requests:** 60/minute per IP address

**Cost Protection:**

- **Maximum Daily Liability (per IP):**
  - OpenAI: 100 \* $0.0025 = **$0.25/day**
  - Groq: 100 \* $0.006 = **$0.60/day**
- **Maximum Monthly Liability (per IP):**
  - OpenAI: ~$7.50/month
  - Groq: ~$18.00/month
- Prevents single user/IP from causing runaway costs

### 2. Word Count Limits ✅

**Implementation:** `apps/web/app/lib/actions.ts` (lines 273-289)

- **Minimum:** 250 words (prevents very short submissions)
- **Maximum:** 500 words (soft cap - warns but allows)
- **AI Processing Limit:** 15,000 characters (~2,500 words) - text truncated for AI calls

**Cost Protection:**

- Limits input token usage per submission
- Prevents extremely long essays from driving up costs
- Average essay length: ~300-400 words = predictable token usage

### 3. Token Limits ✅

**Implementation:** `apps/api-worker/src/services/openai.ts` and `apps/api-worker/src/services/llm.ts`

- Grammar check: Max 2,500 tokens (reduced from 4,000)
- Detailed feedback: Max 500 tokens
- Teacher feedback: Max 150 (initial/clues) or 800 (explanation)

**Cost Protection:**

- Prevents excessive output token generation
- Reduces cost per API call by ~25-40%

### 4. Text Truncation ✅

**Implementation:** `apps/api-worker/src/services/feedback.ts` (lines 56-70)

- Essays truncated to 15,000 characters for AI processing
- Questions truncated to 500 characters
- Error context limited to top 10 errors

**Cost Protection:**

- Limits input token usage
- Prevents very long essays from causing excessive costs

### 5. Mocking System ✅

**Implementation:** `apps/api-worker/src/services/openai.ts` + `llm.mock.ts`

- Tests automatically use mocks (no real API calls)
- Development can use mocks to avoid costs
- Prevents test runs from incurring costs

---

## Cost Guardrails & Prevention Strategies

### Already Implemented ✅

1. **Rate Limiting** - Prevents runaway costs from single user/IP
2. **Word Count Limits** - Controls input size and token usage
3. **Token Limits** - Caps output token generation
4. **Text Truncation** - Limits input token usage
5. **Mocking System** - Prevents test costs

### Request Size Limits ✅

**Already Implemented:** 1MB request body limit

- Prevents extremely large submissions
- Protects against abuse

---

## Cost Optimization Strategies

### Already Implemented ✅

1. **Reduced Token Limits** - Grammar check: 2,500 (was 4,000), saves ~$0.001-0.002 per request
2. **Combined Feedback Calls** - Single LLM call for detailed + teacher feedback (50% fewer calls)
3. **Text Truncation** - Limits input tokens
4. **Error Context Limits** - Only top 10 errors included
5. **Parallel Processing** - Faster responses, but doesn't reduce costs

---

## Mocking System (Test Cost Control)

To keep costs under control during testing, Writeo includes a mocking system that prevents real API calls in test environments.

### How It Works

The mocking system automatically detects test environments and returns deterministic mock responses instead of making real API calls.

**Detection Methods:**

1. **Environment variable:** `USE_MOCK_LLM=true` (unified flag for all LLM APIs)
2. **API key pattern:** API key is `"MOCK"` or starts with `"test_"` (for worker environments)

### Mock Responses

Mock responses are designed to:

- Return realistic data structures matching real API responses
- Include common error patterns (tense errors, subject-verb agreement)
- Provide sample feedback that tests can validate
- Avoid API costs during test runs

### Enabling Mocks

**Automatic (Default):**

- Pre-push hook sets `USE_MOCK_LLM=true` when starting the worker
- Vitest config sets `USE_MOCK_LLM=true` by default
- Tests automatically use mocks unless explicitly disabled

**Manual Control:**

```bash
# Use mocks (default)
USE_MOCK_LLM=true npm test

# Use real API (for integration tests)
USE_MOCK_LLM=false npm test
```

**Worker Environment:**

```bash
# Start worker with mock API key
OPENAI_API_KEY=MOCK npm run dev

# Or use test_ prefix
OPENAI_API_KEY=test_anything npm run dev
```

### Troubleshooting Mock Issues

**Tests Making Real API Calls:**

- Verify `USE_MOCK_LLM=true` is set in test environment
- Check that worker is started with `USE_MOCK_LLM=true` or API key is `MOCK`
- Ensure `.dev.vars` doesn't override with real API key during tests

**Mock Responses Not Working:**

- Check `llm.mock.ts` for correct response format
- Verify mock detection logic in `openai.ts`
- Ensure mock module is imported correctly

---

## Cost Monitoring

### Current Monitoring

- **Development:** Token usage logged to console (if `NODE_ENV=development`)
  ```
  [OpenAI API] Tokens used: 3245 prompt + 487 completion = 3732 total
  ```
- **Production:** No automatic cost tracking
- **OpenAI Dashboard:** Manual checking at https://platform.openai.com/usage
- **Cloudflare Logs:** Review Worker logs for error patterns and usage

---

## Emergency Cost Controls

If costs spike unexpectedly:

### Immediate Actions

1. **Reduce Rate Limits** - Lower from 10/min to 5/min or 1/min
2. **Enable Maintenance Mode** - Return 503 for all submissions
3. **Disable AI Feedback** - Return results without AI feedback (fallback mode)
4. **Revoke API Key** - Temporarily disable OpenAI API key

### Code Changes Required

**Reduce Rate Limit:**

```typescript
// apps/api-worker/src/middleware/rate-limit.ts
maxRequests = isTest ? 500 : 5; // Changed from 10 to 5
```

**Maintenance Mode:**

```typescript
// Add environment variable: MAINTENANCE_MODE=true
if (process.env.MAINTENANCE_MODE === "true") {
  return errorResponse(503, "Service temporarily unavailable", c);
}
```

**Disable AI Feedback:**

```typescript
// Add environment variable: DISABLE_AI_FEEDBACK=true
if (process.env.DISABLE_AI_FEEDBACK === "true") {
  // Skip AI feedback calls, return results without AI feedback
}
```

---

## Cost Comparison: Scenarios

### Scenario 1: Current Implementation (With OpenAI, Optimized)

- Token limits reduced
- Text truncation enabled
- Rate limiting active
- **Cost:** ~$0.0017-0.0025 per submission (OpenAI) / ~$0.0048-0.0060 (Groq)
- **Monthly (100/day):** ~$7.50 (OpenAI) / ~$18 (Groq)

### Scenario 2: Without Optimizations (Hypothetical)

- No token limits
- No text truncation
- No rate limiting
- **Estimated Cost:** ~$0.05-0.10 per submission
- **Monthly (100/day):** ~$150-300

**Savings:** ~60-80% reduction through optimizations

### Scenario 3: Without OpenAI API

- No AI feedback features
- Only LanguageTool + Essay Scoring
- **Cost:** ~$0.0001-0.001 per submission
- **Monthly (100/day):** ~$0.11-1.10

**Savings:** ~99% cost reduction, but loses AI feedback features

---

## Cost Transparency

### For Users

- **No cost to users** - Service is free
- **Rate limits disclosed** - Terms of Service mentions 10 submissions/min
- **Word limits disclosed** - UI shows 250-500 word limits

### For Operators

- **Cost tracking** - Manual via OpenAI dashboard
- **Rate limit monitoring** - Via Cloudflare Workers logs
- **Usage patterns** - Via Cloudflare Analytics

---

## Best Practices

1. **Always use mocks in tests** - Never commit tests that use real API keys
2. **Monitor token usage** - Check logs regularly for unexpected spikes
3. **Review prompts** - Keep prompts concise but effective
4. **Test locally first** - Use mocks during development, test with real API only when needed
5. **Set budgets** - Configure spending limits in OpenAI dashboard
6. **Track daily costs** - Monitor OpenAI dashboard daily for unexpected usage

## References

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [OpenAI API Pricing](https://openai.com/pricing) - Check for current pricing
- [Token Usage Guide](https://platform.openai.com/docs/guides/tokens)
- [System Architecture](../architecture/overview.md) - System architecture and cost estimates
- [Operations Guide](monitoring.md) - Operations guide with cost information

---
