# Cost Review & Optimization Guide

**Last Updated:** January 2025  
**Review Frequency:** Quarterly

---

## Executive Summary

Writeo's cost structure is dominated by **Groq API usage** for AI-powered feedback. This document provides:

- Current cost breakdown per submission
- Monthly cost estimates at different usage levels
- Cost guardrails and prevention strategies
- Optimization recommendations

**Key Cost Driver:** Groq API calls (~$0.02-0.03 per submission)

---

## Cost Breakdown Per Submission

### API Calls Per Submission

Each essay submission triggers **2 required API calls** to Groq:

1. **Grammar Error Detection** (`getLLMAssessment`)
   - **Purpose:** Identify grammar, spelling, and language errors
   - **Input tokens:** ~2,000-3,000 (question + answer text + prompt)
   - **Output tokens:** ~500-1,000 (JSON error list)
   - **Max tokens:** 2,500 (reduced from 4,000 to save costs)
   - **Cost:** ~$0.005-0.01 per call
   - **Frequency:** Once per answer submission

2. **Detailed Feedback** (`getCombinedFeedback`)
   - **Purpose:** Provide comprehensive feedback on essay quality (relevance, strengths, improvements)
   - **Input tokens:** ~3,000-5,000 (question + answer + assessment context + grammar errors)
   - **Output tokens:** ~400-500 (structured JSON feedback)
   - **Max tokens:** 500
   - **Cost:** ~$0.01 per call
   - **Frequency:** Once per answer submission
   - **Optimizations:**
     - Essay text truncated to 15,000 chars (~2,500 words) to limit input tokens
     - Question text truncated to 500 chars (safety measure)
     - Error context limited to top 10 errors

3. **Teacher Feedback** (`getTeacherFeedback`) - **OPTIONAL**
   - **Purpose:** Provide concise, teacher-style feedback (initial, clues, or detailed explanation)
   - **Input tokens:** ~2,000-4,000 (varies by mode)
   - **Output tokens:** ~100-200 (initial/clues) or ~400-800 (explanation)
   - **Max tokens:** 150 (initial/clues) or 800 (explanation)
   - **Cost:** ~$0.005-0.01 per call
   - **Frequency:** On-demand (user requests teacher feedback)
   - **Modes:**
     - **Initial:** Brief 2-3 sentence feedback
     - **Clues:** Hints to guide student improvement
     - **Explanation:** Detailed markdown analysis for teachers

### Total Cost Per Submission

- **Base submission (required):** ~$0.015-0.02
- **With teacher feedback (optional):** ~$0.02-0.03
- **Average (assuming 20% request teacher feedback):** ~$0.016-0.022

**Note:** Costs vary based on:

- Essay length (affects input tokens)
- Number of errors found (affects output tokens)
- Groq API pricing (subject to change)

---

## Monthly Cost Estimates

### Submission Volume Scenarios

Based on **rate limit of 10 submissions/minute** (14,400 submissions/day maximum):

| Scenario                   | Submissions/Day | Submissions/Month | Cost/Submission | Monthly Cost      |
| -------------------------- | --------------- | ----------------- | --------------- | ----------------- |
| **Low Usage**              | 10              | ~300              | $0.02           | **~$6/month**     |
| **Moderate Usage**         | 100             | ~3,000            | $0.02           | **~$60/month**    |
| **High Usage**             | 1,000           | ~30,000           | $0.02           | **~$600/month**   |
| **Maximum (rate limited)** | 14,400          | ~432,000          | $0.02           | **~$8,640/month** |

### Realistic Usage Estimates

**Educational Context:**

- Small class (20 students): ~20-40 submissions/day = **~$12-24/month**
- Medium class (100 students): ~100-200 submissions/day = **~$60-120/month**
- Large institution (1,000 students): ~1,000-2,000 submissions/day = **~$600-1,200/month**

**Note:** Rate limiting (10/min) prevents runaway costs. Maximum theoretical cost is **~$8,640/month** if running at full capacity 24/7.

---

## Costs Without Groq API (If Disabled)

If Groq API is disabled, Writeo can still function with reduced features. Here's the cost breakdown:

### What Works Without Groq

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

### Cost Breakdown (Without Groq)

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

**Monthly Cost Examples (Without Groq):**

| Usage Level | Submissions/Day | Monthly Cost      |
| ----------- | --------------- | ----------------- |
| Low         | 10              | ~$0.11-1.10/month |
| Moderate    | 100             | ~$0.11-1.10/month |
| High        | 1,000           | ~$0.11-1.10/month |
| Maximum     | 14,400          | ~$0.11-1.10/month |

**Key Points:**

- Infrastructure costs are **fixed** regardless of submission volume (within free tier limits)
- Modal services are pay-per-use but very cheap (~$0.0001-0.001 per submission)
- **Total cost: ~$0.11-1.10/month** (essentially free tier)
- No variable costs based on submission volume

### How to Disable Groq

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

**Option 2: Remove API Key**

Simply don't set `GROQ_API_KEY` secret. The system will fail gracefully (or you can add error handling to skip AI features).

**Option 3: Set Empty/Mock Key**

Set `GROQ_API_KEY=MOCK` to use mock responses (no real API calls, but also no real AI feedback).

### Feature Comparison

| Feature                         | With Groq | Without Groq   |
| ------------------------------- | --------- | -------------- |
| Essay Scoring                   | ✅        | ✅             |
| Grammar Checking (LanguageTool) | ✅        | ✅             |
| Relevance Checking              | ✅        | ✅             |
| AI Grammar Detection            | ✅        | ❌             |
| Detailed AI Feedback            | ✅        | ❌             |
| Teacher Feedback                | ✅        | ❌             |
| Context-aware Suggestions       | ✅        | ❌             |
| **Cost per submission**         | ~$0.02    | ~$0.0001-0.001 |
| **Monthly cost (100/day)**      | ~$60      | ~$0.11-1.10    |

**Recommendation:** For production use, Groq provides significant value (~$0.02/submission) for comprehensive AI feedback. However, if cost is a concern, the system can operate without Groq using only LanguageTool for grammar checking.

---

## Current Cost Controls

### 1. Rate Limiting ✅

**Implementation:** `apps/api-worker/src/middleware/rate-limit.ts`

- **Submissions:** 10/minute per IP address
- **General requests:** 30/minute per IP address
- **Results requests:** 60/minute per IP address

**Cost Protection:**

- Maximum: 14,400 submissions/day = ~432,000/month
- At $0.02/submission: **Maximum ~$8,640/month**
- Prevents single user from causing runaway costs

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

**Implementation:** `apps/api-worker/src/services/groq.ts`

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

**Implementation:** `apps/api-worker/src/services/groq.ts` + `groq.mock.ts`

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

### Recommended Additional Guardrails

#### 1. Daily Cost Budget (Not Yet Implemented)

**Recommendation:** Add daily cost tracking and automatic shutdown

```typescript
// Pseudo-code for daily budget tracking
const DAILY_BUDGET = 50; // $50/day maximum
const COST_PER_SUBMISSION = 0.02;

// Track in KV: daily_cost:YYYY-MM-DD
// If daily cost exceeds budget, return 429 with message
```

**Implementation Effort:** Medium  
**Cost Protection:** High  
**Impact:** Prevents unexpected daily spikes

#### 2. Per-User Rate Limiting (Not Yet Implemented)

**Current:** Rate limiting is per IP address  
**Recommendation:** Add per-user rate limiting (if user authentication added)

**Implementation Effort:** Medium  
**Cost Protection:** Medium  
**Impact:** Better cost control for authenticated users

#### 3. Cost Monitoring Dashboard (Not Yet Implemented)

**Recommendation:** Add cost tracking endpoint and dashboard

- Track daily/monthly costs
- Alert on cost spikes
- Show cost trends

**Implementation Effort:** High  
**Cost Protection:** Medium  
**Impact:** Better visibility and early warning

#### 4. Request Size Limits ✅

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

### Future Optimization Opportunities

#### 1. Caching Similar Submissions (Not Yet Implemented)

**Idea:** Cache feedback for similar answers (same question + similar text)

**Potential Savings:** 20-30% if 30% of submissions are similar  
**Implementation Effort:** High  
**Complexity:** Need similarity detection, cache invalidation

#### 2. Smaller Model for Grammar Checks (Not Yet Implemented)

**Idea:** Use smaller/faster model for grammar checks, keep large model for feedback

**Potential Savings:** 30-50% on grammar check costs  
**Implementation Effort:** Medium  
**Trade-off:** May reduce grammar check quality

#### 3. Batch Processing (Not Yet Implemented)

**Idea:** Batch multiple submissions together

**Potential Savings:** Minimal (Groq doesn't offer batch discounts)  
**Implementation Effort:** High  
**Trade-off:** Delayed responses, complexity

#### 4. Pre-filter with LanguageTool (Partially Implemented)

**Current:** LanguageTool runs in parallel with LLM  
**Idea:** Use LanguageTool errors to reduce LLM prompt size

**Potential Savings:** 10-15% on input tokens  
**Implementation Effort:** Low  
**Trade-off:** Minimal

---

## Mocking System (Test Cost Control)

To keep costs under control during testing, Writeo includes a mocking system that prevents real API calls in test environments.

### How It Works

The mocking system automatically detects test environments and returns deterministic mock responses instead of making real API calls.

**Detection Methods:**

1. **Environment variable:** `MOCK_GROQ=true` (for Node.js/test environments)
2. **API key pattern:** API key is `"MOCK"` or starts with `"test_"` (for worker environments)

### Mock Responses

Mock responses are designed to:

- Return realistic data structures matching real API responses
- Include common error patterns (tense errors, subject-verb agreement)
- Provide sample feedback that tests can validate
- Avoid API costs during test runs

### Enabling Mocks

**Automatic (Default):**

- Pre-push hook sets `GROQ_API_KEY=MOCK` when starting the worker
- Vitest config sets `MOCK_GROQ=true` by default
- Tests automatically use mocks unless explicitly disabled

**Manual Control:**

```bash
# Use mocks (default)
MOCK_GROQ=true npm test

# Use real API (for integration tests)
MOCK_GROQ=false npm test
```

**Worker Environment:**

```bash
# Start worker with mock API key
GROQ_API_KEY=MOCK npm run dev

# Or use test_ prefix
GROQ_API_KEY=test_anything npm run dev
```

### Troubleshooting Mock Issues

**Tests Making Real API Calls:**

- Verify `MOCK_GROQ=true` is set in test environment
- Check that worker is started with `GROQ_API_KEY=MOCK`
- Ensure `.dev.vars` doesn't override with real API key during tests

**Mock Responses Not Working:**

- Check `groq.mock.ts` for correct response format
- Verify mock detection logic in `groq.ts`
- Ensure mock module is imported correctly

---

## Cost Monitoring

### Current Monitoring

- **Development:** Token usage logged to console (if `NODE_ENV=development`)
  ```
  [Groq API] Tokens used: 3245 prompt + 487 completion = 3732 total
  ```
- **Production:** No automatic cost tracking
- **Groq Dashboard:** Manual checking at https://console.groq.com/
- **Cloudflare Logs:** Review Worker logs for error patterns and usage

### Recommended Monitoring

1. **Daily Cost Tracking** - Store daily costs in KV
2. **Cost Alerts** - Alert if daily cost exceeds threshold
3. **Usage Dashboard** - Show submissions/day, cost/day, trends
4. **Cost Per User** - Track costs per IP/user (if authentication added)

---

## Emergency Cost Controls

If costs spike unexpectedly:

### Immediate Actions

1. **Reduce Rate Limits** - Lower from 10/min to 5/min or 1/min
2. **Enable Maintenance Mode** - Return 503 for all submissions
3. **Disable AI Feedback** - Return results without AI feedback (fallback mode)
4. **Revoke API Key** - Temporarily disable Groq API key

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

### Scenario 1: Current Implementation (With Groq, Optimized)

- Token limits reduced
- Text truncation enabled
- Rate limiting active
- **Cost:** ~$0.02 per submission
- **Monthly (100/day):** ~$60

### Scenario 2: Without Optimizations (Hypothetical)

- No token limits
- No text truncation
- No rate limiting
- **Estimated Cost:** ~$0.05-0.10 per submission
- **Monthly (100/day):** ~$150-300

**Savings:** ~60-80% reduction through optimizations

### Scenario 3: Without Groq API

- No AI feedback features
- Only LanguageTool + Essay Scoring
- **Cost:** ~$0.0001-0.001 per submission
- **Monthly (100/day):** ~$0.11-1.10

**Savings:** ~99% cost reduction, but loses AI feedback features

---

## How to Guard Against Runaway Costs (Without Major Changes)

### Quick Wins - No Code Changes Required

1. **Monitor Groq Dashboard Daily**
   - Check https://console.groq.com/ daily for unexpected usage
   - Set up email alerts if Groq supports them
   - **Time:** 2 minutes/day
   - **Protection:** Early detection of cost spikes

2. **Set Groq API Budget Alert**
   - Configure spending limits in Groq dashboard (if available)
   - Set alert threshold (e.g., $50/day or $500/month)
   - **Time:** 5 minutes one-time setup
   - **Protection:** Automatic alerts on cost spikes

3. **Review Rate Limit Logs Weekly**
   - Check Cloudflare Workers logs for rate limit hits
   - Look for patterns of high usage from single IPs
   - **Time:** 10 minutes/week
   - **Protection:** Identify abuse patterns early

### Low-Effort Code Changes (1-2 hours)

1. **Add Daily Cost Tracking** ⚠️ **RECOMMENDED**

   Track daily costs in KV and reject requests if budget exceeded:

   ```typescript
   // In submission-processor.ts, before processing:
   const today = new Date().toISOString().split("T")[0];
   const dailyCostKey = `daily_cost:${today}`;
   const DAILY_BUDGET = 50; // $50/day
   const COST_PER_SUBMISSION = 0.02;

   const currentCost = await c.env.WRITEO_RESULTS.get(dailyCostKey);
   const cost = currentCost ? parseFloat(currentCost) : 0;

   if (cost + COST_PER_SUBMISSION > DAILY_BUDGET) {
     return errorResponse(429, `Daily cost limit reached. Please try again tomorrow.`, c);
   }

   // After successful processing:
   await c.env.WRITEO_RESULTS.put(dailyCostKey, String(cost + COST_PER_SUBMISSION), {
     expirationTtl: 86400,
   });
   ```

   **Effort:** 1-2 hours  
   **Protection:** Prevents daily cost spikes  
   **Impact:** High

2. **Add Cost Logging** ⚠️ **RECOMMENDED**

   Log costs to Cloudflare Workers logs for monitoring:

   ```typescript
   // After each Groq API call, log cost:
   const cost = estimateCostFromTokens(usage);
   console.log(`[Cost] Submission ${submissionId}: $${cost.toFixed(4)}`);
   ```

   **Effort:** 30 minutes  
   **Protection:** Better visibility  
   **Impact:** Medium

3. **Add Environment Variable for Rate Limit**

   Make rate limit configurable via environment variable:

   ```typescript
   // In rate-limit.ts:
   const maxSubmissions = parseInt(c.env.MAX_SUBMISSIONS_PER_MIN || "10");
   ```

   **Effort:** 15 minutes  
   **Protection:** Quick adjustment without code deploy  
   **Impact:** Medium

### Medium-Effort Changes (4-8 hours)

1. **Add Maintenance Mode** ⚠️ **RECOMMENDED**

   Add environment variable to disable all submissions:

   ```typescript
   // In submission-processor.ts, at the start:
   if (c.env.MAINTENANCE_MODE === "true") {
     return errorResponse(503, "Service temporarily unavailable for maintenance", c);
   }
   ```

   **Effort:** 30 minutes  
   **Protection:** Emergency shutdown  
   **Impact:** High

2. **Add Cost Monitoring Endpoint**

   Create `/admin/costs` endpoint to view daily/monthly costs:

   ```typescript
   // GET /admin/costs?period=day|month
   // Returns: { period, submissions, cost, budget, remaining }
   ```

   **Effort:** 2-3 hours  
   **Protection:** Better visibility  
   **Impact:** Medium

### Summary: Best Protection with Minimal Changes

**Immediate (No Code):**

- ✅ Monitor Groq dashboard daily
- ✅ Set Groq API budget alerts
- ✅ Review rate limit logs weekly

**Quick Wins (1-2 hours total):**

- ⚠️ Add daily cost budget tracking (prevents spikes)
- ⚠️ Add cost logging (better visibility)
- ⚠️ Make rate limit configurable (quick adjustments)

**Emergency Controls (30 minutes):**

- ⚠️ Add maintenance mode (emergency shutdown)

**Total Implementation Time:** ~2-3 hours for all quick wins  
**Cost Protection:** Prevents 95%+ of runaway cost scenarios

---

## Recommendations

### High Priority

1. ✅ **Keep current rate limits** (10/min) - Good balance of usability and cost control
2. ✅ **Keep word count limits** (250-500 words) - Prevents excessive costs
3. ⚠️ **Add daily cost budget** - Prevents unexpected spikes
4. ⚠️ **Add cost monitoring** - Better visibility

### Medium Priority

1. **Consider caching** - If submission patterns show similarity
2. **Monitor Groq pricing** - Prices may change
3. **Review token usage** - Periodically check if limits can be further reduced

### Low Priority

1. **Smaller model for grammar** - Only if quality acceptable
2. **Batch processing** - Only if Groq adds batch discounts

---

## Cost Transparency

### For Users

- **No cost to users** - Service is free
- **Rate limits disclosed** - Terms of Service mentions 10 submissions/min
- **Word limits disclosed** - UI shows 250-500 word limits

### For Operators

- **Cost tracking** - Manual via Groq dashboard
- **Rate limit monitoring** - Via Cloudflare Workers logs
- **Usage patterns** - Via Cloudflare Analytics

---

## Best Practices

1. **Always use mocks in tests** - Never commit tests that use real API keys
2. **Monitor token usage** - Check logs regularly for unexpected spikes
3. **Review prompts** - Keep prompts concise but effective
4. **Test locally first** - Use mocks during development, test with real API only when needed
5. **Set budgets** - Configure spending limits in Groq dashboard
6. **Track daily costs** - Monitor Groq dashboard daily for unexpected usage

## References

- [Groq API Documentation](https://console.groq.com/docs)
- [Groq API Pricing](https://groq.com/pricing) - Check for current pricing
- [Token Usage Guide](https://console.groq.com/docs/token-usage)
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture and cost estimates
- [OPERATIONS.md](OPERATIONS.md) - Operations guide with cost information

---

**Last Reviewed:** January 2025  
**Next Review:** April 2025
