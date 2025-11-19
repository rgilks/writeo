# Groq API Cost Control

**Last Updated:** January 2025

---

## Overview

Writeo uses Groq API for AI-powered feedback and grammar assessment. To keep costs under control, especially during testing, we've implemented a mocking system that allows tests to run without making real API calls.

---

## Token Usage Patterns

### 1. Grammar Error Detection (`getLLMAssessment`)

**Purpose:** Identify grammar, spelling, and language errors in student essays

**Token Usage:**

- **Input tokens:** ~2,000-3,000 (question + answer text + prompt)
- **Output tokens:** ~500-1,000 (JSON error list)
- **Max tokens:** 2,500 (reduced from 4,000 to save costs)

**Cost:** ~$0.005-0.01 per request

**Frequency:** Once per answer submission

---

### 2. Detailed Feedback (`getCombinedFeedback`)

**Purpose:** Provide comprehensive feedback on essay quality (relevance, strengths, improvements)

**Token Usage:**

- **Input tokens:** ~3,000-5,000 (question + answer + assessment context + grammar errors)
- **Output tokens:** ~400-500 (structured JSON feedback)
- **Max tokens:** 500

**Cost:** ~$0.01 per request

**Frequency:** Once per answer submission

**Optimizations:**

- Essay text truncated to 15,000 chars (~2,500 words) to limit input tokens
- Question text truncated to 500 chars (safety measure)
- Error context limited to top 10 errors

---

### 3. Teacher Feedback (`getTeacherFeedback`)

**Purpose:** Provide concise, teacher-style feedback (initial, clues, or detailed explanation)

**Token Usage:**

- **Input tokens:** ~2,000-4,000 (varies by mode)
- **Output tokens:**
  - Initial/Clues: ~100-200 tokens (max 150)
  - Explanation: ~400-800 tokens (max 800)
- **Max tokens:** 150 (initial/clues) or 800 (explanation)

**Cost:** ~$0.005-0.01 per request

**Frequency:** On-demand (user requests teacher feedback)

**Modes:**

- **Initial:** Brief 2-3 sentence feedback
- **Clues:** Hints to guide student improvement
- **Explanation:** Detailed markdown analysis for teachers

---

## Cost Estimates

### Per Submission (Full Assessment)

Assuming one answer submission with all features enabled:

1. Grammar check: ~$0.01
2. Detailed feedback: ~$0.01
3. Teacher feedback (if requested): ~$0.01

**Total per submission:** ~$0.02-0.03

### Monthly Estimates

- **100 submissions/day:** ~$60-90/month
- **1,000 submissions/day:** ~$600-900/month
- **10,000 submissions/day:** ~$6,000-9,000/month

**Note:** Groq pricing varies by model. Current estimates use `llama-3.3-70b-versatile` pricing.

---

## Mocking System

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

### Enabling Mocks in Tests

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

---

## Cost Optimization Strategies

### Already Implemented

1. ✅ **Token Limits:** Reduced max tokens for grammar checks (2,500) and teacher feedback (150/800)
2. ✅ **Text Truncation:** Essays truncated to 15,000 chars to limit input tokens
3. ✅ **Error Context Limits:** Only top 10 errors included in feedback prompts
4. ✅ **Mocking in Tests:** All tests use mocks by default
5. ✅ **Token Usage Logging:** Development mode logs token usage for monitoring

### Future Optimizations

1. **Caching:** Cache feedback for similar answers (same question + similar text)
2. **Batch Processing:** Combine multiple requests when possible
3. **Model Selection:** Use smaller models for simpler tasks (if quality acceptable)
4. **Rate Limiting:** Implement client-side rate limiting to prevent excessive usage
5. **Cost Monitoring:** Add cost tracking and alerts for unexpected usage spikes

---

## Monitoring Token Usage

### Development Mode

When `NODE_ENV=development`, token usage is logged to console:

```
[Groq API] Tokens used: 3245 prompt + 487 completion = 3732 total
```

### Production Monitoring

To monitor costs in production:

1. Check Groq dashboard: https://console.groq.com/
2. Review Cloudflare Worker logs for error patterns
3. Monitor API response times (may indicate token usage)

---

## Best Practices

1. **Always use mocks in tests** - Never commit tests that use real API keys
2. **Monitor token usage** - Check logs regularly for unexpected spikes
3. **Review prompts** - Keep prompts concise but effective
4. **Test locally first** - Use mocks during development, test with real API only when needed
5. **Set budgets** - Configure spending limits in Groq dashboard

---

## Troubleshooting

### Tests Making Real API Calls

**Problem:** Tests are calling real Groq API instead of using mocks

**Solutions:**

1. Verify `MOCK_GROQ=true` is set in test environment
2. Check that worker is started with `GROQ_API_KEY=MOCK`
3. Ensure `.dev.vars` doesn't override with real API key during tests

### Mock Responses Not Working

**Problem:** Mock responses don't match expected format

**Solutions:**

1. Check `groq.mock.ts` for correct response format
2. Verify mock detection logic in `groq.ts`
3. Ensure mock module is imported correctly

---

## References

- [Groq API Documentation](https://console.groq.com/docs)
- [Groq Pricing](https://groq.com/pricing)
- [Token Usage Guide](https://console.groq.com/docs/token-usage)

---

**Last Updated:** January 2025
