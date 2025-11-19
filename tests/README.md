# Test Suite

Quick reference for running and writing tests. For complete test plan, see [docs/TEST_PLAN.md](../docs/TEST_PLAN.md).

## Quick Start

```bash
npm test              # API tests (Vitest)
npm run test:e2e      # E2E tests (Playwright)
npm run test:all      # Both
```

## Test Structure

- **API Tests** (`api.test.ts`) - Integration tests for API endpoints
- **E2E Tests** (`e2e/*.spec.ts`) - Browser tests for user flows

## Configuration

Create `.env.local` in project root:

```bash
API_KEY=your-api-key
API_BASE=http://localhost:8787
PLAYWRIGHT_BASE_URL=http://localhost:3000
```

## Cost Control: Groq API Mocking

**Tests automatically use mocked Groq API responses by default** to avoid API costs. The mocking system:

- ✅ **No API costs** - Tests use deterministic mock responses
- ✅ **Automatic detection** - Enabled when `GROQ_API_KEY=MOCK` or `MOCK_GROQ=true`
- ✅ **Realistic responses** - Mocks return proper data structures for testing

**To use real API (for integration tests only):**

```bash
MOCK_GROQ=false npm test
```

See [docs/GROQ_COST_CONTROL.md](../docs/GROQ_COST_CONTROL.md) for details on token usage and cost optimization.

## Writing Tests

**API Tests:**

```typescript
test.concurrent("test name", async () => {
  const { questionId, answerId, submissionId } = generateIds();
  const { status, json } = await apiRequest("PUT", `/text/submissions/${submissionId}`, {...});
  expect(status).toBe(200);
});
```

**E2E Tests:**

```typescript
test("description", async ({ writePage, page }) => {
  await writePage.goto("1");
  await writePage.typeEssay("text");
  await writePage.clickSubmit();
});
```

## Test Data

Standard test essays available via `helpers.ts`:

- `getTestEssay("withErrors")` - Essay with grammar errors
- `generateValidEssay()` - 250-500 word essay

## References

- [docs/TEST_PLAN.md](../docs/TEST_PLAN.md) - Complete test plan
- [scripts/README.md](../scripts/README.md) - Scripts documentation
