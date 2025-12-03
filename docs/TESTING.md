# Testing Guide

**Test Coverage:** 453 unit tests + 28 API integration tests + 20 E2E tests covering critical functionality

---

## Quick Start

```bash
npm test                    # All Vitest tests (unit + integration)
npm run test:unit           # Unit tests only (fast, no server required)
npm run test:integration   # Integration tests (requires running server)
npm run test:e2e           # E2E tests (Playwright)
npm run test:smoke         # Smoke test (production verification)
npm run test:all           # All tests (unit + integration + E2E)
npm run test:watch         # Watch mode (Vitest)
npm run test:ui            # Vitest UI mode
npm run test:e2e:ui        # Playwright UI mode
npm run test:e2e:debug     # Playwright debug mode
```

---

## Test Structure

### Unit Tests (`npm run test:unit` - Vitest)

**Coverage:** Utilities, middleware, validation, error handling, storage, shared packages

**Test Files (30 files):**

- **API Worker** (11 files): Auth, rate limiting, security, validation, errors, context, HTTP utilities, Zod utilities, fetch-with-timeout, request-id, middleware integration
- **Web App** (13 files): Error handling, validation, progress, storage, submission, text utils, UUID utils, grammar rules, error logger, API client, learner results, error utils
- **Shared Package** (4 files): Validation, text utils, retry logic, type guards
- **Integration** (1 file): Position validation
- **PWA** (1 file): Service worker, manifest, install prompt

**Total:** 453 unit tests covering critical utilities and middleware

**Note:** Unit tests run fast and don't require a running server. They're excluded from integration test runs.

### Integration Tests (`npm run test:integration` - Vitest)

**Coverage:** Core API functionality, error handling, validation, synchronous processing

**Test File:** `tests/api.test.ts`

**Test Cases:**

- Full E2E workflow
- AI feedback integration
- Grammar error detection
- Confidence tiers and scores
- Context-aware tense detection
- LLM assessment integration
- Teacher feedback persistence
- Streaming endpoints
- Performance timing
- Input validation
- Error handling and retry logic
- Relevance checking
- Cost controls (essay truncation)

**Total:** 28 test cases covering critical API functionality

**Note:** Integration tests require a running API server and are excluded from unit test runs.

### E2E Tests (`tests/e2e/*.spec.ts` - Playwright)

**Coverage:** User-facing flows, UI interactions, visual design, responsive layout

**Test Files (4 files):**

- `core.spec.ts` - Comprehensive core functionality:
  - Homepage and navigation (including custom question card)
  - Essay submission and validation (including custom questions and free writing)
  - Results display and feedback
  - Interactive learning flow
  - Draft tracking and navigation
  - Error handling
  - Results persistence
  - Progress dashboard
- `responsive.spec.ts` - Responsive layouts (mobile/tablet/desktop)
- `visual.spec.ts` - Visual design verification (contrast, touch targets)
- `smoke.spec.ts` - Production smoke test (verifies deployment with real APIs, only runs in CI against production)

**Total:** 20 test cases covering critical user flows

### PWA Tests (`tests/web/pwa.test.ts` - Vitest)

**Coverage:** Progressive Web App functionality

**Test Cases:**

- Service worker registration
- Manifest validation
- Install prompt handling
- Offline functionality (manual testing recommended)

See [PWA_SETUP.md](PWA_SETUP.md) for PWA setup and testing details.

---

## Configuration

Create `.env.local` in project root:

```bash
API_KEY=your-api-key
API_BASE=http://localhost:8787
PLAYWRIGHT_BASE_URL=http://localhost:3000
```

---

## Cost Control: Service Mocking

**Tests automatically use mocked service responses by default** to avoid API costs.

**Mocking System:**

- ✅ **No API costs** - Tests use deterministic mock responses
- ✅ **Automatic detection** - Enabled when `USE_MOCK_SERVICES=true` (default in test configs)
- ✅ **Realistic responses** - Mocks return proper data structures for testing
- ✅ **All services mocked** - Modal, Groq, OpenAI, and other external services
- ✅ **Enhanced error scenarios** - Mocks support timeout, rate limit, and server error simulation
- ✅ **Input validation** - Mocks validate inputs to catch bugs early
- ✅ **Fast execution** - Minimal delays (1-5ms) for rapid test iteration
- ✅ **Mock validation** - Automatic checks ensure mocks are actually being used

**To use real APIs (for integration testing only):**

```bash
USE_MOCK_SERVICES=false npm test
USE_MOCK_SERVICES=false npm run test:integration
USE_MOCK_SERVICES=false npm run test:e2e
```

**Note:** The smoke test (`npm run test:smoke`) is designed to run against production with real APIs and will skip if mocks are enabled.

### Enhanced Mock Features

**Error Scenario Testing:**

Mocks now support simulating error scenarios for comprehensive error handling tests:

```typescript
import { setLLMErrorScenario, MOCK_ERROR_SCENARIOS } from "./helpers/error-scenarios";

test("handles timeout errors", async () => {
  setLLMErrorScenario("TIMEOUT");
  // Test timeout handling
  setLLMErrorScenario(null); // Clear after test
});
```

Available error scenarios:

- `TIMEOUT` - Simulates request timeout
- `RATE_LIMIT` - Simulates rate limiting (429)
- `SERVER_ERROR` - Simulates server errors (500)
- `INVALID_RESPONSE` - Simulates malformed responses

**Mock Validation:**

Tests automatically validate that mocks are enabled. If mocks are disabled, tests will warn (or fail in strict mode) to prevent accidental real API usage.

See [COST_REVIEW.md](COST_REVIEW.md) for cost optimization details.

---

## Test Performance & Parallelization

**Optimized for Speed:**

- ✅ **Fast execution** - Unit tests run in <1s, integration tests in <30s
- ✅ **Parallel execution** - Tests run concurrently by default
- ✅ **Optimized thread pool** - Uses CPU count for optimal parallelism
- ✅ **Minimal delays** - Mock delays reduced to 1-5ms (was 10-50ms)
- ✅ **Test isolation** - Each test runs in isolation for reliability
- ✅ **Fast E2E tests** - Delays reduced when mocks are enabled (10ms vs 300-500ms)

**Configuration:**

- Vitest uses thread pool with up to 8 threads (or CPU count, whichever is lower)
- Tests run concurrently unless explicitly marked as sequential
- Test timeout reduced to 30s (from 60s) - tests should be fast with mocks
- Hook timeout reduced to 10s (from 60s)

## Writing Tests

### API Tests

```typescript
test.concurrent("test name", async () => {
  const { questionId, answerId, submissionId } = generateIds();
  const { status, json } = await apiRequest("POST", `/v1/text/submissions`, {
    submissionId,
    submission: [
      {
        part: 1,
        answers: [
          {
            id: answerId,
            questionId,
            questionText: "Test question",
            text: "Test answer",
          },
        ],
      },
    ],
    template: { name: "generic", version: 1 },
  });
  expect(status).toBe(200);
  expect(json.status).toBe("success");
});
```

### E2E Tests

```typescript
test("description", async ({ writePage, page }) => {
  await writePage.goto("1");
  await writePage.typeEssay("text");
  await page.waitForTimeout(1500);
  await writePage.clickSubmit();
  await expect(page).toHaveURL(/\/results\/[a-f0-9-]+/);
});
```

---

## Test Data

Standard test essays available via `tests/helpers.ts`:

- `getTestEssay("withErrors")` - Essay with grammar errors
- `generateValidEssay()` - 250-500 word essay (valid for submission)

---

## Manual Testing

Some visual and subjective aspects require manual browser verification:

- Color palette and typography
- Spacing and layout (8pt grid)
- Error highlighting opacity
- Heat map visual appearance
- Cross-browser consistency
- Animation verification

---

## Git Hooks

**Pre-commit hook:**

- Security check (prevents committing sensitive files)
- Formats code with Prettier (only staged files for speed)
- Formats Python code with ruff (if Python files changed)
- Runs linting (TypeScript/JavaScript and Python)
- Type checking (TypeScript and Python with mypy)
- Runs unit tests (fast feedback before commit)
- Commit message format hints

**Pre-push hook:**

- Checks/starts local dev servers (reuses if already running)
- Runs unit tests
- Runs API integration tests (with mocked services)
- Runs E2E tests (can be skipped in quick mode)

**Install hooks:**

```bash
npm run install-hooks
# or
./scripts/install-hooks.sh
```

**Quick push mode (skip integration/E2E tests):**

```bash
QUICK_PUSH=true git push  # Saves time by skipping API integration and E2E tests
```

The hook automatically detects docs-only changes and skips integration/E2E tests in that case.

**Bypass hooks (if needed):**

```bash
git commit --no-verify  # Skip pre-commit hook
git push --no-verify    # Skip pre-push hook
```

---

## CI/CD

GitHub Actions automatically:

- Runs tests on pull requests
- Deploys and tests on push to `main`

See `.github/workflows/` for GitHub Actions workflow details.

---

## Test Maintenance

- **Add tests**: When adding new features or endpoints
- **Update tests**: When API contracts change or UI flows change
- **Remove tests**: When features are deprecated

---

## References

- [SCRIPTS.md](SCRIPTS.md) - Scripts documentation
- [COST_REVIEW.md](COST_REVIEW.md) - Cost optimization details
