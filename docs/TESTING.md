# Testing Guide

**Test Coverage:** 453 unit tests + 28 API integration tests + 20 E2E tests covering critical functionality

---

## Quick Start

```bash
npm test              # API tests (Vitest)
npm run test:e2e      # E2E tests (Playwright)
npm run test:all      # Both API + E2E
npm run test:watch    # Watch mode (API tests)
npm run test:e2e:ui   # Playwright UI mode
```

> Temporarily need the old lightweight run? Set `SKIP_API_TESTS=true` before invoking
> Vitest (for example, `SKIP_API_TESTS=true npm test`) to skip the API suite. By default,
> API tests now run locally.

---

## Test Structure

### API Tests (`tests/api.test.ts` - Vitest)

**Coverage:** Core API functionality, error handling, validation, synchronous processing

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

### Unit Tests (`tests/**/*.test.ts` - Vitest)

**Coverage:** Utilities, middleware, validation, error handling, storage, shared packages

**Test Files (31 files):**

- **API Worker** (11 files): Auth, rate limiting, security, validation, errors, context, HTTP utilities, Zod utilities, fetch-with-timeout, request-id
- **Web App** (13 files): Error handling, validation, progress, storage, submission, text utils, UUID utils, grammar rules, error logger, API client, learner results, error utils
- **Shared Package** (4 files): Validation, text utils, retry logic, type guards
- **Integration** (1 file): Position validation
- **API Integration** (1 file): Full E2E API workflows

**Total:** 453 unit tests covering critical utilities and middleware

### E2E Tests (`tests/e2e/*.spec.ts` - Playwright)

**Coverage:** User-facing flows, UI interactions, visual design, responsive layout

**Test Files (3 files):**

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

**Total:** 20 test cases covering critical user flows

---

## Configuration

Create `.env.local` in project root:

```bash
API_KEY=your-api-key
API_BASE=http://localhost:8787
PLAYWRIGHT_BASE_URL=http://localhost:3000
```

---

## Cost Control: LLM API Mocking

**Tests automatically use mocked LLM API responses by default** to avoid API costs.

**Mocking System:**

- ✅ **No API costs** - Tests use deterministic mock responses
- ✅ **Automatic detection** - Enabled when `USE_MOCK_LLM=true` or API key is `MOCK`/`test_*`
- ✅ **Realistic responses** - Mocks return proper data structures for testing

**To use real API (for integration tests only):**

```bash
USE_MOCK_LLM=false npm test
```

See [COST_REVIEW.md](COST_REVIEW.md) for cost optimization details.

---

## Writing Tests

### API Tests

```typescript
test.concurrent("test name", async () => {
  const { questionId, answerId, submissionId } = generateIds();
  const { status, json } = await apiRequest("PUT", `/text/submissions/${submissionId}`, {
    submission: [
      {
        part: "1",
        answers: [
          {
            id: answerId,
            "question-number": 1,
            "question-id": questionId,
            "question-text": "Test question",
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
- Runs linting
- Type checking
- Commit message format hints

**Pre-push hook:**

- Checks/starts local dev servers (reuses if already running)
- Runs unit tests
- Runs E2E tests (can be skipped in quick mode)

**Install hooks:**

```bash
npm run install-hooks
# or
./scripts/install-hooks.sh
```

**Quick push mode (skip E2E tests):**

```bash
QUICK_PUSH=true git push  # Saves ~20 seconds by skipping E2E tests
```

The hook automatically detects docs-only changes and skips E2E tests in that case.

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

See [.github/README.md](../.github/README.md) for workflow details.

---

## Test Maintenance

- **Add tests**: When adding new features or endpoints
- **Update tests**: When API contracts change or UI flows change
- **Remove tests**: When features are deprecated

---

## References

- [SCRIPTS.md](SCRIPTS.md) - Scripts documentation
- [COST_REVIEW.md](COST_REVIEW.md) - Cost optimization details
