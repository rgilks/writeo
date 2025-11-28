# Remaining Test Improvements

## ✅ Completed (High Priority)

1. **Shared Package Tests** ✅
   - `tests/shared/validation.test.ts` - 14 tests
   - `tests/shared/text-utils.test.ts` - 20 tests
   - `tests/shared/retry.test.ts` - 15 tests
   - `tests/shared/types.test.ts` - 30 tests
   - **Total: 79 tests**

2. **Rate Limiting Tests** ✅
   - `tests/api-worker/middleware.rate-limit.test.ts` - 16 tests

3. **Storage Utility Tests** ✅
   - `tests/web/storage.test.ts` - 25 tests

**Total New Tests Added: 120 tests**

---

## 🔴 High Priority Remaining

### 1. Middleware Tests (API Worker)

**Missing:**

- `tests/api-worker/middleware.security.test.ts`
  - Test security headers middleware
  - Test CORS origin validation
  - Test security header injection

- `tests/api-worker/middleware.request-id.test.ts`
  - Test request ID generation
  - Test request ID propagation
  - Test request ID in logs

**Files to Test:**

- `apps/api-worker/src/middleware/security.ts`
- `apps/api-worker/src/middleware/request-id.ts`

**Impact:** Security-critical code not tested

---

### 2. HTTP & Fetch Utilities (API Worker)

**Missing:**

- `tests/api-worker/utils.fetch-with-timeout.test.ts`
  - Test timeout handling
  - Test abort controller
  - Test error handling
  - Test timeout edge cases

- `tests/api-worker/utils.http.test.ts`
  - Test `postJsonWithAuth()` function
  - Test authentication header formatting
  - Test JSON body serialization
  - Test timeout propagation

**Files to Test:**

- `apps/api-worker/src/utils/fetch-with-timeout.ts`
- `apps/api-worker/src/utils/http.ts`

**Impact:** Network request handling not tested

---

### 3. Text Processing Utilities (API Worker)

**Missing:**

- `tests/api-worker/utils.text-processing.test.ts`
  - Test confidence score calculation
  - Test position validation and correction
  - Test word boundary alignment
  - Test tense detection
  - Test error position fuzzy matching
  - Test essay truncation

**Files to Test:**

- `apps/api-worker/src/utils/text-processing.ts`

**Impact:** Complex logic for error positioning and confidence scoring not tested

**Note:** `tests/position-validation.test.ts` exists but only covers `validateAndCorrectErrorPosition()`. Need comprehensive tests for all functions.

---

### 4. Web App Utilities

**Missing:**

- `tests/web/text-utils.test.ts`
  - Test `pluralize()` function
  - Test edge cases (0, negative, etc.)

- `tests/web/uuid-utils.test.ts`
  - Test `generateUUID()` function
  - Test crypto.randomUUID() fallback
  - Test UUID format validation

- `tests/web/grammar-rules.test.ts`
  - Test `getGrammarRule()` function
  - Test `getAvailableGrammarRuleTypes()` function
  - Test rule lookup edge cases

- `tests/web/error-logger.test.ts`
  - Test error logging
  - Test warning logging
  - Test error context handling
  - Test error formatting in logs

- `tests/web/error-handling.test.ts`
  - Test `getErrorMessage()` from Response
  - Test `makeSerializableError()` function
  - Test error serialization

**Files to Test:**

- `apps/web/app/lib/utils/text-utils.ts`
- `apps/web/app/lib/utils/uuid-utils.ts`
- `apps/web/app/lib/utils/grammar-rules.ts`
- `apps/web/app/lib/utils/error-logger.ts`
- `apps/web/app/lib/utils/error-handling.ts`

**Impact:** Utility functions used throughout the app not tested

---

### 5. API Client Tests (Web App)

**Missing:**

- `tests/web/api-client.test.ts`
  - Test API request functions
  - Test error handling in API calls
  - Test retry logic integration
  - Test request/response transformation
  - Test authentication handling

**Files to Test:**

- `apps/web/app/lib/utils/api-client.ts`

**Impact:** All API communication not tested at unit level

---

## 🟡 Medium Priority Remaining

### 6. Additional API Worker Utilities

**Missing:**

- `tests/api-worker/utils.logging.test.ts`
  - Test `safeLogError()`, `safeLogWarning()`, `safeLogInfo()`
  - Test sensitive data redaction
  - Test log sanitization
  - Test request ID inclusion

- `tests/api-worker/utils.handlers.test.ts`
  - Test `withErrorHandling()` wrapper
  - Test error propagation
  - Test error logging in handlers

- `tests/api-worker/utils.zod.test.ts`
  - Test `uuidStringSchema()` function
  - Test UUID validation
  - Test error messages

**Files to Test:**

- `apps/api-worker/src/utils/logging.ts`
- `apps/api-worker/src/utils/handlers.ts`
- `apps/api-worker/src/utils/zod.ts`

**Impact:** Logging and error handling utilities not tested

---

### 7. Service Layer Unit Tests (API Worker)

**Missing:**

- `tests/api-worker/services/config.test.ts`
  - Test `buildConfig()` function
  - Test environment variable validation
  - Test LLM config building
  - Test missing env var handling

- `tests/api-worker/services/openai.test.ts`
  - Test OpenAI API calls (mocked)
  - Test response parsing
  - Test error handling
  - Test mock detection

- `tests/api-worker/services/groq.test.ts`
  - Test Groq API calls (mocked)
  - Test response parsing
  - Test error handling
  - Test mock detection

**Files to Test:**

- `apps/api-worker/src/services/config.ts`
- `apps/api-worker/src/services/openai.ts`
- `apps/api-worker/src/services/groq.ts`

**Impact:** Business logic in services not unit tested (relies on integration tests)

**Note:** These are complex and may require significant mocking setup.

---

## 🟢 Low Priority / Future Work

### 8. Component Tests (Web App)

**Missing:**

- React component tests using React Testing Library
- Test user interactions
- Test component rendering
- Test form submissions
- Test error states

**Files to Consider:**

- Critical components in `apps/web/app/components/`
- Form components
- Error boundary components
- Results display components

**Impact:** UI behavior not tested at component level

**Note:** Requires React Testing Library setup and may be lower priority than utility tests.

---

### 9. Server Action Tests (Web App)

**Missing:**

- Tests for Next.js Server Actions
- Mock database/storage calls
- Test form submission handling
- Test server-side validation

**Files to Consider:**

- Server actions in `apps/web/app/actions/` or similar

**Impact:** Server-side logic not tested

**Note:** May require Next.js testing setup.

---

### 10. Route Handler Tests (API Worker)

**Missing:**

- Unit tests for route handlers
- Test request/response transformation
- Test error handling in routes
- Test validation in routes

**Files to Consider:**

- `apps/api-worker/src/routes/*.ts`

**Impact:** Route-level logic not unit tested (covered by integration tests)

**Note:** Lower priority since integration tests cover these well.

---

## 📊 Current Test Coverage Estimate

### Updated Coverage (After Recent Additions)

- **Shared Package**: ~85% coverage ✅ (was 0%)
  - ✅ Validation, text-utils, retry, types all tested
  - ⚠️ Constants don't need tests (just exports)

- **API Worker**: ~50% coverage (was ~40%)
  - ✅ Auth, validation, errors, context, rate limiting
  - ❌ Missing: Security middleware, request-id, fetch-with-timeout, http, text-processing, logging, handlers, zod, services

- **Web App**: ~50% coverage (was ~35%)
  - ✅ Error handling, validation, progress, storage, submission
  - ❌ Missing: text-utils, uuid-utils, grammar-rules, error-logger, error-handling, api-client

- **Overall**: ~45% coverage (was ~30%)

---

## 🎯 Recommended Next Steps (Priority Order)

### Phase 1: Complete High Priority (Next Sprint)

1. **Middleware Tests** (2 files)
   - `tests/api-worker/middleware.security.test.ts`
   - `tests/api-worker/middleware.request-id.test.ts`
   - **Estimated effort:** 2-3 hours

2. **HTTP Utilities** (2 files)
   - `tests/api-worker/utils.fetch-with-timeout.test.ts`
   - `tests/api-worker/utils.http.test.ts`
   - **Estimated effort:** 2-3 hours

3. **Web App Utilities** (5 files)
   - `tests/web/text-utils.test.ts`
   - `tests/web/uuid-utils.test.ts`
   - `tests/web/grammar-rules.test.ts`
   - `tests/web/error-logger.test.ts`
   - `tests/web/error-handling.test.ts`
   - **Estimated effort:** 3-4 hours

4. **API Client Tests** (1 file)
   - `tests/web/api-client.test.ts`
   - **Estimated effort:** 2-3 hours

**Total Phase 1: ~10-13 hours**

---

### Phase 2: Medium Priority (Following Sprint)

5. **Text Processing** (1 file)
   - `tests/api-worker/utils.text-processing.test.ts`
   - **Estimated effort:** 4-5 hours (complex logic)

6. **Additional Utilities** (3 files)
   - `tests/api-worker/utils.logging.test.ts`
   - `tests/api-worker/utils.handlers.test.ts`
   - `tests/api-worker/utils.zod.test.ts`
   - **Estimated effort:** 2-3 hours

**Total Phase 2: ~6-8 hours**

---

### Phase 3: Service Layer (Future)

7. **Service Layer Tests** (3 files)
   - `tests/api-worker/services/config.test.ts`
   - `tests/api-worker/services/openai.test.ts`
   - `tests/api-worker/services/groq.test.ts`
   - **Estimated effort:** 6-8 hours (requires extensive mocking)

---

## 📝 Quick Wins (Can be done quickly)

These are simple utility functions that can be tested quickly:

1. **`tests/web/text-utils.test.ts`** - `pluralize()` function (~30 min)
2. **`tests/web/uuid-utils.test.ts`** - `generateUUID()` function (~30 min)
3. **`tests/api-worker/utils.zod.test.ts`** - `uuidStringSchema()` function (~30 min)
4. **`tests/api-worker/middleware.request-id.test.ts`** - Simple middleware (~1 hour)

**Total Quick Wins: ~2.5 hours**

---

## 🎯 Summary

**High Priority Remaining:**

- 10 test files to create
- Estimated effort: 10-13 hours
- Focus: Security, network utilities, web utilities

**Medium Priority:**

- 4 test files to create
- Estimated effort: 6-8 hours
- Focus: Complex logic, service layer

**Low Priority:**

- Component tests, server action tests
- Estimated effort: Variable
- Focus: UI and server-side logic

**Current Status:**

- ✅ High priority items 1-3 completed (shared, rate limiting, storage)
- 🔴 High priority items 4-7 remaining
- 🟡 Medium priority items 5-7 remaining
- 🟢 Low priority items 8-10 for future consideration

**Next Immediate Actions:**

1. Add middleware tests (security, request-id)
2. Add HTTP/fetch utilities tests
3. Add remaining web app utility tests
4. Add API client tests
