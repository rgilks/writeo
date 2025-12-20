# Testing Guide

## Overview

Writeo uses **Vitest** for unit/integration testing and **Playwright** for end-to-end (E2E) testing.

## Running Tests Locally

### Prerequisites

- Node.js installed
- Dependencies installed (`npm install`)

### Unit Tests

Run unit tests (isolated logic, mocks services). These are fast and don't require running servers.

```bash
npm run test:unit
```

**Watch Mode:**

```bash
npm run test:watch
```

**UI Mode:**

```bash
npm run test:ui
```

### Integration & API Tests

The API tests (`tests/api.test.ts`) require the API Worker and Web App to be running locally.

**Option 1: Automated Helper (Recommended)**

Start both the API Worker (port 8787) and Web App (port 3000) in test mode:

```bash
npm run start:test-server
```

Then, in a separate terminal, run the integration tests:

```bash
npm run test:integration
```

**Option 2: Manual Startup**

1. Start the API worker:
   ```bash
   cd apps/api-worker
   npm run dev
   ```
2. Start the Web app (in a separate terminal):
   ```bash
   cd apps/web
   npm run dev
   ```
3. Run the tests:
   ```bash
   npm run test:integration
   ```

### End-to-End (E2E) Tests

E2E tests use Playwright to drive a real browser. Requires servers to be running (see above).

**Run all E2E tests:**

```bash
npm run test:e2e
```

**Run with UI (Time Travel Debugging):**

```bash
npm run test:e2e:ui
```

**Debug Mode:**

```bash
npm run test:e2e:debug
```

### Smoke Tests

Smoke tests run against a **deployed environment** (Production or Staging) using real APIs (not mocked).

```bash
npm run test:smoke
```

## CI/CD Workflow

Tests are automatically run on GitHub Actions via `.github/workflows/deploy-and-test.yml`.

1.  **Pull Requests & Pushes**:
    - Runs `lint`, `format:check`, `type-check`.
    - Runs **Unit Tests** (mocks enabled).
    - Runs **E2E Tests** against a local environment with mocked services.

2.  **Deployment (Main Branch)**:
    - After successful deployment, runs **Smoke Tests** against the live production URL.
    - These tests use **Real APIs** (Modal/Groq/OpenAI) to verify the full system.
