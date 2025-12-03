# Testing Guide

## Overview

Writeo uses `vitest` for unit and integration testing and `playwright` for end-to-end testing.

## Running Tests Locally

### Prerequisites

- Node.js installed
- Dependencies installed (`npm install`)

### Unit Tests

Run unit tests (isolated logic, mocks):

```bash
npm run test:unit
```

### Integration & API Tests

The API tests (`tests/api.test.ts`) require the API Worker and Web App to be running locally.

**Option 1: Automated (Recommended)**
Use the helper script to start servers and run tests:

```bash
npm run test:local
```

**Option 2: Manual**

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
   npm test
   ```

## CI/CD

Tests are automatically run on GitHub Actions via `.github/workflows/deploy-and-test.yml`.
