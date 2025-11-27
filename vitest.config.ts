import { defineConfig } from "vitest/config";
import { config } from "dotenv";
import { resolve } from "path";

// Suppress dotenvx verbose messages
process.env.DOTENVX_QUIET = "true";

// Fix NO_COLOR/FORCE_COLOR conflict
if (process.env.NO_COLOR && process.env.FORCE_COLOR) {
  delete process.env.NO_COLOR;
}

// Load environment variables (.env.local takes precedence)
config({ path: resolve(process.cwd(), ".env") });
config({ path: resolve(process.cwd(), ".env.local"), override: true });

// Enable LLM mocking by default to avoid API costs
// Override with USE_MOCK_LLM=false to use real APIs
if (!process.env.USE_MOCK_LLM) {
  process.env.USE_MOCK_LLM = "true";
}

export default defineConfig({
  test: {
    globals: true,
    environment: "node",
    testTimeout: 60000,
    hookTimeout: 60000,
    include: ["tests/**/*.test.ts"],
    // API tests excluded locally to save costs - run explicitly in CI with: npx vitest run tests/api.test.ts
    exclude: ["tests/e2e/**", "tests/api.test.ts"],
    pool: "threads",
    poolOptions: {
      threads: { maxThreads: 3, minThreads: 1 },
    },
    retry: 1,
    env: {
      USE_MOCK_LLM: process.env.USE_MOCK_LLM,
    },
  },
});
