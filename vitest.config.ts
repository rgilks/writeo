import { defineConfig } from "vitest/config";
import { config } from "dotenv";
import { resolve } from "path";

// Suppress dotenvx verbose messages
process.env.DOTENVX_QUIET = "true";

// Fix NO_COLOR/FORCE_COLOR conflict - ensure only FORCE_COLOR is set if needed
if (process.env.NO_COLOR && process.env.FORCE_COLOR) {
  delete process.env.NO_COLOR;
}

config({ path: resolve(process.cwd(), ".env") });
config({ path: resolve(process.cwd(), ".env.local"), override: true });

// Enable LLM mocking by default in tests to avoid API costs
// Set USE_MOCK_LLM=false to use real API (for integration tests)
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
    exclude: ["tests/e2e/**", "node_modules/**", "dist/**"],
    pool: "threads",
    poolOptions: {
      threads: { maxThreads: 3, minThreads: 1 },
    },
    retry: 1,
    env: {
      // Ensure USE_MOCK_LLM is available to test code
      USE_MOCK_LLM: process.env.USE_MOCK_LLM || "true",
    },
  },
});
