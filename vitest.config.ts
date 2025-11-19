import { defineConfig } from "vitest/config";
import { config } from "dotenv";
import { resolve } from "path";

config({ path: resolve(process.cwd(), ".env") });
config({ path: resolve(process.cwd(), ".env.local"), override: true });

// Enable LLM mocking by default in tests to avoid API costs
// Set MOCK_GROQ=false and MOCK_OPENAI=false to use real API (for integration tests)
if (!process.env.MOCK_GROQ) {
  process.env.MOCK_GROQ = "true";
}
if (!process.env.MOCK_OPENAI) {
  process.env.MOCK_OPENAI = "true";
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
      // Ensure MOCK_GROQ and MOCK_OPENAI are available to test code
      MOCK_GROQ: process.env.MOCK_GROQ || "true",
      MOCK_OPENAI: process.env.MOCK_OPENAI || "true",
    },
  },
});
