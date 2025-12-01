import { defineConfig, devices } from "@playwright/test";
import { config } from "dotenv";
import { resolve } from "path";

// Suppress dotenvx verbose messages
process.env.DOTENVX_QUIET = "true";

// Enable LLM mocking by default for E2E tests to avoid API costs
// Override with USE_MOCK_LLM=false to use real APIs (for integration testing only)
if (!process.env.USE_MOCK_LLM) {
  process.env.USE_MOCK_LLM = "true";
}

// Fix NO_COLOR/FORCE_COLOR conflict
if (process.env.NO_COLOR && process.env.FORCE_COLOR) {
  delete process.env.NO_COLOR;
}

// Load .env files (.env.local takes precedence)
// Preserve PLAYWRIGHT_BASE_URL precedence: environment > .env.local > .env
const envBaseUrl = process.env.PLAYWRIGHT_BASE_URL;
config({ path: resolve(process.cwd(), ".env") });
const envFileBaseUrl = process.env.PLAYWRIGHT_BASE_URL;
config({ path: resolve(process.cwd(), ".env.local"), override: true });
// Restore environment variable if it was set (highest precedence)
// Otherwise fall back to .env if .env.local didn't set it
if (envBaseUrl) {
  process.env.PLAYWRIGHT_BASE_URL = envBaseUrl;
} else if (!process.env.PLAYWRIGHT_BASE_URL && envFileBaseUrl) {
  process.env.PLAYWRIGHT_BASE_URL = envFileBaseUrl;
}

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 60000, // Increased for slower operations
  expect: { timeout: 5000 }, // Increased for better reliability
  fullyParallel: false, // Disable parallel execution to reduce flakiness
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  maxFailures: process.env.CI ? 5 : 3,
  workers: 1, // Run tests sequentially to avoid resource contention and flakiness
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL || "http://localhost:3000",
    actionTimeout: 10000,
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    video: "off",
    // Clear storage state between tests for better isolation
    storageState: undefined,
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
