import { defineConfig, devices } from "@playwright/test";
import { config } from "dotenv";
import { resolve } from "path";

// Suppress dotenvx verbose messages
process.env.DOTENVX_QUIET = "true";

// Enable Service mocking by default for E2E tests to avoid API costs
// Override with USE_MOCK_SERVICES=false to use real APIs (for integration testing only)
if (!process.env.USE_MOCK_SERVICES) {
  process.env.USE_MOCK_SERVICES = "true";
}

// Fix NO_COLOR/FORCE_COLOR conflict
if (process.env.NO_COLOR && process.env.FORCE_COLOR) {
  delete process.env.NO_COLOR;
}

// Load .env files (.env.local takes precedence)
// Preserve PLAYWRIGHT_BASE_URL precedence: environment > .env.local > .env
// But default to localhost:3000 for local development if not explicitly set
const envBaseUrl = process.env.PLAYWRIGHT_BASE_URL;
config({ path: resolve(process.cwd(), ".env") });
const envFileBaseUrl = process.env.PLAYWRIGHT_BASE_URL;
config({ path: resolve(process.cwd(), ".env.local"), override: true });
// Restore environment variable if it was set (highest precedence)
// Otherwise fall back to .env if .env.local didn't set it
// But if it's pointing to production and we're not in CI, default to localhost
if (envBaseUrl) {
  process.env.PLAYWRIGHT_BASE_URL = envBaseUrl;
} else if (!process.env.PLAYWRIGHT_BASE_URL && envFileBaseUrl) {
  // Only use .env.local URL if it's not production or if we're in CI
  if (envFileBaseUrl.includes("writeo.tre.systems") && !process.env.CI) {
    // Default to localhost for local development
    process.env.PLAYWRIGHT_BASE_URL = "http://localhost:3000";
  } else {
    process.env.PLAYWRIGHT_BASE_URL = envFileBaseUrl;
  }
}

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: process.env.CI ? 120000 : 60000, // Increased for CI/production
  expect: { timeout: process.env.CI ? 10000 : 5000 }, // Increased for CI
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0, // More retries in CI
  maxFailures: process.env.CI ? 10 : 3, // Allow more failures before stopping
  // Tests are now resilient to localStorage state and hydration timing
  // They clear localStorage at start and work regardless of hydration state
  workers: process.env.CI ? 2 : 4, // Reduce workers in CI to avoid rate limits
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL || "http://localhost:3000",
    actionTimeout: process.env.CI ? 30000 : 10000, // Longer timeout for CI
    navigationTimeout: process.env.CI ? 60000 : 30000, // Longer navigation timeout
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    video: "off",
    // Clear storage state between tests for better isolation
    storageState: undefined,
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command:
      "npm run dev --workspace=@writeo/api-worker -- --port 8788 --var API_KEY:test-key-for-mocked-services --var TEST_API_KEY:test-key-for-mocked-services --var USE_MOCK_SERVICES:true & npm run dev --workspace=@writeo/web",
    port: 3000,
    reuseExistingServer: !process.env.CI,
    stdout: "ignore",
    stderr: "pipe",
    timeout: 120 * 1000,
    env: {
      NEXT_PUBLIC_API_BASE: "http://localhost:8788",
      NEXT_PUBLIC_API_KEY: "test-key-for-mocked-services",
    },
  },
});
