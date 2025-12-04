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

// Load .env files (.env.local takes precedence)
config({ path: resolve(process.cwd(), ".env") });
config({ path: resolve(process.cwd(), ".env.local"), override: true });

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
    command: "npm run start:test-server",
    url: "http://localhost:3000",
    reuseExistingServer: true,
    stdout: "ignore",
    stderr: "pipe",
    timeout: 120 * 1000,
    env: {
      USE_MOCK_SERVICES: "true",
      API_KEY: "test-key-for-mocked-services",
      TEST_API_KEY: "test-key-for-mocked-services",
      API_BASE_URL: "http://127.0.0.1:8787",
      NEXT_PUBLIC_API_BASE: "http://127.0.0.1:8787",
      NEXT_PUBLIC_API_KEY: "test-key-for-mocked-services",
    },
  },
});
