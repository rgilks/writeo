import { defineConfig, devices } from "@playwright/test";
import { config } from "dotenv";
import { resolve } from "path";

// Suppress dotenvx verbose messages
process.env.DOTENVX_QUIET = "true";

// Fix NO_COLOR/FORCE_COLOR conflict - ensure only FORCE_COLOR is set if needed
if (process.env.NO_COLOR && process.env.FORCE_COLOR) {
  delete process.env.NO_COLOR;
}

// Load .env files (.env.local takes precedence, but preserve PLAYWRIGHT_BASE_URL if set)
config({ path: resolve(process.cwd(), ".env") });
const baseUrl = process.env.PLAYWRIGHT_BASE_URL;
config({ path: resolve(process.cwd(), ".env.local"), override: true });
if (baseUrl) process.env.PLAYWRIGHT_BASE_URL = baseUrl;

export default defineConfig({
  testDir: "./tests/e2e",
  // Reduced timeouts - most tests use mocks now, so they're fast
  timeout: 30000,
  expect: { timeout: 3000 },
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0, // Reduced retries - tests should be deterministic
  // Allow more failures to see full test results
  maxFailures: process.env.CI ? 5 : 3,
  // More workers for faster parallel execution
  // Mock tests don't hit rate limits, so we can run more in parallel
  workers: process.env.CI ? 4 : 6,
  reporter: process.env.CI ? "github" : "list", // Faster reporter for local dev
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL || "https://writeo.tre.systems/",
    actionTimeout: 5000, // Reduced from 10s - mocked tests are faster
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    video: "off",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
