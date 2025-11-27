import { defineConfig, devices } from "@playwright/test";
import { config } from "dotenv";
import { resolve } from "path";

// Suppress dotenvx verbose messages
process.env.DOTENVX_QUIET = "true";

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
  timeout: 30000,
  expect: { timeout: 3000 },
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  maxFailures: process.env.CI ? 5 : 3,
  workers: process.env.CI ? 4 : 6,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL || "https://writeo.tre.systems/",
    actionTimeout: 5000,
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    video: "off",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
