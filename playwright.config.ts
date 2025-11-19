import { defineConfig, devices } from "@playwright/test";
import { config } from "dotenv";
import { resolve } from "path";

// Load .env files (.env.local takes precedence, but preserve PLAYWRIGHT_BASE_URL if set)
config({ path: resolve(process.cwd(), ".env") });
const baseUrl = process.env.PLAYWRIGHT_BASE_URL;
config({ path: resolve(process.cwd(), ".env.local"), override: true });
if (baseUrl) process.env.PLAYWRIGHT_BASE_URL = baseUrl;

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 60000,
  expect: { timeout: 5000 },
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  // Reduce workers for local testing to avoid rate limits (CI uses 1 worker)
  // With 3 workers, max ~180 submissions/min if all run simultaneously, well under 500/min limit
  workers: process.env.CI ? 1 : 3,
  reporter: "html",
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL || "https://writeo.tre.systems/",
    actionTimeout: 10000,
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    video: "off",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
