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

if (process.env.API_BASE_OVERRIDE) {
  process.env.API_BASE = process.env.API_BASE_OVERRIDE;
  process.env.API_BASE_URL = process.env.API_BASE_OVERRIDE;
}

// Enable Service mocking by default to avoid API costs
// Override with USE_MOCK_SERVICES=false to use real APIs
if (!process.env.USE_MOCK_SERVICES) {
  process.env.USE_MOCK_SERVICES = "true";
}

const shouldSkipApiTests = process.env.SKIP_API_TESTS === "true";

const buildExcludeList = (): string[] => {
  const baseExclude = ["tests/e2e/**"];
  if (!process.env.CI && shouldSkipApiTests) {
    baseExclude.push("tests/api.test.ts");
  }
  // Exclude integration tests from unit test runs
  if (process.env.TEST_TYPE === "unit") {
    baseExclude.push("tests/**/integration.*.test.ts");
    baseExclude.push("tests/api.test.ts"); // api.test.ts requires running server
  }
  // Exclude unit tests from integration test runs
  if (process.env.TEST_TYPE === "integration") {
    baseExclude.push("tests/**/middleware.*.test.ts");
    baseExclude.push("tests/**/utils.*.test.ts");
    // Include api.test.ts in integration tests (requires running server)
    // This is handled by not excluding it when TEST_TYPE=integration
  }
  return baseExclude;
};

export default defineConfig({
  test: {
    globals: true,
    environment: "node",
    testTimeout: 60000,
    hookTimeout: 60000,
    include: ["tests/**/*.test.ts"],
    exclude: buildExcludeList(),
    pool: "threads",
    poolOptions: {
      threads: { maxThreads: 3, minThreads: 1 },
    },
    retry: 1,
    env: {
      USE_MOCK_SERVICES: process.env.USE_MOCK_SERVICES,
    },
  },
});
