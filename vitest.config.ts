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
    testTimeout: 30000, // Reduced from 60s - tests should be fast with mocks
    hookTimeout: 10000, // Reduced from 60s
    include: ["tests/**/*.test.ts"],
    exclude: buildExcludeList(),
    // Setup file to validate mocks are enabled
    setupFiles: ["./tests/setup.ts"],
    pool: "threads",
    poolOptions: {
      threads: {
        // Optimize for parallel execution - use more threads for faster tests
        // Cap at CPU count to avoid overhead, but allow more parallelism
        maxThreads: process.env.CI ? 4 : Math.max(2, Math.min(8, require("os").cpus().length || 4)),
        minThreads: 1,
        // Use single fork for faster startup
        isolate: true,
      },
    },
    // Enable test isolation for better parallelization
    isolate: true,
    // Run tests in parallel by default (can be overridden per test)
    sequence: {
      shuffle: false, // Deterministic order for reliability
      concurrent: true, // Allow concurrent execution
    },
    retry: process.env.CI ? 1 : 0, // Only retry in CI
    // Faster bail - fail fast on errors
    bail: 0, // Don't bail, but can be set to 1 for faster feedback
    env: {
      USE_MOCK_SERVICES: process.env.USE_MOCK_SERVICES || "true",
    },
    // Coverage settings (if needed)
    coverage: {
      enabled: false, // Disable by default for speed
      provider: "v8",
    },
  },
});
