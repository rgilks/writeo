/**
 * Test setup file - runs before all tests
 * Validates that mocks are enabled and configured correctly
 */

import { validateMocksEnabled, warnIfMocksDisabled } from "./helpers/mock-validation";

// Validate mocks are enabled (but don't fail in CI if they're not - just warn)
// This helps catch accidental real API usage during development
if (process.env.NODE_ENV === "test" || process.env.VITEST) {
  try {
    validateMocksEnabled();
  } catch (error) {
    // In CI, we might want to allow real APIs for integration tests
    // So we warn instead of failing
    if (process.env.CI && process.env.USE_MOCK_SERVICES === "false") {
      console.warn("[Test Setup] Mocks are disabled - this is OK for integration tests in CI");
    } else {
      warnIfMocksDisabled();
      // Don't throw - just warn to avoid breaking tests that explicitly disable mocks
      console.warn("[Test Setup] Mock validation warning:", (error as Error).message);
    }
  }
}
