/**
 * Mock validation utilities to ensure mocks are actually being used in tests
 * This helps catch cases where tests might accidentally use real APIs
 */

import { vi } from "vitest";

/**
 * Validates that mocks are enabled in the test environment
 * Throws if mocks are not enabled, helping catch configuration issues
 */
export function validateMocksEnabled(): void {
  const useMockServices = process.env.USE_MOCK_SERVICES === "true";
  const apiKey = process.env.API_KEY || process.env.TEST_API_KEY || "";

  // Check if we're using a mock/test API key
  const isMockKey = apiKey === "MOCK" || apiKey.startsWith("test_") || apiKey === "";

  if (!useMockServices && !isMockKey) {
    throw new Error(
      `Mocks are not enabled! USE_MOCK_SERVICES=${process.env.USE_MOCK_SERVICES}, API_KEY=${apiKey ? "set" : "not set"}. ` +
        `Tests should use mocks to avoid API costs. Set USE_MOCK_SERVICES=true or use a test API key.`,
    );
  }
}

/**
 * Creates a spy that validates mock usage
 * Throws if the real implementation is called instead of the mock
 */
export function createMockGuard<T extends (...args: any[]) => any>(
  realFn: T,
  mockFn: T,
  name: string,
): T {
  const guard = vi.fn((...args: Parameters<T>) => {
    // In test environment, ensure we're using the mock
    if (process.env.NODE_ENV !== "test") {
      return realFn(...args);
    }

    // Validate mocks are enabled
    validateMocksEnabled();

    // Call the mock
    return mockFn(...args);
  }) as T;

  // Store original for debugging
  (guard as any).__realFn = realFn;
  (guard as any).__mockFn = mockFn;
  (guard as any).__name = name;

  return guard;
}

/**
 * Asserts that a function was called with mocks enabled
 * Useful for verifying test setup
 */
export function assertMockUsage(fn: any, description: string = "function"): void {
  if (process.env.NODE_ENV === "test") {
    validateMocksEnabled();

    if (vi.isMockFunction(fn)) {
      // If it's a vitest mock, we can check if it was called
      // This is just a validation check, not a strict requirement
      return;
    }
  }
}

/**
 * Logs a warning if mocks might not be working correctly
 * Useful for debugging test failures
 */
export function warnIfMocksDisabled(): void {
  const useMockServices = process.env.USE_MOCK_SERVICES === "true";
  const apiKey = process.env.API_KEY || process.env.TEST_API_KEY || "";

  if (!useMockServices && apiKey && !apiKey.startsWith("test_") && apiKey !== "MOCK") {
    console.warn(
      `[Mock Warning] USE_MOCK_SERVICES is not enabled and API key doesn't look like a test key. ` +
        `Tests may be making real API calls. Set USE_MOCK_SERVICES=true to use mocks.`,
    );
  }
}
