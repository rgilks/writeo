/**
 * Test utilities for simulating error scenarios with mocks
 * This allows tests to verify error handling without making real API calls
 */

import {
  setMockErrorScenario,
  getMockErrorScenario,
  MOCK_ERROR_SCENARIOS,
} from "../../apps/api-worker/src/services/llm.mock";
import {
  setMockModalErrorScenario,
  getMockModalErrorScenario,
  MOCK_MODAL_ERROR_SCENARIOS,
} from "../../apps/api-worker/src/services/modal/mock";
import { beforeEach, afterEach } from "vitest";

/**
 * Reset all mock error scenarios before each test
 * Ensures test isolation
 */
export function setupMockErrorScenarios(): void {
  beforeEach(() => {
    setMockErrorScenario(null);
    setMockModalErrorScenario(null);
  });

  afterEach(() => {
    setMockErrorScenario(null);
    setMockModalErrorScenario(null);
  });
}

/**
 * Set an error scenario for LLM mocks
 * @param scenario - The error scenario to simulate, or null to clear
 */
export function setLLMErrorScenario(scenario: keyof typeof MOCK_ERROR_SCENARIOS | null): void {
  setMockErrorScenario(scenario ? MOCK_ERROR_SCENARIOS[scenario] : null);
}

/**
 * Set an error scenario for Modal mocks
 * @param scenario - The error scenario to simulate, or null to clear
 */
export function setModalErrorScenario(
  scenario: keyof typeof MOCK_MODAL_ERROR_SCENARIOS | null,
): void {
  setMockModalErrorScenario(scenario ? MOCK_MODAL_ERROR_SCENARIOS[scenario] : null);
}

/**
 * Clear all error scenarios
 */
export function clearErrorScenarios(): void {
  setMockErrorScenario(null);
  setMockModalErrorScenario(null);
}

/**
 * Get current error scenarios (for debugging)
 */
export function getErrorScenarios(): {
  llm: string | null;
  modal: string | null;
} {
  return {
    llm: getMockErrorScenario(),
    modal: getMockModalErrorScenario(),
  };
}

// Export error scenario constants for use in tests
export { MOCK_ERROR_SCENARIOS, MOCK_MODAL_ERROR_SCENARIOS };
