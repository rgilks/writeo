/**
 * Unit tests for context utilities
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { getServices } from "../../apps/api-worker/src/utils/context";
import { createContext } from "./helpers";

describe("getServices", () => {
  const mockR2 = {} as any;
  const mockKv = {} as any;

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("returns config and storage services", () => {
    const c = createContext({
      env: {
        API_KEY: "test-key",
        MODAL_DEBERTA_URL: "https://modal.example.com/deberta",
        WRITEO_DATA: mockR2,
        WRITEO_RESULTS: mockKv,
        OPENAI_API_KEY: "openai-key",
      },
    });

    const services = getServices(c);
    expect(services.config).toBeDefined();
    expect(services.storage).toBeDefined();
    expect(services.config.api.key).toBe("test-key");
  });

  it("throws error on missing required env vars", () => {
    // Create context without default env values
    const c = {
      ...createContext(),
      env: {
        WRITEO_DATA: mockR2,
        WRITEO_RESULTS: mockKv,
        // Missing API_KEY and MODAL_DEBERTA_URL - should throw
      } as any,
    };

    expect(() => getServices(c)).toThrow();
  });
});
