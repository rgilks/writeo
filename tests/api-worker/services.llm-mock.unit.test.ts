/**
 * Tests for enhanced LLM mock functionality
 * Verifies that mocks work correctly and support error scenarios
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import {
  mockCallLLMAPI,
  mockStreamLLMAPI,
  MOCK_ERROR_SCENARIOS,
} from "../../apps/api-worker/src/services/llm.mock";
import {
  setLLMErrorScenario,
  clearErrorScenarios,
  setupMockErrorScenarios,
} from "../helpers/error-scenarios";

// Setup error scenario cleanup
setupMockErrorScenarios();

describe("LLM Mock - Enhanced Functionality", () => {
  beforeEach(() => {
    clearErrorScenarios();
  });

  afterEach(() => {
    clearErrorScenarios();
  });

  it("should detect grammar check requests correctly", async () => {
    const response = await mockCallLLMAPI(
      "test-key",
      "gpt-4",
      [
        {
          role: "system",
          content:
            "You are an expert English grammar checker. Check the ENTIRE text systematically.",
        },
        {
          role: "user",
          content: "Check this text: I goes to park yesterday.",
        },
      ],
      2500,
    );

    expect(response).toContain("I go|weekend|to|GRAMMAR");
    expect(response).toContain("We was|park.|playing|GRAMMAR");
  });

  it("should detect teacher feedback requests correctly", async () => {
    const response = await mockCallLLMAPI(
      "test-key",
      "gpt-4",
      [
        {
          role: "system",
          content: "You are a professional writing tutor. Give clear, direct feedback.",
        },
        {
          role: "user",
          content: "Provide clues for this essay.",
        },
      ],
      1000,
    );

    expect(response).toContain("verb tenses");
    expect(response).toContain("yesterday");
  });

  it("should detect combined feedback requests correctly", async () => {
    const response = await mockCallLLMAPI(
      "test-key",
      "gpt-4",
      [
        {
          role: "system",
          content: "You are an expert English language tutor. Always respond with valid JSON only.",
        },
        {
          role: "user",
          content: "Provide detailed feedback for this essay.",
        },
      ],
      2000,
    );

    const parsed = JSON.parse(response);
    expect(parsed).toHaveProperty("detailed");
    expect(parsed).toHaveProperty("teacher");
    expect(parsed.detailed).toHaveProperty("relevance");
    expect(parsed.detailed).toHaveProperty("feedback");
  });

  it("should validate input messages", async () => {
    await expect(mockCallLLMAPI("test-key", "gpt-4", [], 1000)).rejects.toThrow(
      "messages array is required",
    );

    await expect(
      mockCallLLMAPI("test-key", "gpt-4", [{ role: "user" } as any], 1000),
    ).rejects.toThrow("each message must have 'role' and 'content'");
  });

  it("should handle timeout error scenario", async () => {
    setLLMErrorScenario("TIMEOUT");

    // Use Promise.race to timeout the test if the mock doesn't throw
    const testPromise = expect(
      mockCallLLMAPI(
        "test-key",
        "gpt-4",
        [
          {
            role: "user",
            content: "Test message",
          },
        ],
        1000,
      ),
    ).rejects.toThrow("Request timeout");

    const timeoutPromise = new Promise((_, reject) =>
      setTimeout(() => reject(new Error("Test timeout - mock should have thrown")), 40000),
    );

    await Promise.race([testPromise, timeoutPromise]);
  }, 45000); // Allow time for the 35s mock timeout

  it("should handle rate limit error scenario", async () => {
    setLLMErrorScenario("RATE_LIMIT");

    try {
      await mockCallLLMAPI(
        "test-key",
        "gpt-4",
        [
          {
            role: "user",
            content: "Test message",
          },
        ],
        1000,
      );
      expect.fail("Should have thrown rate limit error");
    } catch (error: any) {
      expect(error.message).toContain("Rate limit");
      expect(error.status).toBe(429);
    }
  });

  it("should handle server error scenario", async () => {
    setLLMErrorScenario("SERVER_ERROR");

    try {
      await mockCallLLMAPI(
        "test-key",
        "gpt-4",
        [
          {
            role: "user",
            content: "Test message",
          },
        ],
        1000,
      );
      expect.fail("Should have thrown server error");
    } catch (error: any) {
      expect(error.message).toContain("Internal server error");
      expect(error.status).toBe(500);
    }
  });

  it("should stream responses correctly", async () => {
    const chunks: string[] = [];
    const stream = mockStreamLLMAPI(
      "test-key",
      "gpt-4",
      [
        {
          role: "user",
          content: "Test message",
        },
      ],
      1000,
    );

    for await (const chunk of stream) {
      chunks.push(chunk);
    }

    expect(chunks.length).toBeGreaterThan(0);
    const fullResponse = chunks.join("");
    expect(fullResponse).toBe("Mock LLM API response");
  });

  it("should handle streaming errors", async () => {
    setLLMErrorScenario("TIMEOUT");

    const stream = mockStreamLLMAPI(
      "test-key",
      "gpt-4",
      [
        {
          role: "user",
          content: "Test message",
        },
      ],
      1000,
    );

    // Use Promise.race to timeout the test if the mock doesn't throw
    const testPromise = (async () => {
      for await (const _chunk of stream) {
        // Consume stream
      }
    })();

    const timeoutPromise = new Promise((_, reject) =>
      setTimeout(() => reject(new Error("Test timeout - mock should have thrown")), 40000),
    );

    await expect(Promise.race([testPromise, timeoutPromise])).rejects.toThrow();
  }, 45000); // Allow time for the 35s mock timeout

  it("should return responses quickly (performance test)", async () => {
    const start = Date.now();
    await mockCallLLMAPI(
      "test-key",
      "gpt-4",
      [
        {
          role: "user",
          content: "Test message",
        },
      ],
      1000,
    );
    const duration = Date.now() - start;

    // Should be very fast with minimal delays (1-5ms)
    expect(duration).toBeLessThan(50); // Allow some buffer
  });
});
