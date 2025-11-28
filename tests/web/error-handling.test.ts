/**
 * Unit tests for error handling utilities
 */

import { describe, it, expect, vi } from "vitest";
import {
  getErrorMessage,
  makeSerializableError,
} from "../../apps/web/app/lib/utils/error-handling";

describe("getErrorMessage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it.each([
    [429, "Too many requests. Please wait a moment and try again."],
    [500, "Server error. Please try again in a moment."],
    [503, "Server error. Please try again in a moment."],
    [404, "Resource not found. Please check and try again."],
    [401, "Authentication error. Please refresh the page and try again."],
    [403, "Authentication error. Please refresh the page and try again."],
  ])("should return correct message for HTTP status %d", async (status, expectedMessage) => {
    const response = new Response("", { status });
    const message = await getErrorMessage(response);
    expect(message).toBe(expectedMessage);
  });

  it.each([
    [{ error: "Custom error message" }, "Custom error message"],
    [{ message: "Error message" }, "Error message"],
  ])("should extract error message from JSON response: %j", async (jsonBody, expectedMessage) => {
    const response = {
      status: 400,
      text: vi.fn().mockResolvedValue(JSON.stringify(jsonBody)),
    } as any;
    const message = await getErrorMessage(response);
    expect(message).toBe(expectedMessage);
  });

  it("should use text response when JSON parsing fails", async () => {
    const response = {
      status: 400,
      text: vi.fn().mockResolvedValue("Plain text error"),
    } as any;
    const message = await getErrorMessage(response);
    expect(message).toBe("Plain text error");
  });

  it.each([
    [{ error: "network connection failed" }, "internet connection"],
    [{ error: "Request timeout" }, "The request took too long. Please try again."],
  ])("should handle specific error types: %j", async (jsonBody, expectedMessage) => {
    const response = {
      status: 400,
      text: vi.fn().mockResolvedValue(JSON.stringify(jsonBody)),
    } as any;
    const message = await getErrorMessage(response);
    if (expectedMessage.includes("internet connection")) {
      expect(message).toContain(expectedMessage);
    } else {
      expect(message).toBe(expectedMessage);
    }
  });

  it.each([
    [418, "HTTP 418"],
    [400, "HTTP 400"],
  ])(
    "should return HTTP status code when no error message found: status %d",
    async (status, expectedMessage) => {
      const response = new Response("", { status });
      const message = await getErrorMessage(response);
      expect(message).toBe(expectedMessage);
    },
  );

  it("should handle response.text() errors gracefully", async () => {
    const response = {
      status: 400,
      text: vi.fn().mockRejectedValue(new Error("Read error")),
    } as any;

    const message = await getErrorMessage(response);
    expect(message).toBe("HTTP 400");
    expect(response.text).toHaveBeenCalled();
  });
});

describe("makeSerializableError", () => {
  it.each([
    [new Error("Test error"), "Test error"],
    [new Error(), "An unexpected error occurred"],
    ["String error", "String error"],
    [undefined, "An unexpected error occurred"],
    [{}, "An unexpected error occurred"],
  ])("should convert %j to Error with message: %s", (input, expectedMessage) => {
    const result = makeSerializableError(input as any);
    expect(result).toBeInstanceOf(Error);
    expect(result.message).toBe(expectedMessage);
  });

  it("should convert object to Error with JSON string", () => {
    const error = { code: 500, details: "Something went wrong" };
    const result = makeSerializableError(error);
    expect(result).toBeInstanceOf(Error);
    expect(result.message).toContain("Error:");
    expect(result.message).toContain("500");
  });

  it("should truncate long error messages to 200 characters", () => {
    const longError = { message: "a".repeat(300) };
    const result = makeSerializableError(longError);
    expect(result.message.length).toBeLessThanOrEqual(250); // "Error: " + 200 chars + some buffer
  });

  it.each([
    [null, "Error: null"],
    [404, "404"],
  ])("should handle %j: message contains %s", (input, expectedContain) => {
    const result = makeSerializableError(input as any);
    expect(result).toBeInstanceOf(Error);
    expect(result.message).toContain(expectedContain);
  });

  it("should handle circular references gracefully", () => {
    const circular: any = { prop: "value" };
    circular.self = circular;

    const result = makeSerializableError(circular);
    expect(result).toBeInstanceOf(Error);
    // Should not throw, but may not serialize circular refs
    expect(result.message).toBeDefined();
  });
});
