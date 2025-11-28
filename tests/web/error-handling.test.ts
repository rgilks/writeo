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

  it("should return rate limit message for 429 status", async () => {
    const response = new Response("", { status: 429 });
    const message = await getErrorMessage(response);
    expect(message).toBe("Too many requests. Please wait a moment and try again.");
  });

  it("should return server error message for 500 status", async () => {
    const response = new Response("", { status: 500 });
    const message = await getErrorMessage(response);
    expect(message).toBe("Server error. Please try again in a moment.");
  });

  it("should return server error message for 503 status", async () => {
    const response = new Response("", { status: 503 });
    const message = await getErrorMessage(response);
    expect(message).toBe("Server error. Please try again in a moment.");
  });

  it("should return not found message for 404 status", async () => {
    const response = new Response("", { status: 404 });
    const message = await getErrorMessage(response);
    expect(message).toBe("Resource not found. Please check and try again.");
  });

  it("should return authentication error for 401 status", async () => {
    const response = new Response("", { status: 401 });
    const message = await getErrorMessage(response);
    expect(message).toBe("Authentication error. Please refresh the page and try again.");
  });

  it("should return authentication error for 403 status", async () => {
    const response = new Response("", { status: 403 });
    const message = await getErrorMessage(response);
    expect(message).toBe("Authentication error. Please refresh the page and try again.");
  });

  it("should extract error message from JSON response", async () => {
    // Mock response.text() to return immediately
    const response = {
      status: 400,
      text: vi.fn().mockResolvedValue(JSON.stringify({ error: "Custom error message" })),
    } as any;
    const message = await getErrorMessage(response);
    expect(message).toBe("Custom error message");
  });

  it("should extract message from JSON response", async () => {
    const response = {
      status: 400,
      text: vi.fn().mockResolvedValue(JSON.stringify({ message: "Error message" })),
    } as any;
    const message = await getErrorMessage(response);
    expect(message).toBe("Error message");
  });

  it("should use text response when JSON parsing fails", async () => {
    const response = {
      status: 400,
      text: vi.fn().mockResolvedValue("Plain text error"),
    } as any;
    const message = await getErrorMessage(response);
    expect(message).toBe("Plain text error");
  });

  it("should return network error message for network-related errors", async () => {
    const response = {
      status: 400,
      text: vi.fn().mockResolvedValue(JSON.stringify({ error: "network connection failed" })),
    } as any;
    const message = await getErrorMessage(response);
    expect(message).toContain("internet connection");
  });

  it("should return timeout message for timeout errors", async () => {
    const response = {
      status: 400,
      text: vi.fn().mockResolvedValue(JSON.stringify({ error: "Request timeout" })),
    } as any;
    const message = await getErrorMessage(response);
    expect(message).toBe("The request took too long. Please try again.");
  });

  it("should return HTTP status code when no error message found", async () => {
    const response = new Response("", { status: 418 });
    const message = await getErrorMessage(response);
    expect(message).toBe("HTTP 418");
  });

  it("should return HTTP status code for empty response", async () => {
    const response = new Response("", { status: 400 });
    const message = await getErrorMessage(response);
    expect(message).toBe("HTTP 400");
  });

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
  it("should convert Error instance to serializable Error", () => {
    const error = new Error("Test error");
    const result = makeSerializableError(error);
    expect(result).toBeInstanceOf(Error);
    expect(result.message).toBe("Test error");
  });

  it("should use default message when Error has no message", () => {
    const error = new Error();
    const result = makeSerializableError(error);
    expect(result.message).toBe("An unexpected error occurred");
  });

  it("should convert string to Error", () => {
    const result = makeSerializableError("String error");
    expect(result).toBeInstanceOf(Error);
    expect(result.message).toBe("String error");
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

  it("should handle empty object", () => {
    const result = makeSerializableError({});
    expect(result.message).toBe("An unexpected error occurred");
  });

  it("should handle null", () => {
    const result = makeSerializableError(null as any);
    // JSON.stringify(null) returns "null", so it creates Error: null
    expect(result.message).toBe("Error: null");
  });

  it("should handle undefined", () => {
    const result = makeSerializableError(undefined);
    expect(result.message).toBe("An unexpected error occurred");
  });

  it("should handle circular references gracefully", () => {
    const circular: any = { prop: "value" };
    circular.self = circular;

    const result = makeSerializableError(circular);
    expect(result).toBeInstanceOf(Error);
    // Should not throw, but may not serialize circular refs
    expect(result.message).toBeDefined();
  });

  it("should handle number", () => {
    const result = makeSerializableError(404);
    expect(result).toBeInstanceOf(Error);
    expect(result.message).toContain("404");
  });
});
