/**
 * Unit tests for HTTP utility functions
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { postJsonWithAuth } from "../../apps/api-worker/src/utils/http";
import * as fetchWithTimeoutModule from "../../apps/api-worker/src/utils/fetch-with-timeout";

describe("postJsonWithAuth", () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.clearAllMocks();
    global.fetch = vi.fn();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    global.fetch = originalFetch;
  });

  it("should make POST request with JSON body and auth header", async () => {
    const mockResponse = new Response("OK", { status: 200 });
    vi.spyOn(fetchWithTimeoutModule, "fetchWithTimeout").mockResolvedValue(mockResponse);

    const result = await postJsonWithAuth(
      "https://api.example.com/endpoint",
      "test-api-key",
      { data: "value" },
      30000,
    );

    expect(result).toBe(mockResponse);
    expect(fetchWithTimeoutModule.fetchWithTimeout).toHaveBeenCalledWith(
      "https://api.example.com/endpoint",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({
          "Content-Type": "application/json",
          Authorization: "Token test-api-key",
        }),
        body: JSON.stringify({ data: "value" }),
        timeout: 30000,
      }),
    );
  });

  it("should use Token prefix for Authorization header", async () => {
    const mockResponse = new Response("OK", { status: 200 });
    vi.spyOn(fetchWithTimeoutModule, "fetchWithTimeout").mockResolvedValue(mockResponse);

    await postJsonWithAuth("https://api.example.com", "my-key", {}, 30000);

    expect(fetchWithTimeoutModule.fetchWithTimeout).toHaveBeenCalledWith(
      "https://api.example.com",
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: "Token my-key",
        }),
      }),
    );
  });

  it("should stringify request body", async () => {
    const mockResponse = new Response("OK", { status: 200 });
    vi.spyOn(fetchWithTimeoutModule, "fetchWithTimeout").mockResolvedValue(mockResponse);

    const body = { name: "test", count: 123, nested: { value: true } };
    await postJsonWithAuth("https://api.example.com", "key", body, 30000);

    expect(fetchWithTimeoutModule.fetchWithTimeout).toHaveBeenCalledWith(
      "https://api.example.com",
      expect.objectContaining({
        body: JSON.stringify(body),
      }),
    );
  });

  it("should pass timeout to fetchWithTimeout", async () => {
    const mockResponse = new Response("OK", { status: 200 });
    vi.spyOn(fetchWithTimeoutModule, "fetchWithTimeout").mockResolvedValue(mockResponse);

    await postJsonWithAuth("https://api.example.com", "key", {}, 10000);

    expect(fetchWithTimeoutModule.fetchWithTimeout).toHaveBeenCalledWith(
      "https://api.example.com",
      expect.objectContaining({
        timeout: 10000,
      }),
    );
  });

  it.each([
    [{ data: "string" }],
    [{ count: 123 }],
    [{ active: true }],
    [{ items: [1, 2, 3] }],
    [{ nested: { deep: { value: "test" } } }],
  ])("should handle different body types: %j", async (body) => {
    const mockResponse = new Response("OK", { status: 200 });
    vi.spyOn(fetchWithTimeoutModule, "fetchWithTimeout").mockResolvedValue(mockResponse);

    await postJsonWithAuth("https://api.example.com", "key", body, 30000);
    expect(fetchWithTimeoutModule.fetchWithTimeout).toHaveBeenCalledWith(
      "https://api.example.com",
      expect.objectContaining({
        body: JSON.stringify(body),
      }),
    );
  });

  it("should propagate errors from fetchWithTimeout", async () => {
    const error = new Error("Network error");
    vi.spyOn(fetchWithTimeoutModule, "fetchWithTimeout").mockRejectedValue(error);

    await expect(postJsonWithAuth("https://api.example.com", "key", {}, 30000)).rejects.toThrow(
      "Network error",
    );
  });

  it("should handle empty body", async () => {
    const mockResponse = new Response("OK", { status: 200 });
    vi.spyOn(fetchWithTimeoutModule, "fetchWithTimeout").mockResolvedValue(mockResponse);

    await postJsonWithAuth("https://api.example.com", "key", {}, 30000);

    expect(fetchWithTimeoutModule.fetchWithTimeout).toHaveBeenCalledWith(
      "https://api.example.com",
      expect.objectContaining({
        body: "{}",
      }),
    );
  });

  it("should set Content-Type header to application/json", async () => {
    const mockResponse = new Response("OK", { status: 200 });
    vi.spyOn(fetchWithTimeoutModule, "fetchWithTimeout").mockResolvedValue(mockResponse);

    await postJsonWithAuth("https://api.example.com", "key", { test: "data" }, 30000);

    expect(fetchWithTimeoutModule.fetchWithTimeout).toHaveBeenCalledWith(
      "https://api.example.com",
      expect.objectContaining({
        headers: expect.objectContaining({
          "Content-Type": "application/json",
        }),
      }),
    );
  });
});
