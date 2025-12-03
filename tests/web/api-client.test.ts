/**
 * Unit tests for API client utilities
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { apiRequest } from "../../apps/web/app/lib/utils/api-client";
import * as apiConfigModule from "../../apps/web/app/lib/api-config";

describe.sequential("apiRequest", () => {
  const originalFetch = global.fetch;
  const originalProcessEnv = process.env;

  beforeEach(() => {
    vi.clearAllMocks();
    process.env = { ...originalProcessEnv };
  });

  afterEach(() => {
    vi.restoreAllMocks();
    global.fetch = originalFetch;
    process.env = originalProcessEnv;
  });

  it("should make GET request with auth header", async () => {
    const mockResponse = new Response("OK", { status: 200 });
    const mockFetch = vi.fn().mockResolvedValue(mockResponse);
    global.fetch = mockFetch;

    vi.spyOn(apiConfigModule, "getApiBase").mockReturnValue("https://api.example.com");
    vi.spyOn(apiConfigModule, "getApiKey").mockReturnValue("test-api-key");

    const result = await apiRequest({ endpoint: "/test" });

    expect(result).toBe(mockResponse);
    expect(mockFetch).toHaveBeenCalled();
    const fetchCall = mockFetch.mock.calls[0];
    expect(fetchCall[0]).toBe("https://api.example.com/test");
    expect(fetchCall[1].method).toBe("GET");
    expect(fetchCall[1].headers.get("Content-Type")).toBe("application/json");
    expect(fetchCall[1].headers.get("Authorization")).toBe("Token test-api-key");
  });

  it("should make POST request with JSON body", async () => {
    const mockResponse = new Response("OK", { status: 200 });
    const mockFetch = vi.fn().mockResolvedValue(mockResponse);
    global.fetch = mockFetch;

    vi.spyOn(apiConfigModule, "getApiBase").mockReturnValue("https://api.example.com");
    vi.spyOn(apiConfigModule, "getApiKey").mockReturnValue("test-api-key");

    const body = { name: "test", value: 123 };
    await apiRequest({ endpoint: "/test", method: "POST", body });

    expect(mockFetch).toHaveBeenCalledWith(
      "https://api.example.com/test",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify(body),
      }),
    );
  });

  it("should use Token prefix for Authorization header", async () => {
    const mockResponse = new Response("OK", { status: 200 });
    const mockFetch = vi.fn().mockResolvedValue(mockResponse);
    global.fetch = mockFetch;

    vi.spyOn(apiConfigModule, "getApiBase").mockReturnValue("https://api.example.com");
    vi.spyOn(apiConfigModule, "getApiKey").mockReturnValue("my-api-key");

    await apiRequest({ endpoint: "/test" });

    expect(mockFetch).toHaveBeenCalled();
    const fetchCall = mockFetch.mock.calls[0];
    expect(fetchCall[1].headers.get("Authorization")).toBe("Token my-api-key");
  });

  it.each([
    ["MISSING_API_KEY", "https://api.example.com", "API key"],
    ["test-key", "MISSING_API_BASE_URL", "API base URL"],
  ])("should throw error when %s is missing", async (apiKey, apiBase, description) => {
    vi.spyOn(apiConfigModule, "getApiBase").mockReturnValue(apiBase);
    vi.spyOn(apiConfigModule, "getApiKey").mockReturnValue(apiKey);

    await expect(apiRequest({ endpoint: "/test" })).rejects.toThrow(
      "Server configuration error: API credentials not set",
    );
  });

  it("should pass through additional fetch options", async () => {
    const mockResponse = new Response("OK", { status: 200 });
    const mockFetch = vi.fn().mockResolvedValue(mockResponse);
    global.fetch = mockFetch;

    vi.spyOn(apiConfigModule, "getApiBase").mockReturnValue("https://api.example.com");
    vi.spyOn(apiConfigModule, "getApiKey").mockReturnValue("test-key");

    await apiRequest({
      endpoint: "/test",
      headers: { "X-Custom-Header": "value" },
      cache: "no-cache",
    });

    expect(mockFetch).toHaveBeenCalled();
    const fetchCall = mockFetch.mock.calls[0];
    expect(fetchCall[1].cache).toBe("no-cache");
    expect(fetchCall[1].headers.get("Content-Type")).toBe("application/json");
    expect(fetchCall[1].headers.get("Authorization")).toBe("Token test-key");
    expect(fetchCall[1].headers.get("X-Custom-Header")).toBe("value");
  });

  it("should not include body for GET requests", async () => {
    const mockResponse = new Response("OK", { status: 200 });
    const mockFetch = vi.fn().mockResolvedValue(mockResponse);
    global.fetch = mockFetch;

    vi.spyOn(apiConfigModule, "getApiBase").mockReturnValue("https://api.example.com");
    vi.spyOn(apiConfigModule, "getApiKey").mockReturnValue("test-key");

    await apiRequest({ endpoint: "/test", method: "GET" });

    const fetchCall = mockFetch.mock.calls[0];
    expect(fetchCall[1].body).toBeUndefined();
  });

  it("should stringify body for POST requests", async () => {
    const mockResponse = new Response("OK", { status: 200 });
    const mockFetch = vi.fn().mockResolvedValue(mockResponse);
    global.fetch = mockFetch;

    vi.spyOn(apiConfigModule, "getApiBase").mockReturnValue("https://api.example.com");
    vi.spyOn(apiConfigModule, "getApiKey").mockReturnValue("test-key");

    const body = { test: "data", count: 42 };
    await apiRequest({ endpoint: "/test", method: "POST", body });

    expect(mockFetch).toHaveBeenCalledWith(
      "https://api.example.com/test",
      expect.objectContaining({
        body: JSON.stringify(body),
      }),
    );
  });

  it("should handle undefined body", async () => {
    const mockResponse = new Response("OK", { status: 200 });
    const mockFetch = vi.fn().mockResolvedValue(mockResponse);
    global.fetch = mockFetch;

    vi.spyOn(apiConfigModule, "getApiBase").mockReturnValue("https://api.example.com");
    vi.spyOn(apiConfigModule, "getApiKey").mockReturnValue("test-key");

    await apiRequest({ endpoint: "/test", method: "GET", body: undefined });

    expect(mockFetch).toHaveBeenCalled();
    const fetchCall = mockFetch.mock.calls[0];
    expect(fetchCall).toBeDefined();
    expect(fetchCall[1].body).toBeUndefined();
  });

  it("should merge custom headers with default headers", async () => {
    const mockResponse = new Response("OK", { status: 200 });
    const mockFetch = vi.fn().mockResolvedValue(mockResponse);
    global.fetch = mockFetch;

    vi.spyOn(apiConfigModule, "getApiBase").mockReturnValue("https://api.example.com");
    vi.spyOn(apiConfigModule, "getApiKey").mockReturnValue("test-key");

    await apiRequest({
      endpoint: "/test",
      headers: {
        "X-Custom": "value",
        "Content-Type": "application/xml", // Note: default overwrites this
      },
    });

    expect(mockFetch).toHaveBeenCalled();
    const fetchCall = mockFetch.mock.calls[0];
    // Default Content-Type is set after custom headers, so it overwrites
    expect(fetchCall[1].headers.get("Content-Type")).toBe("application/json");
    expect(fetchCall[1].headers.get("X-Custom")).toBe("value");
    expect(fetchCall[1].headers.get("Authorization")).toBe("Token test-key");
  });

  it("should return response from fetch", async () => {
    const mockResponse = new Response("OK", { status: 200 });
    const mockFetch = vi.fn().mockResolvedValue(mockResponse);
    global.fetch = mockFetch;

    vi.spyOn(apiConfigModule, "getApiBase").mockReturnValue("https://api.example.com");
    vi.spyOn(apiConfigModule, "getApiKey").mockReturnValue("test-key");

    const result = await apiRequest({ endpoint: "/test" });
    expect(result).toBe(mockResponse);
  });
});
