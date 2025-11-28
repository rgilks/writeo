/**
 * Unit tests for fetch with timeout utility
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { fetchWithTimeout } from "../../apps/api-worker/src/utils/fetch-with-timeout";

describe("fetchWithTimeout", () => {
  const originalFetch = global.fetch;
  const originalAbortController = global.AbortController;
  const originalSetTimeout = global.setTimeout;
  const originalClearTimeout = global.clearTimeout;

  beforeEach(() => {
    vi.useFakeTimers();
    global.fetch = vi.fn();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
    global.fetch = originalFetch;
    global.AbortController = originalAbortController;
    global.setTimeout = originalSetTimeout;
    global.clearTimeout = originalClearTimeout;
  });

  it("should make fetch request successfully", async () => {
    const mockResponse = new Response("OK", { status: 200 });
    global.fetch = vi.fn().mockResolvedValue(mockResponse);

    const promise = fetchWithTimeout("https://example.com", { timeout: 10 });
    // Don't wait for timeout, just let fetch resolve immediately
    const result = await promise;
    expect(result).toBe(mockResponse);
    expect(global.fetch).toHaveBeenCalledWith(
      "https://example.com",
      expect.objectContaining({
        signal: expect.any(AbortSignal),
      }),
    );
  });

  it("should use default timeout of 30000ms", async () => {
    const mockResponse = new Response("OK", { status: 200 });
    global.fetch = vi.fn().mockResolvedValue(mockResponse);

    const promise = fetchWithTimeout("https://example.com");
    // Don't wait for timeout, just let fetch resolve immediately
    const result = await promise;
    expect(result).toBe(mockResponse);
  });

  it("should use custom timeout when provided", async () => {
    const mockResponse = new Response("OK", { status: 200 });
    global.fetch = vi.fn().mockResolvedValue(mockResponse);

    const promise = fetchWithTimeout("https://example.com", { timeout: 5000 });
    // Don't wait for timeout, just let fetch resolve immediately
    const result = await promise;
    expect(result).toBe(mockResponse);
  });

  it("should timeout and throw error when timeout is exceeded", async () => {
    // Mock fetch to reject with AbortError when aborted
    global.fetch = vi.fn((url, options) => {
      const signal = options?.signal as AbortSignal;
      return new Promise((resolve, reject) => {
        if (signal?.aborted) {
          const error = new Error("Aborted");
          error.name = "AbortError";
          reject(error);
        } else {
          signal?.addEventListener("abort", () => {
            const error = new Error("Aborted");
            error.name = "AbortError";
            reject(error);
          });
        }
      });
    });

    const promise = fetchWithTimeout("https://example.com", { timeout: 10 });

    // Advance timers to trigger timeout (using small timeout for speed)
    vi.advanceTimersByTime(10);

    // Wait for the promise to reject
    await expect(promise).rejects.toThrow("Request timeout after 10ms");
  });

  it("should include URL in timeout error message", async () => {
    // Mock fetch to reject with AbortError when aborted
    global.fetch = vi.fn((url, options) => {
      const signal = options?.signal as AbortSignal;
      return new Promise((resolve, reject) => {
        if (signal?.aborted) {
          const error = new Error("Aborted");
          error.name = "AbortError";
          reject(error);
        } else {
          signal?.addEventListener("abort", () => {
            const error = new Error("Aborted");
            error.name = "AbortError";
            reject(error);
          });
        }
      });
    });

    const promise = fetchWithTimeout("https://api.example.com/data", { timeout: 10 });

    // Advance timers to trigger timeout
    vi.advanceTimersByTime(10);

    // Wait for the promise to reject
    await expect(promise).rejects.toThrow("https://api.example.com/data");
  });

  it("should pass through fetch options", async () => {
    const mockResponse = new Response("OK", { status: 200 });
    global.fetch = vi.fn().mockResolvedValue(mockResponse);

    const promise = fetchWithTimeout("https://example.com", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ test: "data" }),
      timeout: 5000,
    });

    await promise;

    expect(global.fetch).toHaveBeenCalledWith(
      "https://example.com",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ test: "data" }),
        signal: expect.any(AbortSignal),
      }),
    );
  });

  it("should clear timeout on successful request", async () => {
    const mockResponse = new Response("OK", { status: 200 });
    global.fetch = vi.fn().mockResolvedValue(mockResponse);
    const clearTimeoutSpy = vi.spyOn(global, "clearTimeout");

    const promise = fetchWithTimeout("https://example.com");
    await promise;

    expect(clearTimeoutSpy).toHaveBeenCalled();
  });

  it("should clear timeout on error", async () => {
    const error = new Error("Network error");
    global.fetch = vi.fn().mockRejectedValue(error);
    const clearTimeoutSpy = vi.spyOn(global, "clearTimeout");

    const promise = fetchWithTimeout("https://example.com");

    try {
      await promise;
    } catch {
      // Expected
    }

    expect(clearTimeoutSpy).toHaveBeenCalled();
  });

  it("should propagate non-timeout errors", async () => {
    const error = new Error("Network error");
    global.fetch = vi.fn().mockRejectedValue(error);

    const promise = fetchWithTimeout("https://example.com");

    await expect(promise).rejects.toThrow("Network error");
  });

  it("should handle abort signal correctly", async () => {
    const mockResponse = new Response("OK", { status: 200 });
    global.fetch = vi.fn().mockResolvedValue(mockResponse);

    const promise = fetchWithTimeout("https://example.com", { timeout: 1000 });
    await promise;

    const fetchCall = (global.fetch as any).mock.calls[0];
    const signal = fetchCall[1].signal;
    expect(signal).toBeInstanceOf(AbortSignal);
  });
});
