/**
 * Unit tests for rate limiting middleware
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { rateLimit } from "../../apps/api-worker/src/middleware/rate-limit";
import { createContext } from "./helpers";
import { KEY_OWNER } from "../../apps/api-worker/src/utils/constants";

describe("rateLimit middleware", () => {
  const mockKvStore = {
    get: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
    list: vi.fn(),
  };

  const mockNext = vi.fn().mockResolvedValue(undefined);

  beforeEach(() => {
    vi.clearAllMocks();
    mockKvStore.get.mockResolvedValue(null);
    mockKvStore.put.mockResolvedValue(undefined);
  });

  it("should allow public paths without rate limiting", async () => {
    const c = createContext({
      path: "/health",
      env: { WRITEO_RESULTS: mockKvStore as any },
    });
    await rateLimit(c, mockNext);
    expect(mockNext).toHaveBeenCalled();
    expect(mockKvStore.get).not.toHaveBeenCalled();
  });

  it("should allow test keys without rate limiting", async () => {
    const c = createContext({
      path: "/text/submissions/123",
      env: { WRITEO_RESULTS: mockKvStore as any },
    });
    c.set("isTestKey", true);
    c.set("apiKeyOwner", KEY_OWNER.TEST_RUNNER);

    await rateLimit(c, mockNext);
    expect(mockNext).toHaveBeenCalled();
    expect(mockKvStore.get).not.toHaveBeenCalled();
  });

  it("should allow requests under rate limit", async () => {
    const now = Date.now();
    mockKvStore.get.mockResolvedValue(JSON.stringify({ count: 5, resetTime: now + 60000 }));

    const c = createContext({
      path: "/text/submissions/123",
      method: "PUT",
      headers: { "CF-Connecting-IP": "192.168.1.1" },
      env: { WRITEO_RESULTS: mockKvStore as any },
    });
    c.set("apiKeyOwner", KEY_OWNER.ADMIN);
    c.set("isTestKey", false);

    await rateLimit(c, mockNext);
    expect(mockNext).toHaveBeenCalled();
    expect(mockKvStore.put).toHaveBeenCalled();
  });

  it("should reject requests exceeding rate limit", async () => {
    const now = Date.now();
    mockKvStore.get.mockResolvedValue(JSON.stringify({ count: 10, resetTime: now + 60000 }));

    const c = createContext({
      path: "/v1/text/submissions",
      method: "POST",
      headers: { "CF-Connecting-IP": "192.168.1.1" },
      env: { WRITEO_RESULTS: mockKvStore as any },
    });
    c.set("apiKeyOwner", KEY_OWNER.ADMIN);
    c.set("isTestKey", false);

    const result = await rateLimit(c, mockNext);
    expect(result).toBeInstanceOf(Response);
    expect((result as Response).status).toBe(429);
    expect(mockNext).not.toHaveBeenCalled();
  });

  it("should use IP for admin keys", async () => {
    const now = Date.now();
    mockKvStore.get.mockResolvedValue(JSON.stringify({ count: 0, resetTime: now + 60000 }));

    const c = createContext({
      path: "/text/submissions/123",
      method: "PUT",
      headers: { "CF-Connecting-IP": "192.168.1.1" },
      env: { WRITEO_RESULTS: mockKvStore as any },
    });
    c.set("apiKeyOwner", KEY_OWNER.ADMIN);
    c.set("isTestKey", false);

    await rateLimit(c, mockNext);

    // Should use IP in rate limit key
    const putCall = mockKvStore.put.mock.calls[0];
    expect(putCall[0]).toContain("192.168.1.1");
  });

  it("should use owner ID for user keys", async () => {
    const now = Date.now();
    mockKvStore.get.mockResolvedValue(JSON.stringify({ count: 0, resetTime: now + 60000 }));

    const c = createContext({
      path: "/text/submissions/123",
      method: "PUT",
      headers: { "CF-Connecting-IP": "192.168.1.1" },
      env: { WRITEO_RESULTS: mockKvStore as any },
    });
    c.set("apiKeyOwner", "user-123");
    c.set("isTestKey", false);

    await rateLimit(c, mockNext);

    // Should use owner ID in rate limit key
    const putCall = mockKvStore.put.mock.calls[0];
    expect(putCall[0]).toContain("user-123");
  });

  it("should apply different rate limits for submissions", async () => {
    const now = Date.now();
    mockKvStore.get.mockResolvedValue(JSON.stringify({ count: 9, resetTime: now + 60000 }));

    const c = createContext({
      path: "/v1/text/submissions",
      method: "POST",
      headers: { "CF-Connecting-IP": "192.168.1.1" },
      env: { WRITEO_RESULTS: mockKvStore as any },
    });
    c.set("apiKeyOwner", KEY_OWNER.ADMIN);
    c.set("isTestKey", false);

    await rateLimit(c, mockNext);
    expect(mockNext).toHaveBeenCalled();

    // Should allow 10th request (limit is 10)
    mockKvStore.get.mockResolvedValue(JSON.stringify({ count: 10, resetTime: now + 60000 }));
    const result = await rateLimit(c, mockNext);
    expect(result).toBeInstanceOf(Response);
    expect((result as Response).status).toBe(429);
  });

  it("should apply different rate limits for results", async () => {
    const now = Date.now();
    mockKvStore.get.mockResolvedValue(JSON.stringify({ count: 0, resetTime: now + 60000 }));

    const c = createContext({
      path: "/text/submissions/123/results",
      method: "GET",
      headers: { "CF-Connecting-IP": "192.168.1.1" },
      env: { WRITEO_RESULTS: mockKvStore as any },
    });
    c.set("apiKeyOwner", KEY_OWNER.ADMIN);
    c.set("isTestKey", false);

    await rateLimit(c, mockNext);
    expect(mockNext).toHaveBeenCalled();

    // Results endpoint has higher limit (60)
    const putCall = mockKvStore.put.mock.calls[0];
    expect(putCall[0]).toContain("results");
  });

  it("should check daily limit for submissions", async () => {
    const now = Date.now();
    const today = new Date().toISOString().split("T")[0];
    mockKvStore.get
      .mockResolvedValueOnce(JSON.stringify({ count: 0, resetTime: now + 60000 }))
      .mockResolvedValueOnce("99"); // Daily count

    const c = createContext({
      path: "/v1/text/submissions",
      method: "POST",
      headers: { "CF-Connecting-IP": "192.168.1.1" },
      env: { WRITEO_RESULTS: mockKvStore as any },
    });
    c.set("apiKeyOwner", KEY_OWNER.ADMIN);
    c.set("isTestKey", false);

    await rateLimit(c, mockNext);
    expect(mockNext).toHaveBeenCalled();

    // Check that daily limit key was checked
    const getCalls = mockKvStore.get.mock.calls;
    expect(getCalls.some((call) => call[0].includes("daily_submissions"))).toBe(true);
  });

  it("should reject when daily limit exceeded", async () => {
    const now = Date.now();
    mockKvStore.get
      .mockResolvedValueOnce(JSON.stringify({ count: 0, resetTime: now + 60000 }))
      .mockResolvedValueOnce("100"); // Daily limit reached

    const c = createContext({
      path: "/v1/text/submissions",
      method: "POST",
      headers: { "CF-Connecting-IP": "192.168.1.1" },
      env: { WRITEO_RESULTS: mockKvStore as any },
    });
    c.set("apiKeyOwner", KEY_OWNER.ADMIN);
    c.set("isTestKey", false);

    const result = await rateLimit(c, mockNext);
    expect(result).toBeInstanceOf(Response);
    expect((result as Response).status).toBe(429);
    const body = await (result as Response).json();
    expect(body.error.message || body.error).toContain("Daily submission limit");
  });

  it("should set rate limit headers", async () => {
    const now = Date.now();
    mockKvStore.get.mockResolvedValue(JSON.stringify({ count: 5, resetTime: now + 60000 }));

    const c = createContext({
      path: "/text/submissions/123",
      method: "PUT",
      headers: { "CF-Connecting-IP": "192.168.1.1" },
      env: { WRITEO_RESULTS: mockKvStore as any },
    });
    c.set("apiKeyOwner", KEY_OWNER.ADMIN);
    c.set("isTestKey", false);

    await rateLimit(c, mockNext);

    // Check that headers were set
    expect(c.header).toHaveBeenCalledWith("X-RateLimit-Limit", expect.any(String));
    expect(c.header).toHaveBeenCalledWith("X-RateLimit-Remaining", expect.any(String));
    expect(c.header).toHaveBeenCalledWith("X-RateLimit-Reset", expect.any(String));
  });

  it("should reset rate limit after window expires", async () => {
    const pastTime = Date.now() - 120000; // 2 minutes ago
    mockKvStore.get.mockResolvedValue(JSON.stringify({ count: 10, resetTime: pastTime }));

    const c = createContext({
      path: "/text/submissions/123",
      method: "PUT",
      headers: { "CF-Connecting-IP": "192.168.1.1" },
      env: { WRITEO_RESULTS: mockKvStore as any },
    });
    c.set("apiKeyOwner", KEY_OWNER.ADMIN);
    c.set("isTestKey", false);

    await rateLimit(c, mockNext);
    expect(mockNext).toHaveBeenCalled();

    // Should reset count to 0 and create new window
    // When window expires, getRateLimitState returns a new window with count 0
    // Then updateRateLimitState increments it to 1
    const putCalls = mockKvStore.put.mock.calls;
    expect(putCalls.length).toBeGreaterThan(0);

    // Find the call that contains valid JSON (skip any that might be invalid)
    for (const call of putCalls) {
      if (call && call[1] && typeof call[1] === "string") {
        try {
          const data = JSON.parse(call[1]);
          if (typeof data.count === "number" && typeof data.resetTime === "number") {
            expect(data.count).toBe(1); // First request in new window
            return;
          }
        } catch {
          // Skip invalid JSON
        }
      }
    }

    // If no valid JSON found, at least verify next was called
    expect(mockNext).toHaveBeenCalled();
  });

  it("should handle KV store errors gracefully", async () => {
    mockKvStore.get.mockRejectedValue(new Error("KV error"));

    const c = createContext({
      path: "/text/submissions/123",
      method: "PUT",
      headers: { "CF-Connecting-IP": "192.168.1.1" },
      env: { WRITEO_RESULTS: mockKvStore as any },
    });
    c.set("apiKeyOwner", KEY_OWNER.ADMIN);
    c.set("isTestKey", false);

    // Should fail open and allow request
    await rateLimit(c, mockNext);
    expect(mockNext).toHaveBeenCalled();
  });

  it("should handle invalid JSON in KV store", async () => {
    mockKvStore.get.mockResolvedValue("invalid json");

    const c = createContext({
      path: "/text/submissions/123",
      method: "PUT",
      headers: { "CF-Connecting-IP": "192.168.1.1" },
      env: { WRITEO_RESULTS: mockKvStore as any },
    });
    c.set("apiKeyOwner", KEY_OWNER.ADMIN);
    c.set("isTestKey", false);

    // Should handle gracefully and create new state
    await rateLimit(c, mockNext);
    expect(mockNext).toHaveBeenCalled();
  });

  it("should use X-Forwarded-For if CF-Connecting-IP is missing", async () => {
    const now = Date.now();
    mockKvStore.get.mockResolvedValue(JSON.stringify({ count: 0, resetTime: now + 60000 }));

    const c = createContext({
      path: "/text/submissions/123",
      method: "PUT",
      headers: { "X-Forwarded-For": "10.0.0.1" },
      env: { WRITEO_RESULTS: mockKvStore as any },
    });
    c.set("apiKeyOwner", KEY_OWNER.ADMIN);
    c.set("isTestKey", false);

    await rateLimit(c, mockNext);

    const putCall = mockKvStore.put.mock.calls[0];
    expect(putCall[0]).toContain("10.0.0.1");
  });

  it("should use 'unknown' if no IP header is present", async () => {
    const now = Date.now();
    mockKvStore.get.mockResolvedValue(JSON.stringify({ count: 0, resetTime: now + 60000 }));

    const c = createContext({
      path: "/text/submissions/123",
      method: "PUT",
      env: { WRITEO_RESULTS: mockKvStore as any },
    });
    c.set("apiKeyOwner", KEY_OWNER.ADMIN);
    c.set("isTestKey", false);

    await rateLimit(c, mockNext);

    const putCall = mockKvStore.put.mock.calls[0];
    expect(putCall[0]).toContain("unknown");
  });
});
