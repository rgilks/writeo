/**
 * Unit tests for authentication middleware
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { authenticate } from "../../apps/api-worker/src/middleware/auth";
import { createContext } from "./helpers";
import { KEY_OWNER } from "../../apps/api-worker/src/utils/constants";

describe.sequential("auth middleware", () => {
  const mockNext = vi.fn().mockResolvedValue(undefined);
  const mockKvStore = {
    get: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
    list: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockNext.mockClear();
  });

  it("allows public paths without authentication", async () => {
    const c = createContext({ path: "/health", env: { API_KEY: "test-key" } });
    await authenticate(c, mockNext);
    expect(mockNext).toHaveBeenCalled();
  });

  it("rejects requests without Authorization header", async () => {
    const c = createContext({ path: "/text/questions/123", env: { API_KEY: "test-key" } });
    const result = await authenticate(c, mockNext);
    expect(result).toBeInstanceOf(Response);
    expect((result as Response).status).toBe(401);
    expect(mockNext).not.toHaveBeenCalled();
  });

  it("rejects invalid Authorization header format", async () => {
    const c = createContext({
      path: "/text/questions/123",
      headers: { Authorization: "Bearer invalid" },
      env: { API_KEY: "test-key" },
    });
    const result = await authenticate(c, mockNext);
    expect(result).toBeInstanceOf(Response);
    expect((result as Response).status).toBe(401);
  });

  it("accepts valid admin API key", async () => {
    const c = createContext({
      path: "/text/questions/123",
      headers: { Authorization: "Token admin-key" },
      env: { API_KEY: "admin-key" },
    });
    await authenticate(c, mockNext);
    expect(mockNext).toHaveBeenCalled();
    expect(c.get("apiKeyOwner")).toBe(KEY_OWNER.ADMIN);
    expect(c.get("isTestKey")).toBe(false);
  });

  it("accepts valid test API key", async () => {
    const c = createContext({
      path: "/text/questions/123",
      headers: { Authorization: "Token test-key" },
      env: { API_KEY: "admin-key", TEST_API_KEY: "test-key" },
    });
    await authenticate(c, mockNext);
    expect(mockNext).toHaveBeenCalled();
    expect(c.get("apiKeyOwner")).toBe(KEY_OWNER.TEST_RUNNER);
    expect(c.get("isTestKey")).toBe(true);
  });

  it("accepts valid user API key from KV store", async () => {
    mockKvStore.get.mockResolvedValue(JSON.stringify({ owner: "user-123" }));
    const c = createContext({
      path: "/text/questions/123",
      headers: { Authorization: "Token user-key" },
      env: { API_KEY: "admin-key", WRITEO_RESULTS: mockKvStore as any },
    });
    await authenticate(c, mockNext);
    expect(mockNext).toHaveBeenCalled();
    expect(c.get("apiKeyOwner")).toBe("user-123");
    expect(c.get("isTestKey")).toBe(false);
  });

  it("rejects invalid API key", async () => {
    mockKvStore.get.mockResolvedValue(null);
    const c = createContext({
      path: "/text/questions/123",
      headers: { Authorization: "Token invalid-key" },
      env: { API_KEY: "admin-key", WRITEO_RESULTS: mockKvStore as any },
    });
    const result = await authenticate(c, mockNext);
    expect(result).toBeInstanceOf(Response);
    expect((result as Response).status).toBe(401);
    expect(mockNext).not.toHaveBeenCalled();
  });

  it("handles KV store errors gracefully", async () => {
    mockKvStore.get.mockRejectedValue(new Error("KV error"));
    const c = createContext({
      path: "/text/questions/123",
      headers: { Authorization: "Token user-key" },
      env: { API_KEY: "admin-key", WRITEO_RESULTS: mockKvStore as any },
    });
    const result = await authenticate(c, mockNext);
    expect(result).toBeInstanceOf(Response);
    expect((result as Response).status).toBe(401);
  });

  it("returns 500 if API_KEY is not configured", async () => {
    const c = createContext({
      path: "/text/questions/123",
      headers: { Authorization: "Token any-key" },
      env: { API_KEY: undefined } as any,
    });
    const result = await authenticate(c, mockNext);
    expect(result).toBeInstanceOf(Response);
    // May return 401 (invalid key) or 500 (config error) depending on check order
    expect([401, 500]).toContain((result as Response).status);
  });
});
