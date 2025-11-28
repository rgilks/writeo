/**
 * Unit tests for request ID middleware
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { requestId } from "../../apps/api-worker/src/middleware/request-id";
import { createContext } from "./helpers";

describe("requestId middleware", () => {
  const mockNext = vi.fn().mockResolvedValue(undefined);
  const originalCrypto = global.crypto;

  beforeEach(() => {
    vi.clearAllMocks();
    // Mock crypto.randomUUID - ensure it's a function
    if (!global.crypto) {
      global.crypto = {} as any;
    }
    global.crypto.randomUUID = vi.fn(() => "550e8400-e29b-41d4-a716-446655440000");
  });

  afterEach(() => {
    // Don't try to restore crypto - it's read-only in some environments
    vi.restoreAllMocks();
  });

  it("should generate and set request ID", async () => {
    const c = createContext();
    await requestId(c, mockNext);

    expect(c.set).toHaveBeenCalledWith("requestId", expect.any(String));
    expect(mockNext).toHaveBeenCalled();
  });

  it("should use first 8 characters of UUID", async () => {
    if (!global.crypto) {
      global.crypto = {} as any;
    }
    global.crypto.randomUUID = vi.fn(() => "550e8400-e29b-41d4-a716-446655440000");

    const c = createContext();
    await requestId(c, mockNext);

    expect(c.set).toHaveBeenCalledWith("requestId", "550e8400");
  });

  it("should generate unique request IDs for different requests", async () => {
    let callCount = 0;
    if (!global.crypto) {
      global.crypto = {} as any;
    }
    global.crypto.randomUUID = vi.fn(() => {
      callCount++;
      // Return UUIDs where first segment differs
      return `${callCount === 1 ? "aaaa1111" : "bbbb2222"}-1234-5678-90ab-cdef12345678`;
    });

    const c1 = createContext();
    await requestId(c1, mockNext);

    const c2 = createContext();
    await requestId(c2, mockNext);

    const setCalls = c1.set.mock.calls;
    const requestId1 = setCalls.find((call) => call[0] === "requestId")?.[1];
    const setCalls2 = c2.set.mock.calls;
    const requestId2 = setCalls2.find((call) => call[0] === "requestId")?.[1];

    // UUID format: first segment (split("-")[0])
    expect(requestId1).toBe("aaaa1111");
    expect(requestId2).toBe("bbbb2222");
    expect(requestId1).not.toBe(requestId2);
  });

  it("should call next middleware", async () => {
    const c = createContext();
    await requestId(c, mockNext);

    expect(mockNext).toHaveBeenCalledTimes(1);
  });

  it("should set request ID before calling next", async () => {
    const c = createContext();
    let requestIdSet = false;

    mockNext.mockImplementation(async () => {
      // Check that requestId was set before next is called
      requestIdSet = c.get("requestId") !== undefined;
    });

    await requestId(c, mockNext);

    expect(requestIdSet).toBe(true);
  });

  it("should handle UUID generation edge cases", async () => {
    if (!global.crypto) {
      global.crypto = {} as any;
    }
    // UUID with all 'a' in first segment
    global.crypto.randomUUID = vi.fn(() => "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee");

    const c = createContext();
    await requestId(c, mockNext);

    expect(c.set).toHaveBeenCalledWith("requestId", "aaaaaaaa");
  });
});
