/**
 * Unit tests for security middleware
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { securityHeaders, getCorsOrigin } from "../../apps/api-worker/src/middleware/security";
import { createContext } from "./helpers";

describe("securityHeaders middleware", () => {
  const mockNext = vi.fn().mockResolvedValue(undefined);

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should add security headers to response", async () => {
    const c = createContext();
    await securityHeaders(c, mockNext);

    expect(c.header).toHaveBeenCalledWith("X-Content-Type-Options", "nosniff");
    expect(c.header).toHaveBeenCalledWith("X-Frame-Options", "DENY");
    expect(c.header).toHaveBeenCalledWith("X-XSS-Protection", "1; mode=block");
    expect(c.header).toHaveBeenCalledWith("Referrer-Policy", "strict-origin-when-cross-origin");
  });

  it("should call next middleware", async () => {
    const c = createContext();
    await securityHeaders(c, mockNext);

    expect(mockNext).toHaveBeenCalledTimes(1);
  });

  it("should add headers after next is called", async () => {
    const c = createContext();
    let headersSetBeforeNext = false;

    mockNext.mockImplementation(async () => {
      // Check if headers were set before next (they shouldn't be)
      const headerCalls = c.header.mock.calls.length;
      headersSetBeforeNext = headerCalls > 0;
    });

    await securityHeaders(c, mockNext);

    // Headers should be set after next is called
    expect(headersSetBeforeNext).toBe(false);
    expect(c.header).toHaveBeenCalledTimes(4);
  });

  it("should add all required security headers", async () => {
    const c = createContext();
    await securityHeaders(c, mockNext);

    const headerCalls = c.header.mock.calls;
    const headerNames = headerCalls.map((call) => call[0]);

    expect(headerNames).toContain("X-Content-Type-Options");
    expect(headerNames).toContain("X-Frame-Options");
    expect(headerNames).toContain("X-XSS-Protection");
    expect(headerNames).toContain("Referrer-Policy");
    expect(headerNames.length).toBe(4);
  });
});

describe("getCorsOrigin", () => {
  it("should return origin when no allowedOrigins configured", () => {
    const result = getCorsOrigin("https://example.com", undefined);
    expect(result).toBe("https://example.com");
  });

  it("should return null when origin is null and no allowedOrigins", () => {
    const result = getCorsOrigin(null, undefined);
    expect(result).toBeNull();
  });

  it("should return origin when origin is in allowed list", () => {
    const result = getCorsOrigin("https://example.com", "https://example.com,https://other.com");
    expect(result).toBe("https://example.com");
  });

  it("should return null when origin is not in allowed list", () => {
    const result = getCorsOrigin("https://evil.com", "https://example.com,https://other.com");
    expect(result).toBeNull();
  });

  it("should handle comma-separated allowed origins with spaces", () => {
    const result = getCorsOrigin("https://example.com", "https://example.com, https://other.com");
    expect(result).toBe("https://example.com");
  });

  it("should trim whitespace from allowed origins", () => {
    const result = getCorsOrigin(
      "https://example.com",
      "  https://example.com  ,  https://other.com  ",
    );
    expect(result).toBe("https://example.com");
  });

  it("should be case-sensitive for origin matching", () => {
    const result = getCorsOrigin("https://Example.com", "https://example.com");
    expect(result).toBeNull();
  });

  it("should return empty string for empty origin when allowedOrigins is set", () => {
    // Empty string is falsy, so !origin check returns early with origin value (empty string)
    const result = getCorsOrigin("", "https://example.com");
    expect(result).toBe("");
  });

  it("should handle single allowed origin", () => {
    const result = getCorsOrigin("https://example.com", "https://example.com");
    expect(result).toBe("https://example.com");
  });

  it("should return origin when allowedOrigins is empty string", () => {
    // When allowedOrigins is empty string, !allowedOrigins is true (empty string is falsy)
    // So it returns origin early without checking the list
    const result = getCorsOrigin("https://example.com", "");
    expect(result).toBe("https://example.com");
  });
});
