/**
 * Unit tests for error handling utilities
 */

import { describe, it, expect } from "vitest";
import { errorResponse } from "../../apps/api-worker/src/utils/errors";
import { createContext } from "./helpers";

describe("errorResponse", () => {
  it("creates error response with correct status", () => {
    const response = errorResponse(400, "Bad request");
    expect(response.status).toBe(400);
  });

  it("creates error response with JSON body", async () => {
    const response = errorResponse(400, "Bad request");
    const body = await response.json();
    expect(body).toEqual({ error: "Bad request" });
  });

  it("sets Content-Type header", () => {
    const response = errorResponse(400, "Bad request");
    expect(response.headers.get("Content-Type")).toBe("application/json");
  });

  it("sanitizes 5xx errors in production", async () => {
    const prodContext = createContext({ url: "https://api.writeo.com/health" });
    const response = errorResponse(500, "Database connection failed", prodContext);
    expect(response.status).toBe(500);
    // Should sanitize the message
    const body = await response.json();
    expect(body).toEqual({
      error: "An internal error occurred. Please try again later.",
    });
  });

  it("does not sanitize 4xx errors", async () => {
    const prodContext = createContext({ url: "https://api.writeo.com/health" });
    const response = errorResponse(400, "Invalid input", prodContext);
    const body = await response.json();
    expect(body).toEqual({ error: "Invalid input" });
  });

  it("does not sanitize errors in development", async () => {
    const devContext = createContext({
      url: "http://localhost:8787/health",
      env: { ENVIRONMENT: "development" } as any,
    });
    const response = errorResponse(500, "Database connection failed", devContext);
    const body = await response.json();
    expect(body).toEqual({ error: "Database connection failed" });
  });
});
