import type { Context } from "hono";
import { errorResponse } from "../utils/errors";
import { safeLogError } from "../utils/logging";

export async function authenticate(c: Context, next: () => Promise<void>) {
  const path = new URL(c.req.url).pathname;
  if (path === "/health" || path === "/docs" || path === "/openapi.json") {
    return next();
  }

  const authHeader = c.req.header("Authorization");
  if (!authHeader) {
    return errorResponse(401, "Missing Authorization header", c);
  }

  const match = authHeader.match(/^Token\s+(.+)$/);
  if (!match) {
    return errorResponse(401, "Invalid Authorization header format. Expected: 'Token <key>'", c);
  }

  const providedKey = match[1];
  const expectedKey = c.env.API_KEY;
  const testApiKey = c.env.TEST_API_KEY;

  if (!expectedKey) {
    safeLogError("API_KEY not configured in environment");
    return errorResponse(500, "Server configuration error", c);
  }

  // 1. Check Standard Env Keys (Admin/Test)
  if (providedKey === expectedKey) {
    c.set("apiKeyOwner", "admin");
    c.set("isTestKey", false);
    return next();
  }

  if (testApiKey && providedKey === testApiKey) {
    c.set("apiKeyOwner", "test-runner");
    c.set("isTestKey", true);
    return next();
  }

  // 2. Check KV for User Keys
  try {
    const keyInfoStr = await c.env.WRITEO_RESULTS.get(`apikey:${providedKey}`);
    if (keyInfoStr) {
      const keyInfo = JSON.parse(keyInfoStr);
      c.set("apiKeyOwner", keyInfo.owner || "unknown");
      c.set("isTestKey", false); // User keys are production by default
      return next();
    }
  } catch (error) {
    safeLogError("Error checking API key in KV", error);
    // Continue to failure
  }

  return errorResponse(401, "Invalid API key", c);
}
