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
  const testApiKey = (c.env as any).TEST_API_KEY;

  if (!expectedKey) {
    safeLogError("API_KEY not configured in environment");
    return errorResponse(500, "Server configuration error", c);
  }

  // Accept either API_KEY or TEST_API_KEY
  const isValidKey = providedKey === expectedKey || (testApiKey && providedKey === testApiKey);

  if (!isValidKey) {
    return errorResponse(401, "Invalid API key", c);
  }

  return next();
}
