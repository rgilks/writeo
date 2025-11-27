/**
 * Authentication utilities for feedback routes
 */

import { errorResponse } from "../../utils/errors";
import type { Context } from "hono";
import type { Env } from "../../types/env";

/**
 * Validates the API key from the Authorization header.
 * Supports both the production API_KEY and optional TEST_API_KEY.
 *
 * @param c - Hono context with environment bindings
 * @returns Error response if unauthorized, null if valid
 */
export function validateApiKey(
  c: Context<{ Bindings: Env; Variables: { requestId?: string } }>,
): Response | null {
  const authHeader = c.req.header("Authorization");
  if (!authHeader) {
    return errorResponse(401, "Unauthorized", c);
  }

  // Extract API key, handling both "Token <key>" and bare key formats
  const apiKey = authHeader.replace(/^Token\s+/i, "").trim();
  if (!apiKey) {
    return errorResponse(401, "Unauthorized", c);
  }

  const validKeys = [c.env.API_KEY];
  if (c.env.TEST_API_KEY) {
    validKeys.push(c.env.TEST_API_KEY);
  }

  if (!validKeys.includes(apiKey)) {
    return errorResponse(401, "Unauthorized", c);
  }

  return null;
}
