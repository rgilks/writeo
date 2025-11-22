/**
 * Authentication utilities for feedback routes
 */

import { errorResponse } from "../../utils/errors";
import type { Context } from "hono";
import type { Env } from "../../types/env";

export function validateApiKey(c: Context<{ Bindings: Env }>): Response | null {
  const apiKey = c.req.header("Authorization")?.replace(/^Token\s+/i, "");
  const testApiKey = (c.env as any).TEST_API_KEY;
  const isValidKey = apiKey && (apiKey === c.env.API_KEY || (testApiKey && apiKey === testApiKey));
  if (!isValidKey) {
    return errorResponse(401, "Unauthorized", c);
  }
  return null;
}
