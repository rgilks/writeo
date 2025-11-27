import type { Context } from "hono";
import type { Env } from "../types/env";

const SERVER_ERROR_THRESHOLD = 500;
const SANITIZED_ERROR_MESSAGE = "An internal error occurred. Please try again later.";

/**
 * Detects production environment by checking if URL contains localhost/127.0.0.1.
 * Defaults to production when context is unavailable (fail-safe).
 */
function isProduction(c?: Context<{ Bindings: Env }> | Context): boolean {
  if (!c) return true;
  const url = c.req.url;
  return !url.includes("localhost") && !url.includes("127.0.0.1");
}

/**
 * Creates a standardized error response.
 *
 * Error messages are sanitized in production (5xx errors) to prevent information leakage.
 * All errors follow a consistent format: `{ error: string }`
 *
 * @param status - HTTP status code (400-599)
 * @param message - Error message (sanitized in production for 5xx errors)
 * @param c - Optional Hono context for environment detection
 * @returns JSON error response
 *
 * @example
 * ```typescript
 * return errorResponse(400, "Invalid submission_id format", c);
 * return errorResponse(500, "Database connection failed", c); // Sanitized in production
 * ```
 */
export function errorResponse(
  status: number,
  message: string,
  c?: Context<{ Bindings: Env }> | Context,
): Response {
  const shouldSanitize = status >= SERVER_ERROR_THRESHOLD && isProduction(c);
  const safeMessage = shouldSanitize ? SANITIZED_ERROR_MESSAGE : message;

  return new Response(JSON.stringify({ error: safeMessage }), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
