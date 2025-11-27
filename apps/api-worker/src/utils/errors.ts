import type { Context } from "hono";
import type { Env } from "../types/env";

const SERVER_ERROR_THRESHOLD = 500;
const SANITIZED_ERROR_MESSAGE = "An internal error occurred. Please try again later.";

/**
 * Detects production environment using ENVIRONMENT environment variable.
 * Defaults to production when context is unavailable (fail-safe).
 *
 * @param c - Optional Hono context with environment bindings
 * @returns true if production, false otherwise
 */
function isProduction(c?: Context<{ Bindings: Env }> | Context): boolean {
  if (!c) return true;

  // Check environment variable if available
  if ("env" in c && c.env && typeof c.env === "object") {
    const env = c.env as { ENVIRONMENT?: string };
    // Only use ENVIRONMENT env var - no URL fallback
    return env.ENVIRONMENT !== "development" && env.ENVIRONMENT !== "staging";
  }

  // Default to production when context/env is unavailable (fail-safe)
  return true;
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
