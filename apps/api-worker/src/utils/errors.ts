import type { Context } from "hono";
import type { Env } from "../types/env";

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
  c?: Context<{ Bindings: Env }> | Context
): Response {
  const isProduction = c
    ? !c.req.url.includes("localhost") && !c.req.url.includes("127.0.0.1")
    : true;
  const safeMessage =
    status >= 500 && isProduction ? "An internal error occurred. Please try again later." : message;

  return new Response(JSON.stringify({ error: safeMessage }), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
