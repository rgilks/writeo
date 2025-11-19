import type { Context } from "hono";
import type { Env } from "../types/env";

/**
 * Returns a safe error response that doesn't leak sensitive information in production.
 *
 * Security rules:
 * - 4xx errors: Return the message (client errors are safe to show)
 * - 5xx errors in production: Return generic message (don't expose internal details)
 * - 5xx errors in development: Return the actual message (for debugging)
 *
 * @param status HTTP status code
 * @param message Error message (will be sanitized for 5xx in production)
 * @param c Optional Hono context to detect production environment
 * @returns Response with sanitized error message
 */
export function errorResponse(
  status: number,
  message: string,
  c?: Context<{ Bindings: Env }> | Context<any>
): Response {
  // Detect production environment
  // In Cloudflare Workers, production is when URL doesn't include localhost
  // We can't rely on NODE_ENV in Workers environment
  const isProduction = c
    ? !c.req.url.includes("localhost") && !c.req.url.includes("127.0.0.1")
    : true;

  // For 5xx errors in production, always return a generic message
  // For 4xx errors, return the message (client errors are safe)
  const safeMessage =
    status >= 500 && isProduction ? "An internal error occurred. Please try again later." : message;

  return new Response(JSON.stringify({ error: safeMessage }), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
