import type { Context } from "hono";
import type { Env } from "../types/env";

const SERVER_ERROR_THRESHOLD = 500;
const SANITIZED_ERROR_MESSAGE = "An internal error occurred. Please try again later.";

// Defaults to production when context is unavailable (fail-safe)
function isProduction(c?: Context<{ Bindings: Env }> | Context): boolean {
  if (!c) return true;

  // Check environment variable if available
  if ("env" in c && c.env && typeof c.env === "object") {
    const env = c.env as { ENVIRONMENT?: string };
    // Only use ENVIRONMENT env var - no URL fallback
    // If ENVIRONMENT is explicitly set to "development" or "staging", return false
    // Otherwise (including undefined), default to production (fail-safe)
    if (env.ENVIRONMENT === "development" || env.ENVIRONMENT === "staging") {
      return false;
    }
    return true;
  }

  // Default to production when context/env is unavailable (fail-safe)
  return true;
}

// Error messages are sanitized in production (5xx errors) to prevent information leakage
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
