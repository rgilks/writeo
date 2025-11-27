import type { Context } from "hono";

// Security headers to apply to all responses
const SECURITY_HEADERS = {
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY",
  "X-XSS-Protection": "1; mode=block",
  "Referrer-Policy": "strict-origin-when-cross-origin",
} as const;

/**
 * Middleware that adds security headers to all responses.
 */
export async function securityHeaders(c: Context, next: () => Promise<void>) {
  await next();
  for (const [key, value] of Object.entries(SECURITY_HEADERS)) {
    c.header(key, value);
  }
}

/**
 * Determines the allowed CORS origin for a request.
 *
 * If ALLOWED_ORIGINS is not configured, all origins are allowed.
 * This is acceptable for a public API secured by token authentication.
 * Security is provided by the required API key authentication, not by CORS restrictions.
 *
 * If ALLOWED_ORIGINS is configured, only origins in the whitelist are allowed.
 *
 * @param origin - The origin from the request header (may be null)
 * @param allowedOrigins - Comma-separated list of allowed origins (optional)
 * @returns The origin if allowed, null if not allowed, or undefined if origin is null
 */
export function getCorsOrigin(
  origin: string | null,
  allowedOrigins?: string,
): string | null | undefined {
  if (!origin) {
    return origin;
  }

  if (!allowedOrigins) {
    return origin;
  }

  const allowedList = allowedOrigins.split(",").map((o) => o.trim());
  return allowedList.includes(origin) ? origin : null;
}
