import type { Context } from "hono";

export async function securityHeaders(c: Context, next: () => Promise<void>) {
  await next();
  c.header("X-Content-Type-Options", "nosniff");
  c.header("X-Frame-Options", "DENY");
  c.header("X-XSS-Protection", "1; mode=block");
  c.header("Referrer-Policy", "strict-origin-when-cross-origin");
}

function getAllowedOrigin(
  origin: string | null,
  allowedOrigins?: string
): string | null | undefined {
  if (!origin) {
    return origin;
  }

  // If ALLOWED_ORIGINS is not set, allow all origins
  // This is acceptable for a public API secured by token authentication.
  // The API is designed to be used from any origin - security is provided
  // by the required API key authentication, not by CORS restrictions.
  if (!allowedOrigins) {
    return origin; // Allow all origins when not configured
  }

  // If ALLOWED_ORIGINS is set, validate against the whitelist
  const allowed = allowedOrigins.split(",").map((o: string) => o.trim());
  return allowed.includes(origin) ? origin : null;
}

export function getCorsOrigin(origin: string | null, allowedOrigins?: string) {
  return getAllowedOrigin(origin, allowedOrigins);
}
