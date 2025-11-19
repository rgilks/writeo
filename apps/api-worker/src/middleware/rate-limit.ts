import type { Context } from "hono";
import { errorResponse } from "../utils/errors";
import { safeLogError, sanitizeError } from "../utils/logging";

/**
 * Check if the request is using a test API key
 * Test API keys get higher rate limits for automated testing
 */
function isTestApiKey(c: Context): boolean {
  const testApiKey = (c.env as any).TEST_API_KEY;
  if (!testApiKey) {
    return false;
  }

  const authHeader = c.req.header("Authorization");
  if (!authHeader) {
    return false;
  }

  const match = authHeader.match(/^Token\s+(.+)$/);
  if (!match) {
    return false;
  }

  const providedKey = match[1];
  return providedKey === testApiKey;
}

export async function rateLimit(c: Context, next: () => Promise<void>) {
  const path = new URL(c.req.url).pathname;
  if (path === "/health" || path === "/docs" || path === "/openapi.json") {
    return next();
  }

  // Check if this is a test API key request
  const isTest = isTestApiKey(c);

  // Rate limiting applies to all requests
  // Test API keys get much higher limits for automated testing
  let maxRequests = isTest ? 1000 : 30; // Test keys: 1000/min, Production: 30/min
  let limitType = "general";

  if (
    path.startsWith("/text/submissions/") &&
    path.includes("/results") &&
    c.req.method === "GET"
  ) {
    maxRequests = isTest ? 2000 : 60; // Test keys: 2000/min, Production: 60/min
    limitType = "results";
  } else if (path.startsWith("/text/submissions/") && c.req.method === "PUT") {
    // Limit submissions to 10/min to keep costs under control
    // 10/min = 14,400/day max × $0.03/submission = ~$432/day max (but rate limiting prevents constant usage)
    maxRequests = isTest ? 500 : 10; // Test keys: 500/min, Production: 10/min
    limitType = "submissions";
  } else if (path.startsWith("/text/questions/")) {
    maxRequests = isTest ? 1000 : 30; // Test keys: 1000/min, Production: 30/min
    limitType = "writes";
  }

  const ip = c.req.header("CF-Connecting-IP") || c.req.header("X-Forwarded-For") || "unknown";
  // Separate rate limit buckets for test vs production keys
  const keyPrefix = isTest ? "test" : "prod";
  const rateLimitKey = `rate_limit:${keyPrefix}:${limitType}:${ip}`;
  const now = Date.now();
  const windowMs = 60 * 1000;

  try {
    const current = await c.env.WRITEO_RESULTS.get(rateLimitKey);
    let count = 0;
    let resetTime = now + windowMs;

    if (current) {
      const data = JSON.parse(current);
      if (data.resetTime > now) {
        count = data.count;
        resetTime = data.resetTime;
      }
    }

    if (count >= maxRequests) {
      const secondsUntilReset = Math.ceil((resetTime - now) / 1000);
      const friendlyMessage =
        limitType === "submissions"
          ? `Too many essay submissions from this network. Please wait ${secondsUntilReset} second${secondsUntilReset !== 1 ? "s" : ""} before trying again.`
          : `Too many requests. Please wait ${secondsUntilReset} second${secondsUntilReset !== 1 ? "s" : ""} and try again.`;

      c.header("X-RateLimit-Limit", String(maxRequests));
      c.header("X-RateLimit-Remaining", "0");
      c.header("X-RateLimit-Reset", String(Math.ceil(resetTime / 1000)));
      return errorResponse(429, friendlyMessage, c);
    }

    count++;
    // Cloudflare KV requires minimum TTL of 60 seconds
    const ttlSeconds = Math.max(60, Math.ceil((resetTime - now) / 1000));
    await c.env.WRITEO_RESULTS.put(rateLimitKey, JSON.stringify({ count, resetTime }), {
      expirationTtl: ttlSeconds,
    });

    c.header("X-RateLimit-Limit", String(maxRequests));
    c.header("X-RateLimit-Remaining", String(maxRequests - count));
    c.header("X-RateLimit-Reset", String(Math.ceil(resetTime / 1000)));

    return next();
  } catch (error) {
    const sanitized = sanitizeError(error);
    safeLogError("Rate limiting error", sanitized);
    return next();
  }
}
