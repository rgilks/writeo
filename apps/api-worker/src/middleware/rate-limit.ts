import type { Context } from "hono";
import { errorResponse } from "../utils/errors";
import { safeLogError, sanitizeError } from "../utils/logging";
import { KEY_OWNER, isPublicPath } from "../utils/constants";

// Rate limit configuration
const RATE_LIMITS = {
  general: { prod: 30, test: 1000 },
  results: { prod: 60, test: 2000 },
  submissions: { prod: 10, test: 500 },
  writes: { prod: 30, test: 1000 },
} as const;

const DAILY_SUBMISSION_LIMIT = 100; // Per user/IP per day
const RATE_LIMIT_WINDOW_MS = 60 * 1000;
const MIN_KV_TTL_SECONDS = 60; // Cloudflare KV minimum TTL
const DAILY_LIMIT_TTL_SECONDS = 86400;

interface RateLimitData {
  count: number;
  resetTime: number;
}

type LimitType = keyof typeof RATE_LIMITS;

interface RateLimitConfig {
  maxRequests: number;
  limitType: LimitType;
  checkDailyLimit: boolean;
}

/**
 * Determines rate limit configuration based on request path and method.
 */
function getRateLimitConfig(path: string, method: string, isTest: boolean): RateLimitConfig {
  if (path.startsWith("/text/submissions/") && path.includes("/results") && method === "GET") {
    return {
      maxRequests: isTest ? RATE_LIMITS.results.test : RATE_LIMITS.results.prod,
      limitType: "results",
      checkDailyLimit: false,
    };
  }

  if (path.startsWith("/text/submissions/") && method === "PUT") {
    return {
      maxRequests: isTest ? RATE_LIMITS.submissions.test : RATE_LIMITS.submissions.prod,
      limitType: "submissions",
      checkDailyLimit: true,
    };
  }

  if (path.startsWith("/text/questions/")) {
    return {
      maxRequests: isTest ? RATE_LIMITS.writes.test : RATE_LIMITS.writes.prod,
      limitType: "writes",
      checkDailyLimit: false,
    };
  }

  return {
    maxRequests: isTest ? RATE_LIMITS.general.test : RATE_LIMITS.general.prod,
    limitType: "general",
    checkDailyLimit: false,
  };
}

/**
 * Gets the identifier for rate limiting.
 * Uses IP for shared keys (admin/unknown), owner ID for user-specific keys.
 */
function getRateLimitIdentifier(apiKeyOwner: string, ip: string): string {
  return apiKeyOwner === KEY_OWNER.ADMIN || apiKeyOwner === KEY_OWNER.UNKNOWN ? ip : apiKeyOwner;
}

async function getRateLimitState(
  kvStore: KVNamespace,
  rateLimitKey: string,
  now: number,
): Promise<{ count: number; resetTime: number }> {
  const current = await kvStore.get(rateLimitKey);
  if (!current) {
    return { count: 0, resetTime: now + RATE_LIMIT_WINDOW_MS };
  }

  try {
    const data: RateLimitData = JSON.parse(current);
    if (data.resetTime > now) {
      return { count: data.count, resetTime: data.resetTime };
    }
  } catch (error) {
    safeLogError("Failed to parse rate limit data from KV", error);
  }

  return { count: 0, resetTime: now + RATE_LIMIT_WINDOW_MS };
}

function setRateLimitHeaders(
  c: Context,
  maxRequests: number,
  count: number,
  resetTime: number,
): void {
  c.header("X-RateLimit-Limit", String(maxRequests));
  c.header("X-RateLimit-Remaining", String(Math.max(0, maxRequests - count)));
  c.header("X-RateLimit-Reset", String(Math.ceil(resetTime / 1000)));
}

/**
 * Checks daily submission limit and returns the result.
 * @returns Object with `exceeded` boolean and current `count`
 */
async function checkDailyLimit(
  kvStore: KVNamespace,
  identifier: string,
  isTest: boolean,
): Promise<{ exceeded: boolean; count: number }> {
  if (isTest) {
    return { exceeded: false, count: 0 };
  }

  const today = new Date().toISOString().split("T")[0];
  const dailyLimitKey = `rate_limit:prod:daily_submissions:${today}:${identifier}`;

  const dailyCurrent = await kvStore.get(dailyLimitKey);
  const dailyCount = dailyCurrent ? parseInt(dailyCurrent, 10) : 0;

  if (dailyCount >= DAILY_SUBMISSION_LIMIT) {
    return { exceeded: true, count: dailyCount };
  }

  await kvStore.put(dailyLimitKey, String(dailyCount + 1), {
    expirationTtl: DAILY_LIMIT_TTL_SECONDS,
  });

  return { exceeded: false, count: dailyCount + 1 };
}

export async function rateLimit(c: Context, next: () => Promise<void>) {
  const path = new URL(c.req.url).pathname;

  if (isPublicPath(path)) {
    return next();
  }

  const isTest = (c.get("isTestKey") as boolean) || false;
  if (isTest) {
    return next();
  }
  const apiKeyOwner = (c.get("apiKeyOwner") as string) || KEY_OWNER.UNKNOWN;

  const config = getRateLimitConfig(path, c.req.method, isTest);

  const ip = c.req.header("CF-Connecting-IP") || c.req.header("X-Forwarded-For") || "unknown";
  const identifier = getRateLimitIdentifier(apiKeyOwner, ip);

  const keyPrefix = isTest ? "test" : "prod";
  const rateLimitKey = `rate_limit:${keyPrefix}:${config.limitType}:${identifier}`;
  const now = Date.now();

  try {
    const { count, resetTime } = await getRateLimitState(c.env.WRITEO_RESULTS, rateLimitKey, now);

    if (count >= config.maxRequests) {
      const message =
        config.limitType === "submissions"
          ? "Too many essay submissions from this network. Please wait a moment before trying again."
          : "Too many requests. Please wait a moment and try again.";

      setRateLimitHeaders(c, config.maxRequests, count, resetTime);
      return errorResponse(429, message, c);
    }

    if (config.checkDailyLimit) {
      const dailyCheck = await checkDailyLimit(c.env.WRITEO_RESULTS, identifier, isTest);
      if (dailyCheck.exceeded) {
        return errorResponse(
          429,
          "Daily submission limit reached for this account. Please try again tomorrow.",
          c,
        );
      }
    }

    const newCount = count + 1;
    const ttlSeconds = Math.max(MIN_KV_TTL_SECONDS, Math.ceil((resetTime - now) / 1000));
    await c.env.WRITEO_RESULTS.put(rateLimitKey, JSON.stringify({ count: newCount, resetTime }), {
      expirationTtl: ttlSeconds,
    });

    setRateLimitHeaders(c, config.maxRequests, newCount, resetTime);

    return next();
  } catch (error) {
    const sanitized = sanitizeError(error);
    safeLogError("Rate limiting error", sanitized);
    // Fail open to avoid blocking legitimate traffic
    return next();
  }
}
