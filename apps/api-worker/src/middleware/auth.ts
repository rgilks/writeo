import type { Context } from "hono";
import type { Env } from "../types/env";
import { errorResponse } from "../utils/errors";
import { safeLogError } from "../utils/logging";
import { KEY_OWNER, isPublicPath } from "../utils/constants";

// Authorization header format: "Token <key>"
const AUTH_HEADER_REGEX = /^Token\s+(.+)$/;

interface KeyInfo {
  owner?: string;
}

// Checks in order: admin key, test key, then user keys in KV store
async function validateApiKey(
  providedKey: string,
  adminKey: string,
  testKey: string | undefined,
  kvStore: KVNamespace,
): Promise<{ owner: string; isTestKey: boolean } | null> {
  if (providedKey === adminKey) {
    return { owner: KEY_OWNER.ADMIN, isTestKey: false };
  }

  if (testKey && providedKey === testKey) {
    return { owner: KEY_OWNER.TEST_RUNNER, isTestKey: true };
  }

  try {
    const keyInfoStr = await kvStore.get(`apikey:${providedKey}`);
    if (keyInfoStr) {
      try {
        const keyInfo: KeyInfo = JSON.parse(keyInfoStr);
        return {
          owner: keyInfo.owner || KEY_OWNER.UNKNOWN,
          isTestKey: false,
        };
      } catch (parseError) {
        safeLogError("Failed to parse API key info from KV", parseError, undefined);
        // Invalid JSON in KV - treat as invalid key
        return null;
      }
    }
  } catch (error) {
    // KV lookup failed (network error, etc.) - log but don't expose to user
    safeLogError("Error checking API key in KV store", error, undefined);
    return null;
  }

  return null;
}

export async function authenticate(
  c: Context<{
    Bindings: Env;
    Variables: { apiKeyOwner?: string; isTestKey?: boolean; requestId?: string };
  }>,
  next: () => Promise<void>,
) {
  const path = new URL(c.req.url).pathname;

  if (isPublicPath(path)) {
    return next();
  }

  const authHeader = c.req.header("Authorization");
  if (!authHeader) {
    return errorResponse(401, "Missing Authorization header", c);
  }

  const match = authHeader.match(AUTH_HEADER_REGEX);
  if (!match || !match[1]) {
    return errorResponse(401, "Invalid Authorization header format. Expected: 'Token <key>'", c);
  }

  const providedKey = match[1];
  const adminKey = c.env.API_KEY;

  if (!adminKey) {
    safeLogError("API_KEY not configured in environment", undefined, c);
    return errorResponse(500, "Server configuration error", c);
  }

  const testKey = c.env.TEST_API_KEY;
  const authResult = await validateApiKey(providedKey, adminKey, testKey, c.env.WRITEO_RESULTS);

  if (!authResult) {
    return errorResponse(401, "Invalid API key", c);
  }

  c.set("apiKeyOwner", authResult.owner);
  c.set("isTestKey", authResult.isTestKey);

  return next();
}
