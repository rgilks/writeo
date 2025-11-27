/**
 * Context utilities for route handlers
 * Provides common service instantiation to reduce duplication
 */

import type { Context } from "hono";
import type { Env } from "../types/env";
import { buildConfig } from "../services/config";
import { StorageService } from "../services/storage";

/**
 * Gets initialized services for a request context.
 * Reduces duplication across route handlers.
 */
export function getServices(c: Context<{ Bindings: Env; Variables: { requestId?: string } }>) {
  const config = buildConfig(c.env);
  return {
    config,
    storage: new StorageService(config.storage.r2Bucket, config.storage.kvNamespace),
  };
}
