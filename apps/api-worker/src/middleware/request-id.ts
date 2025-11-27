/**
 * Request ID middleware
 * Generates a unique ID for each request and stores it in context.
 * This ID is used for tracing requests across logs and services.
 */

import type { Context, Next } from "hono";
import type { Env } from "../types/env";

/**
 * Generates a unique request ID and stores it in context.
 * The request ID is available via `c.get('requestId')` in all handlers.
 */
export async function requestId(
  c: Context<{ Bindings: Env; Variables: { requestId?: string } }>,
  next: Next,
) {
  // Generate a short, unique request ID (first 8 chars of UUID)
  const id = crypto.randomUUID().split("-")[0];
  c.set("requestId", id);
  await next();
}
