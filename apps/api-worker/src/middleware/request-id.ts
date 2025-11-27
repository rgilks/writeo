import type { Context, Next } from "hono";
import type { Env } from "../types/env";

// Generates a unique request ID (first 8 chars of UUID) for tracing requests across logs
export async function requestId(
  c: Context<{ Bindings: Env; Variables: { requestId?: string } }>,
  next: Next,
) {
  // Generate a short, unique request ID (first 8 chars of UUID)
  const id = crypto.randomUUID().split("-")[0];
  c.set("requestId", id);
  await next();
}
