import type { Context } from "hono";
import type { Env } from "../types/env";
import { errorResponse } from "./errors";
import { safeLogError, sanitizeError } from "./logging";

export function withErrorHandling(
  handler: (c: Context<{ Bindings: Env; Variables: { requestId?: string } }>) => Promise<Response>,
  logContext: string,
) {
  return async (c: Context<{ Bindings: Env; Variables: { requestId?: string } }>) => {
    try {
      return await handler(c);
    } catch (error) {
      const sanitized = sanitizeError(error);
      safeLogError(logContext, sanitized, c);
      return errorResponse(500, "Internal server error", c);
    }
  };
}
