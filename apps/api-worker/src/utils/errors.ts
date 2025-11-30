import type { Context } from "hono";
import type { Env } from "../types/env";

const SERVER_ERROR_THRESHOLD = 500;
const SANITIZED_ERROR_MESSAGE = "An internal error occurred. Please try again later.";

// Error codes for structured error responses
export const ERROR_CODES = {
  // Validation errors (400)
  INVALID_UUID_FORMAT: "INVALID_UUID_FORMAT",
  INVALID_SUBMISSION_FORMAT: "INVALID_SUBMISSION_FORMAT",
  INVALID_QUESTION_FORMAT: "INVALID_QUESTION_FORMAT",
  MISSING_REQUIRED_FIELD: "MISSING_REQUIRED_FIELD",
  INVALID_FIELD_VALUE: "INVALID_FIELD_VALUE",
  DUPLICATE_ANSWER_ID: "DUPLICATE_ANSWER_ID",
  CONFLICTING_QUESTION_TEXT: "CONFLICTING_QUESTION_TEXT",
  PAYLOAD_TOO_LARGE: "PAYLOAD_TOO_LARGE",

  // Not found errors (404)
  SUBMISSION_NOT_FOUND: "SUBMISSION_NOT_FOUND",
  QUESTION_NOT_FOUND: "QUESTION_NOT_FOUND",
  ANSWER_NOT_FOUND: "ANSWER_NOT_FOUND",

  // Conflict errors (409)
  QUESTION_EXISTS_DIFFERENT_CONTENT: "QUESTION_EXISTS_DIFFERENT_CONTENT",
  SUBMISSION_EXISTS_DIFFERENT_CONTENT: "SUBMISSION_EXISTS_DIFFERENT_CONTENT",

  // Server errors (500)
  INTERNAL_SERVER_ERROR: "INTERNAL_SERVER_ERROR",
  ASSESSMENT_FAILED: "ASSESSMENT_FAILED",
  STORAGE_ERROR: "STORAGE_ERROR",
} as const;

export type ErrorCode = (typeof ERROR_CODES)[keyof typeof ERROR_CODES];

export interface StructuredError {
  code: ErrorCode;
  message: string;
  field?: string;
  requestId?: string;
}

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

function getRequestId(
  c?: Context<{ Bindings: Env; Variables: { requestId?: string } }>,
): string | undefined {
  if (!c) return undefined;
  return c.get("requestId") as string | undefined;
}

// Error messages are sanitized in production (5xx errors) to prevent information leakage
export function errorResponse(
  status: number,
  message: string,
  c?: Context<{ Bindings: Env; Variables: { requestId?: string } }> | Context,
  code?: ErrorCode,
  field?: string,
): Response {
  const shouldSanitize = status >= SERVER_ERROR_THRESHOLD && isProduction(c);
  const safeMessage = shouldSanitize ? SANITIZED_ERROR_MESSAGE : message;
  const requestId = getRequestId(c);

  const errorResponse: StructuredError = {
    code:
      code || (status >= 500 ? ERROR_CODES.INTERNAL_SERVER_ERROR : ERROR_CODES.INVALID_FIELD_VALUE),
    message: safeMessage,
    ...(field && { field }),
    ...(requestId && { requestId }),
  };

  return new Response(JSON.stringify({ error: errorResponse }), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
