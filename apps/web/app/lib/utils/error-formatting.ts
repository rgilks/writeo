/**
 * Error formatting utilities for user-friendly error messages
 */

import { getErrorMessage, DEFAULT_ERROR_MESSAGES } from "./error-messages";

/**
 * Formats an unknown error into a user-friendly message
 */
export function formatFriendlyErrorMessage(
  error: unknown,
  context: keyof typeof DEFAULT_ERROR_MESSAGES = "global",
): string {
  if (error instanceof Error) {
    return getErrorMessage(error, context);
  }

  if (
    typeof error === "string" &&
    error.length < 200 &&
    !error.includes("Error:") &&
    !error.includes("at ")
  ) {
    return error;
  }

  return getErrorMessage(new Error("Submission failed"), context);
}
