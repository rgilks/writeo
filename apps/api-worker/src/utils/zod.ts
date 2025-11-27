import { z, type ZodError } from "zod";

/**
 * Creates a Zod schema for UUID string validation with custom error message.
 *
 * @param fieldName - Field name used in error message
 * @returns Zod string schema with UUID validation
 *
 * @example
 * ```typescript
 * const schema = uuidStringSchema("submission_id");
 * schema.safeParse("123e4567-e89b-12d3-a456-426614174000");
 * ```
 */
export function uuidStringSchema(fieldName: string) {
  return z.string().uuid(`Invalid ${fieldName} format`);
}

/**
 * Extracts the first error message from a ZodError, with fallback.
 *
 * @param error - ZodError to extract message from
 * @param fallbackMessage - Default message if no issues found
 * @returns Error message string
 *
 * @example
 * ```typescript
 * if (!result.success) {
 *   formatZodMessage(result.error, "Validation failed");
 * }
 * ```
 */
export function formatZodMessage(error: ZodError, fallbackMessage: string): string {
  return error.issues[0]?.message ?? fallbackMessage;
}
