/**
 * Validation utilities - shared between frontend and backend
 */

import { MIN_ESSAY_WORDS, MAX_ESSAY_WORDS } from "./constants";

export interface WordCountValidation {
  valid: boolean;
  error?: string;
}

/**
 * Validates word count against essay requirements
 *
 * @param wordCount - Number of words to validate
 * @param min - Minimum word count (default: MIN_ESSAY_WORDS)
 * @param max - Maximum word count (default: MAX_ESSAY_WORDS)
 * @returns Validation result with error message if invalid
 *
 * @example
 * ```typescript
 * const validation = validateWordCount(200);
 * if (!validation.valid) {
 *   throw new Error(validation.error);
 * }
 * ```
 */
export function validateWordCount(
  wordCount: number,
  min: number = MIN_ESSAY_WORDS,
  max: number = MAX_ESSAY_WORDS,
): WordCountValidation {
  if (wordCount < min) {
    return {
      valid: false,
      error: `Essay is too short. Please write at least ${min} words (currently ${wordCount} words).`,
    };
  }
  if (wordCount > max) {
    return {
      valid: false,
      error: `Essay is too long. Please keep it under ${max} words (currently ${wordCount} words).`,
    };
  }
  return { valid: true };
}
