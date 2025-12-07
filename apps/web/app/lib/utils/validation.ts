/**
 * Validation utilities for form inputs and essay submissions
 */

export interface ValidationResult {
  isValid: boolean;
  error: string | null;
}

/**
 * Validates essay answer text
 */
export function validateEssayAnswer(answer: string): ValidationResult {
  if (!answer.trim()) {
    return {
      isValid: false,
      error: "Please write your essay before submitting. Add your answer to receive feedback.",
    };
  }
  return { isValid: true, error: null };
}

/**
 * Validates word count against minimum and maximum requirements
 */
export function validateWordCount(
  wordCount: number,
  minWords: number,
  maxWords: number,
): ValidationResult {
  if (wordCount < minWords) {
    return {
      isValid: false,
      error: `Your essay is too short. Please write at least ${minWords} words (currently ${wordCount} words).`,
    };
  }

  if (wordCount > maxWords) {
    return {
      isValid: false,
      error: `Your essay is too long. Please keep it under ${maxWords} words (currently ${wordCount} words).`,
    };
  }

  return { isValid: true, error: null };
}

/**
 * Validates assessment results structure
 */
export function validateAssessmentResults(results: unknown): {
  isValid: boolean;
  error: string | null;
} {
  if (
    typeof results !== "object" ||
    results === null ||
    !("status" in results) ||
    !("requestedAssessors" in results)
  ) {
    return {
      isValid: false,
      error: "Invalid results format: missing required fields (status, requestedAssessors)",
    };
  }

  return { isValid: true, error: null };
}

/**
 * Validates submission response
 */
export function validateSubmissionResponse(response: {
  submissionId?: string;
  results?: unknown;
}): ValidationResult {
  if (!response.submissionId || !response.results) {
    return {
      isValid: false,
      error: "No submission ID or results returned",
    };
  }

  return validateAssessmentResults(response.results);
}
