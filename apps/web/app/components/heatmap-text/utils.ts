/**
 * Utility functions for error filtering and processing
 */

import type { LanguageToolError } from "@writeo/shared";

const ERROR_COLORS = {
  experimental: "#d97706",
  mediumConfidence: "#ea580c",
  error: "#dc2626",
} as const;

const SCORE_WEIGHTS = {
  highConfidence: 100,
  mediumConfidence: 50,
  lowConfidence: 10,
  explanation: 20,
  example: 15,
  errorType: 10,
  suggestions: 5,
} as const;

export function filterErrorsByConfidence(errors: LanguageToolError[]) {
  const highConfidenceErrors: LanguageToolError[] = [];
  const mediumConfidenceErrors: LanguageToolError[] = [];
  const lowConfidenceErrors: LanguageToolError[] = [];

  for (const error of errors) {
    if (error.highConfidence === true) {
      highConfidenceErrors.push(error);
    } else if (error.mediumConfidence === true) {
      mediumConfidenceErrors.push(error);
    } else {
      lowConfidenceErrors.push(error);
    }
  }

  return {
    highConfidenceErrors,
    mediumConfidenceErrors,
    lowConfidenceErrors,
  };
}

export function buildFilteredErrors(
  highConfidenceErrors: LanguageToolError[],
  mediumConfidenceErrors: LanguageToolError[],
  lowConfidenceErrors: LanguageToolError[],
  showMediumConfidence: boolean,
  showExperimental: boolean,
): LanguageToolError[] {
  return [
    ...highConfidenceErrors,
    ...(showMediumConfidence ? mediumConfidenceErrors : []),
    ...(showExperimental ? lowConfidenceErrors : []),
  ];
}

export function validateError(error: LanguageToolError, textLength: number): boolean {
  if (!error || typeof error.start !== "number" || typeof error.end !== "number") {
    return false;
  }
  return error.start >= 0 && error.end <= textLength && error.start < error.end;
}

export function scoreError(error: LanguageToolError): number {
  let score = 0;

  if (error.highConfidence === true) {
    score += SCORE_WEIGHTS.highConfidence;
  } else if (error.mediumConfidence === true) {
    score += SCORE_WEIGHTS.mediumConfidence;
  } else {
    score += SCORE_WEIGHTS.lowConfidence;
  }

  if (error.explanation) score += SCORE_WEIGHTS.explanation;
  if (error.example) score += SCORE_WEIGHTS.example;
  if (error.errorType) score += SCORE_WEIGHTS.errorType;
  if (error.suggestions?.length) score += SCORE_WEIGHTS.suggestions;

  return score;
}

export function deduplicateErrors(errors: LanguageToolError[]): LanguageToolError[] {
  const errorMap = new Map<string, LanguageToolError>();

  for (const error of errors) {
    const key = `${error.start}-${error.end}`;
    const existing = errorMap.get(key);

    if (!existing || scoreError(error) > scoreError(existing)) {
      errorMap.set(key, error);
    }
  }

  return Array.from(errorMap.values()).sort((a, b) => a.start - b.start);
}

export function getErrorColor(
  isExperimental: boolean,
  isMediumConfidence: boolean,
  severity: string,
): string {
  if (isExperimental) return ERROR_COLORS.experimental;
  if (isMediumConfidence) return ERROR_COLORS.mediumConfidence;
  if (severity === "error") return ERROR_COLORS.error;
  return ERROR_COLORS.experimental;
}
