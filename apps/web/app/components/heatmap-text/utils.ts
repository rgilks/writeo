/**
 * Utility functions for error filtering and processing
 */

import type { LanguageToolError } from "@writeo/shared";

export function filterErrorsByConfidence(errors: LanguageToolError[]) {
  const highConfidenceErrors = errors.filter((e) => e.highConfidence === true);
  const mediumConfidenceErrors = errors.filter((e) => e.mediumConfidence === true);
  const lowConfidenceErrors = errors.filter(
    (e) => e.highConfidence === false && e.mediumConfidence !== true
  );

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
  showExperimental: boolean
) {
  let filteredErrors = [...highConfidenceErrors];
  if (showMediumConfidence) {
    filteredErrors = [...filteredErrors, ...mediumConfidenceErrors];
  }
  if (showExperimental) {
    filteredErrors = [...filteredErrors, ...lowConfidenceErrors];
  }
  return filteredErrors;
}

export function validateError(error: LanguageToolError, textLength: number): boolean {
  return (
    !!error &&
    typeof error.start === "number" &&
    typeof error.end === "number" &&
    error.start >= 0 &&
    error.end <= textLength &&
    error.start < error.end
  );
}

export function scoreError(error: LanguageToolError): number {
  let score = 0;
  if (error.highConfidence === true) score += 100;
  else if (error.mediumConfidence === true) score += 50;
  else score += 10;

  if (error.explanation) score += 20;
  if (error.example) score += 15;
  if (error.errorType) score += 10;
  if (error.suggestions && error.suggestions.length > 0) score += 5;

  return score;
}

export function deduplicateErrors(errors: LanguageToolError[]): LanguageToolError[] {
  const errorMap = new Map<string, LanguageToolError>();
  for (const error of errors) {
    const key = `${error.start}-${error.end}`;
    const existing = errorMap.get(key);

    if (!existing) {
      errorMap.set(key, error);
    } else {
      const existingScore = scoreError(existing);
      const currentScore = scoreError(error);
      if (currentScore > existingScore) {
        errorMap.set(key, error);
      }
    }
  }
  return Array.from(errorMap.values()).sort((a, b) => a.start - b.start);
}

export function getErrorColor(
  isExperimental: boolean,
  isMediumConfidence: boolean,
  severity: string
): string {
  if (isExperimental) return "#d97706";
  if (isMediumConfidence) return "#ea580c";
  if (severity === "error") return "#dc2626";
  return "#d97706";
}
