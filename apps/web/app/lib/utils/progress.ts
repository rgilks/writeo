import type { DraftHistory, ProgressMetrics } from "../stores/draft-store";
import type { LanguageToolError } from "@writeo/shared";

/**
 * Calculate error reduction between two drafts
 */
export function calculateErrorReduction(
  previousDraft: DraftHistory | null,
  currentDraft: DraftHistory
): number | null {
  if (
    !previousDraft ||
    previousDraft.errorCount === undefined ||
    currentDraft.errorCount === undefined
  ) {
    return null;
  }
  return previousDraft.errorCount - currentDraft.errorCount;
}

/**
 * Calculate score improvement between two drafts
 */
export function calculateScoreImprovement(
  previousDraft: DraftHistory | null,
  currentDraft: DraftHistory
): number | null {
  if (
    !previousDraft ||
    previousDraft.overallScore === undefined ||
    currentDraft.overallScore === undefined
  ) {
    return null;
  }
  return currentDraft.overallScore - previousDraft.overallScore;
}

/**
 * Calculate word count change between two drafts
 */
export function calculateWordCountChange(
  previousDraft: DraftHistory | null,
  currentDraft: DraftHistory
): number | null {
  if (
    !previousDraft ||
    previousDraft.wordCount === undefined ||
    currentDraft.wordCount === undefined
  ) {
    return null;
  }
  return currentDraft.wordCount - previousDraft.wordCount;
}

/**
 * Analyze error type frequency from an array of errors
 */
export function analyzeErrorTypeFrequency(
  errors: LanguageToolError[]
): Array<{ type: string; count: number }> {
  const frequency = new Map<string, number>();

  errors.forEach((error) => {
    const errorType = error.errorType || error.category || "Other";
    frequency.set(errorType, (frequency.get(errorType) || 0) + 1);
  });

  return Array.from(frequency.entries())
    .map(([type, count]) => ({ type, count }))
    .sort((a, b) => b.count - a.count);
}

/**
 * Get top error types (most frequent)
 */
export function getTopErrorTypes(
  errors: LanguageToolError[],
  limit: number = 3
): Array<{ type: string; count: number }> {
  return analyzeErrorTypeFrequency(errors).slice(0, limit);
}

/**
 * Calculate progress metrics from draft history
 */
export function calculateProgressMetrics(draftHistory: DraftHistory[]): ProgressMetrics | null {
  if (draftHistory.length === 0) {
    return null;
  }

  const firstDraft = draftHistory[0];
  const latestDraft = draftHistory[draftHistory.length - 1];

  return {
    totalDrafts: draftHistory.length,
    firstDraftScore: firstDraft.overallScore,
    latestDraftScore: latestDraft.overallScore,
    scoreImprovement:
      latestDraft.overallScore !== undefined && firstDraft.overallScore !== undefined
        ? latestDraft.overallScore - firstDraft.overallScore
        : undefined,
    errorReduction:
      latestDraft.errorCount !== undefined && firstDraft.errorCount !== undefined
        ? firstDraft.errorCount - latestDraft.errorCount
        : undefined,
    wordCountChange:
      latestDraft.wordCount !== undefined && firstDraft.wordCount !== undefined
        ? latestDraft.wordCount - firstDraft.wordCount
        : undefined,
  };
}

/**
 * Generate a unique error ID from error properties
 */
export function generateErrorId(error: LanguageToolError, text: string): string {
  // Use start/end positions and message to create a unique ID
  return `${error.start}-${error.end}-${error.message?.slice(0, 20) || ""}`;
}

/**
 * Extract error IDs from an array of errors
 */
export function extractErrorIds(errors: LanguageToolError[], text: string): string[] {
  return errors.map((error) => generateErrorId(error, text));
}
