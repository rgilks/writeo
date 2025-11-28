import type { DraftHistory, ProgressMetrics } from "../stores/draft-store";
import type { LanguageToolError } from "@writeo/shared";
import { getErrorType } from "./error-utils";

function calculateDraftDifference<T extends number | undefined>(
  previousDraft: DraftHistory | null,
  currentDraft: DraftHistory,
  getValue: (draft: DraftHistory) => T,
  calculateDiff: (prev: number, curr: number) => number,
): number | null {
  if (!previousDraft) return null;

  const prevValue = getValue(previousDraft);
  const currValue = getValue(currentDraft);

  if (prevValue === undefined || currValue === undefined) {
    return null;
  }

  return calculateDiff(prevValue, currValue);
}

export function calculateErrorReduction(
  previousDraft: DraftHistory | null,
  currentDraft: DraftHistory,
): number | null {
  return calculateDraftDifference(
    previousDraft,
    currentDraft,
    (draft) => draft.errorCount,
    (prev, curr) => prev - curr,
  );
}

export function calculateScoreImprovement(
  previousDraft: DraftHistory | null,
  currentDraft: DraftHistory,
): number | null {
  return calculateDraftDifference(
    previousDraft,
    currentDraft,
    (draft) => draft.overallScore,
    (prev, curr) => curr - prev,
  );
}

export function calculateWordCountChange(
  previousDraft: DraftHistory | null,
  currentDraft: DraftHistory,
): number | null {
  return calculateDraftDifference(
    previousDraft,
    currentDraft,
    (draft) => draft.wordCount,
    (prev, curr) => curr - prev,
  );
}

export function analyzeErrorTypeFrequency(
  errors: LanguageToolError[],
): Array<{ type: string; count: number }> {
  const frequency = new Map<string, number>();

  errors.forEach((error) => {
    const errorType = getErrorType(error);
    frequency.set(errorType, (frequency.get(errorType) || 0) + 1);
  });

  return Array.from(frequency.entries())
    .map(([type, count]) => ({ type, count }))
    .sort((a, b) => b.count - a.count);
}

export function getTopErrorTypes(
  errors: LanguageToolError[],
  limit: number = 3,
): Array<{ type: string; count: number }> {
  return analyzeErrorTypeFrequency(errors).slice(0, limit);
}

function safeDifference(
  first: number | undefined,
  latest: number | undefined,
  calculate: (first: number, latest: number) => number,
): number | undefined {
  if (first === undefined || latest === undefined) {
    return undefined;
  }
  return calculate(first, latest);
}

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
    scoreImprovement: safeDifference(
      firstDraft.overallScore,
      latestDraft.overallScore,
      (first, latest) => latest - first,
    ),
    errorReduction: safeDifference(
      firstDraft.errorCount,
      latestDraft.errorCount,
      (first, latest) => first - latest,
    ),
    wordCountChange: safeDifference(
      firstDraft.wordCount,
      latestDraft.wordCount,
      (first, latest) => latest - first,
    ),
  };
}

export function generateErrorId(error: LanguageToolError): string {
  return `${error.start}-${error.end}-${error.message?.slice(0, 20) || ""}`;
}

export function extractErrorIds(errors: LanguageToolError[]): string[] {
  return errors.map((error) => generateErrorId(error));
}
