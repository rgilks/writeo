/**
 * Metadata building utilities
 */

import type { AssessmentResults, LanguageToolError } from "@writeo/shared";
import { getEssayAssessorResult, countWords } from "@writeo/shared";

function sumErrorCounts(...errorMaps: Map<string, LanguageToolError[]>[]): number {
  let total = 0;
  for (const errorsByAnswerId of errorMaps) {
    for (const errors of errorsByAnswerId.values()) {
      total += errors.length;
    }
  }
  return total;
}

function extractOverallScore(essayAssessment: AssessmentResults | null): number | undefined {
  const firstPart = essayAssessment?.results?.parts?.[0];
  const firstAnswer = firstPart?.answers?.[0];
  const assessorResults = firstAnswer?.assessorResults ?? [];
  const essayAssessor = getEssayAssessorResult(assessorResults);
  return essayAssessor?.overall;
}

export function buildMetadata(
  answerTextsByAnswerId: Map<string, string>,
  ltErrorsByAnswerId: Map<string, LanguageToolError[]>,
  llmErrorsByAnswerId: Map<string, LanguageToolError[]>,
  essayAssessment: AssessmentResults | null,
): {
  wordCount: number;
  errorCount: number;
  overallScore?: number;
  timestamp: string;
} {
  let totalWordCount = 0;
  for (const text of answerTextsByAnswerId.values()) {
    totalWordCount += countWords(text);
  }

  const totalErrorCount = sumErrorCounts(ltErrorsByAnswerId, llmErrorsByAnswerId);
  const overallScore = extractOverallScore(essayAssessment);

  return {
    wordCount: totalWordCount,
    errorCount: totalErrorCount,
    overallScore,
    timestamp: new Date().toISOString(),
  };
}

export function buildResponseHeaders(
  timings: Record<string, number>,
  requestId?: string,
): Record<string, string> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "X-Timing-Data": JSON.stringify(timings),
  };

  if (requestId) {
    headers["X-Request-Id"] = requestId;
  }

  const totalTime = timings["0_total"];
  if (typeof totalTime === "number") {
    headers["X-Timing-Total"] = totalTime.toFixed(2);
  }

  const slowestEntries = Object.entries(timings)
    .filter(([key]) => key !== "0_total")
    .sort(([, a], [, b]) => (b ?? 0) - (a ?? 0))
    .slice(0, 5)
    .map(([key, value]) => `${key}:${(value ?? 0).toFixed(2)}`)
    .filter(Boolean);

  if (slowestEntries.length > 0) {
    headers["X-Timing-Slowest"] = slowestEntries.join("; ");
  }

  return headers;
}
