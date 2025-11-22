/**
 * Metadata building utilities
 */

import type { AssessmentResults, LanguageToolError } from "@writeo/shared";

export function buildMetadata(
  answerTextsByAnswerId: Map<string, string>,
  ltErrorsByAnswerId: Map<string, LanguageToolError[]>,
  llmErrorsByAnswerId: Map<string, LanguageToolError[]>,
  essayAssessment: AssessmentResults | null
): {
  wordCount: number;
  errorCount: number;
  overallScore?: number;
  timestamp: string;
} {
  let totalWordCount = 0;
  let totalErrorCount = 0;

  for (const text of answerTextsByAnswerId.values()) {
    totalWordCount += text
      .trim()
      .split(/\s+/)
      .filter((w) => w.length > 0).length;
  }

  for (const errors of ltErrorsByAnswerId.values()) {
    totalErrorCount += errors.length;
  }
  for (const errors of llmErrorsByAnswerId.values()) {
    totalErrorCount += errors.length;
  }

  const firstPart = essayAssessment?.results?.parts?.[0];
  const overallScore = firstPart?.answers?.[0]?.["assessor-results"]?.find(
    (a: any) => a.id === "T-AES-ESSAY"
  )?.overall;

  return {
    wordCount: totalWordCount,
    errorCount: totalErrorCount,
    overallScore,
    timestamp: new Date().toISOString(),
  };
}

export function buildResponseHeaders(timings: Record<string, number>): Record<string, string> {
  const responseHeaders = new Headers();
  responseHeaders.set("Content-Type", "application/json");
  responseHeaders.set("X-Timing-Data", JSON.stringify(timings));
  responseHeaders.set("X-Timing-Total", timings["0_total"].toFixed(2));

  const sortedTimings = Object.entries(timings)
    .filter(([key]) => key !== "0_total")
    .sort(([, a], [, b]) => (b as number) - (a as number))
    .slice(0, 5);

  const slowestOps = sortedTimings
    .map(([key, value]) => `${key}:${(value as number).toFixed(2)}`)
    .join("; ");
  responseHeaders.set("X-Timing-Slowest", slowestOps);

  const headersObj: Record<string, string> = {};
  responseHeaders.forEach((value, key) => {
    headersObj[key] = value;
  });

  return headersObj;
}
