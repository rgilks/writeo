/**
 * Utilities for processing essay submissions and results
 */

import type { AssessmentResults } from "@writeo/shared";

/**
 * Merges question text into assessment results metadata
 */
export function mergeQuestionTextIntoResults(
  results: AssessmentResults,
  questionText: string,
): AssessmentResults {
  if (typeof window === "undefined" || !questionText) {
    return results;
  }

  const answerTexts = results.meta?.answerTexts as Record<string, string> | undefined;
  const answerId = answerTexts ? Object.keys(answerTexts)[0] : undefined;

  if (!answerId) {
    return results;
  }

  // If questionTexts doesn't exist, create it
  if (!results.meta?.questionTexts) {
    return {
      ...results,
      meta: {
        ...results.meta,
        questionTexts: {
          [answerId]: questionText,
        },
      },
    };
  }

  // If questionTexts exists but doesn't have this answerId, add it
  const existingQuestionTexts = results.meta.questionTexts as Record<string, string>;
  if (!existingQuestionTexts[answerId]) {
    return {
      ...results,
      meta: {
        ...results.meta,
        questionTexts: {
          ...existingQuestionTexts,
          [answerId]: questionText,
        },
      },
    };
  }

  // Already has the question text, return as-is
  return results;
}
