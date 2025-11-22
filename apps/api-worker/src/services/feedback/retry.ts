/**
 * Retry logic for feedback generation
 */

import type { CombinedFeedback } from "./types";
import { getCombinedFeedback } from "./combined";

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export async function getCombinedFeedbackWithRetry(
  params: {
    llmProvider: any;
    apiKey: string;
    questionText: string;
    answerText: string;
    modelName: string;
    essayScores?: any;
    languageToolErrors?: any[];
    llmErrors?: any[];
    relevanceCheck?: any;
  },
  options?: { maxAttempts?: number; baseDelayMs?: number }
): Promise<CombinedFeedback> {
  const maxAttempts = options?.maxAttempts ?? 3;
  const baseDelayMs = options?.baseDelayMs ?? 300;
  let attempt = 0;
  let lastError: Error | undefined;

  while (attempt < maxAttempts) {
    attempt += 1;
    try {
      const feedback = await getCombinedFeedback(
        params.llmProvider,
        params.apiKey,
        params.questionText,
        params.answerText,
        params.modelName,
        params.essayScores,
        params.languageToolErrors,
        params.llmErrors,
        params.relevanceCheck
      );

      if (!feedback?.detailed || !feedback?.teacher) {
        throw new Error("Combined feedback returned incomplete data");
      }

      return feedback;
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      if (attempt < maxAttempts) {
        const delay = baseDelayMs * attempt;
        await sleep(delay);
      }
    }
  }

  throw new Error(
    `Failed after ${maxAttempts} attempts${lastError ? `: ${lastError.message}` : ""}`
  );
}
