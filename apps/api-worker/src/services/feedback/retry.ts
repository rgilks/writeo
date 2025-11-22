/**
 * Retry logic for feedback generation
 */

import type { CombinedFeedback } from "./types";
import { getCombinedFeedback } from "./combined";
import type { LLMProvider } from "../llm";
import type { LanguageToolError } from "@writeo/shared";

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export interface FeedbackRetryParams {
  llmProvider: LLMProvider;
  apiKey: string;
  questionText: string;
  answerText: string;
  modelName: string;
  essayScores?: {
    overall?: number;
    dimensions?: { TA?: number; CC?: number; Vocab?: number; Grammar?: number; Overall?: number };
    label?: string;
  };
  languageToolErrors?: LanguageToolError[];
  llmErrors?: LanguageToolError[];
  relevanceCheck?: { addressesQuestion: boolean; score: number; threshold: number };
}

export async function getCombinedFeedbackWithRetry(
  params: FeedbackRetryParams,
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
