/**
 * Retry logic for feedback generation
 */

import type { CombinedFeedback } from "./types";
import { getCombinedFeedback } from "./combined";
import type { LLMProvider } from "../llm";
import type { LanguageToolError } from "@writeo/shared";
import { retryWithBackoff } from "@writeo/shared";

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
  options?: { maxAttempts?: number; baseDelayMs?: number },
): Promise<CombinedFeedback> {
  return retryWithBackoff(
    async () => {
      const feedback = await getCombinedFeedback(
        params.llmProvider,
        params.apiKey,
        params.questionText,
        params.answerText,
        params.modelName,
        params.essayScores,
        params.languageToolErrors,
        params.llmErrors,
        params.relevanceCheck,
      );

      if (!feedback?.detailed || !feedback?.teacher) {
        throw new Error("Combined feedback returned incomplete data");
      }

      return feedback;
    },
    {
      maxAttempts: options?.maxAttempts ?? 3,
      baseDelayMs: options?.baseDelayMs ?? 300,
    },
  );
}
