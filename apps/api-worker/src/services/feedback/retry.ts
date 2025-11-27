/**
 * Retry logic for feedback generation
 */

import type { CombinedFeedback } from "./types";
import { getCombinedFeedback } from "./combined";
import { retryWithBackoff } from "@writeo/shared";
import type { CombinedFeedbackParams } from "./combined";

export type FeedbackRetryParams = CombinedFeedbackParams;

export async function getCombinedFeedbackWithRetry(
  params: FeedbackRetryParams,
  options?: { maxAttempts?: number; baseDelayMs?: number },
): Promise<CombinedFeedback> {
  return retryWithBackoff(
    async () => {
      const feedback = await getCombinedFeedback(params);

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
