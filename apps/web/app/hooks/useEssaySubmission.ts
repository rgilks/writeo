"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import type { AssessmentResults } from "@writeo/shared";
import { submitEssay } from "@/app/lib/actions";
import { useDraftStore } from "@/app/lib/stores/draft-store";
import { errorLogger } from "@/app/lib/utils/error-logger";
import {
  validateEssayAnswer,
  validateWordCount,
  validateSubmissionResponse,
} from "@/app/lib/utils/validation";
import { mergeQuestionTextIntoResults } from "@/app/lib/utils/submission";
import { formatFriendlyErrorMessage } from "@/app/lib/utils/error-formatting";

const SUBMISSION_TIMEOUT = 60000;

export function useEssaySubmission() {
  const router = useRouter();
  const setResult = useDraftStore((state) => state.setResult);
  const createNewContentDraft = useDraftStore((state) => state.createNewContentDraft);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = useCallback(
    async (
      questionText: string,
      answer: string,
      taskId: string,
      wordCount: number,
      minWords: number,
      maxWords: number,
      storeResults: boolean,
    ) => {
      // Validate answer
      const answerValidation = validateEssayAnswer(answer);
      if (!answerValidation.isValid) {
        setError(answerValidation.error!);
        return;
      }

      // Validate word count
      const wordCountValidation = validateWordCount(wordCount, minWords, maxWords);
      if (!wordCountValidation.isValid) {
        setError(wordCountValidation.error!);
        return;
      }

      setLoading(true);
      setError(null);

      try {
        const submitPromise = submitEssay(questionText, answer, undefined, storeResults);
        const timeoutPromise = new Promise<never>((_, reject) => {
          setTimeout(
            () => reject(new Error("Request timed out. Please try again.")),
            SUBMISSION_TIMEOUT,
          );
        });

        const response = await Promise.race([submitPromise, timeoutPromise]);

        // Log response for debugging
        console.log("[useEssaySubmission] Response received:", {
          hasSubmissionId: !!response.submissionId,
          hasResults: !!response.results,
          resultsType: typeof response.results,
          resultsKeys:
            response.results && typeof response.results === "object"
              ? Object.keys(response.results)
              : null,
        });

        // Validate submission response
        const responseValidation = validateSubmissionResponse(response);
        if (!responseValidation.isValid) {
          console.error("[useEssaySubmission] Validation failed:", responseValidation.error);
          throw new Error(responseValidation.error!);
        }

        const { submissionId, results } = response;
        const resultsObj = results as AssessmentResults;

        // Merge question text into results metadata
        const resultsToStore = mergeQuestionTextIntoResults(resultsObj, questionText);

        setResult(submissionId, resultsToStore);

        // Clear the current content after successful submission
        // This prevents the answer from being prepopulated when returning to the same question
        // Saved drafts remain intact for work-in-progress recovery
        createNewContentDraft();

        router.push(`/results/${submissionId}`);
      } catch (err) {
        errorLogger.logError(err, {
          page: "write",
          action: "submit_essay",
          taskId,
          wordCount,
        });
        setError(formatFriendlyErrorMessage(err, "write"));
        setLoading(false);
      }
    },
    [router, setResult, createNewContentDraft],
  );

  return {
    submit,
    loading,
    error,
  };
}
