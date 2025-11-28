"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import type { AssessmentResults } from "@writeo/shared";
import { submitEssay } from "@/app/lib/actions";
import { useDraftStore } from "@/app/lib/stores/draft-store";
import { getErrorMessage } from "@/app/lib/utils/error-messages";
import { errorLogger } from "@/app/lib/utils/error-logger";

const SUBMISSION_TIMEOUT = 60000;

function getFriendlyErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return getErrorMessage(error, "write");
  }
  if (
    typeof error === "string" &&
    error.length < 200 &&
    !error.includes("Error:") &&
    !error.includes("at ")
  ) {
    return error;
  }
  return getErrorMessage(new Error("Submission failed"), "write");
}

export function useEssaySubmission() {
  const router = useRouter();
  const setResult = useDraftStore((state) => state.setResult);
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
      if (!answer.trim()) {
        setError("Please write your essay before submitting. Add your answer to receive feedback.");
        return;
      }

      if (wordCount < minWords) {
        setError(
          `Your essay is too short. Please write at least ${minWords} words (currently ${wordCount} words).`,
        );
        return;
      }

      if (wordCount > maxWords) {
        setError(
          `Your essay is too long. Please keep it under ${maxWords} words (currently ${wordCount} words).`,
        );
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

        const { submissionId, results } = await Promise.race([submitPromise, timeoutPromise]);

        if (!submissionId || !results) {
          throw new Error("No submission ID or results returned");
        }

        if (
          typeof results !== "object" ||
          results === null ||
          !("status" in results) ||
          !("template" in results)
        ) {
          throw new Error("Invalid results format");
        }

        const resultsObj = results as AssessmentResults;

        // Ensure question text is stored in metadata
        let resultsToStore = resultsObj;
        if (resultsObj && typeof window !== "undefined") {
          const answerTexts = resultsObj.meta?.answerTexts as Record<string, string> | undefined;
          const answerId = answerTexts ? Object.keys(answerTexts)[0] : undefined;

          if (answerId && questionText !== undefined) {
            if (!resultsObj.meta?.questionTexts) {
              resultsToStore = {
                ...resultsObj,
                meta: {
                  ...resultsObj.meta,
                  questionTexts: {
                    [answerId]: questionText,
                  },
                },
              };
            } else {
              const existingQuestionTexts = resultsObj.meta.questionTexts as Record<string, string>;
              if (!existingQuestionTexts[answerId]) {
                resultsToStore = {
                  ...resultsObj,
                  meta: {
                    ...resultsObj.meta,
                    questionTexts: {
                      ...existingQuestionTexts,
                      [answerId]: questionText,
                    },
                  },
                };
              }
            }
          }
        }

        setResult(submissionId, resultsToStore);
        router.push(`/results/${submissionId}`);
      } catch (err) {
        errorLogger.logError(err, {
          page: "write",
          action: "submit_essay",
          taskId,
          wordCount,
        });
        setError(getFriendlyErrorMessage(err));
        setLoading(false);
      }
    },
    [router, setResult],
  );

  return {
    submit,
    loading,
    error,
  };
}
