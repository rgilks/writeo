/**
 * Hook for handling essay resubmission
 */

import { useState } from "react";
import { useRouter } from "next/navigation";
import { submitEssay } from "@/app/lib/actions";
import { usePreferencesStore } from "@/app/lib/stores/preferences-store";
import { useResultsStore } from "@/app/lib/stores/results-store";

export function useResubmit() {
  const router = useRouter();
  const [isResubmitting, setIsResubmitting] = useState(false);
  const getResult = useResultsStore((state) => state.getResult);
  const setResult = useResultsStore((state) => state.setResult);
  // Use hook selector for consistency (even though we read it in async function)
  // This ensures component re-renders if preference changes
  const storeResults = usePreferencesStore((state) => state.storeResults);

  const handleResubmit = async (
    editedText: string,
    questionText: string,
    submissionId: string,
    parentSubmissionId?: string
  ) => {
    if (!submissionId) {
      throw new Error("Cannot resubmit: Submission ID not available");
    }

    setIsResubmitting(true);
    try {
      const parentId = parentSubmissionId || submissionId;

      let parentResults: any = undefined;
      if (!storeResults && parentId) {
        parentResults = getResult(parentId);
      }

      // Use question text if provided, otherwise use empty string for free writing
      const finalQuestionText = questionText?.trim() || "";

      const { submissionId: newSubmissionId, results } = await submitEssay(
        finalQuestionText,
        editedText,
        parentId,
        storeResults,
        parentResults
      );
      if (!newSubmissionId || !results) {
        throw new Error("No submission ID or results returned");
      }

      // Store results in results store
      const parentToUse = parentId || submissionId;
      setResult(newSubmissionId, results, parentToUse);

      // Also store in sessionStorage for immediate display
      if (typeof window !== "undefined") {
        sessionStorage.setItem(`results_${newSubmissionId}`, JSON.stringify(results));
      }

      if (parentToUse) {
        router.push(`/results/${newSubmissionId}?parent=${parentToUse}`);
      } else {
        router.push(`/results/${newSubmissionId}`);
      }
    } catch (error) {
      setIsResubmitting(false);
      throw error;
    }
  };

  return { handleResubmit, isResubmitting };
}
