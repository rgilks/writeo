/**
 * Hook for handling essay resubmission
 */

import { useState } from "react";
import { useRouter } from "next/navigation";
import { submitEssay } from "@/app/lib/actions";
import { usePreferencesStore } from "@/app/lib/stores/preferences-store";

export function useResubmit() {
  const router = useRouter();
  const [isResubmitting, setIsResubmitting] = useState(false);

  const handleResubmit = async (
    editedText: string,
    questionText: string,
    submissionId: string,
    parentSubmissionId?: string
  ) => {
    if (!questionText) {
      throw new Error("Cannot resubmit: Question text not available");
    }
    if (!submissionId) {
      throw new Error("Cannot resubmit: Submission ID not available");
    }

    setIsResubmitting(true);
    try {
      const parentId = parentSubmissionId || submissionId;
      const storeResults = usePreferencesStore.getState().storeResults;

      let parentResults: any = undefined;
      if (!storeResults && typeof window !== "undefined" && parentId) {
        const storedParentResults = localStorage.getItem(`results_${parentId}`);
        if (storedParentResults) {
          try {
            parentResults = JSON.parse(storedParentResults);
          } catch (e) {
            console.warn("[handleResubmit] Failed to parse parent results from localStorage:", e);
          }
        }
      }

      const { submissionId: newSubmissionId, results } = await submitEssay(
        questionText,
        editedText,
        parentId,
        storeResults,
        parentResults
      );
      if (!newSubmissionId || !results) {
        throw new Error("No submission ID or results returned");
      }

      if (typeof window !== "undefined") {
        sessionStorage.setItem(`results_${newSubmissionId}`, JSON.stringify(results));
        localStorage.setItem(`results_${newSubmissionId}`, JSON.stringify(results));

        const parentToUse = parentId || submissionId;
        if (parentToUse) {
          localStorage.setItem(`draft_parent_${newSubmissionId}`, parentToUse);
        }
      }

      const parentToUse = parentId || submissionId;
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
