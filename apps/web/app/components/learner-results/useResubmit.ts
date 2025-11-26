/**
 * Hook for handling essay resubmission
 */

import { useState } from "react";
import { useRouter } from "next/navigation";
import { submitEssay } from "@/app/lib/actions";
import { usePreferencesStore } from "@/app/lib/stores/preferences-store";
import { useDraftStore } from "@/app/lib/stores/draft-store";

export function useResubmit() {
  const router = useRouter();
  const [isResubmitting, setIsResubmitting] = useState(false);
  const getResult = useDraftStore((state) => state.getResult);
  const setResult = useDraftStore((state) => state.setResult);
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
      // Always use the current submissionId as parent for draft number calculation
      // This ensures correct draft numbering (draft 2 -> draft 3, not draft 2 -> draft 2)
      // The getDraftInfo function will find the root submission ID from the parent's metadata
      const parentId = submissionId;

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

      // Store results in draft store (Zustand persist handles localStorage automatically)
      setResult(newSubmissionId, results);

      // Redirect to results page
      router.push(`/results/${newSubmissionId}`);
    } catch (error) {
      setIsResubmitting(false);
      throw error;
    }
  };

  return { handleResubmit, isResubmitting };
}
