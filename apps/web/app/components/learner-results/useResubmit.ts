/**
 * Hook for handling essay resubmission
 */

import { useCallback, useState } from "react";
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

  const handleResubmit = useCallback(
    async (editedText: string, questionText: string, submissionId: string) => {
      if (isResubmitting) {
        return;
      }
      if (!submissionId) {
        throw new Error("Cannot resubmit: Submission ID not available");
      }

      setIsResubmitting(true);
      try {
        const parentId = submissionId;
        const parentResults = !storeResults && parentId ? getResult(parentId) : undefined;
        const finalQuestionText = questionText?.trim() || "";

        const { submissionId: newSubmissionId, results } = await submitEssay(
          finalQuestionText,
          editedText,
          parentId,
          storeResults,
          parentResults,
        );

        if (!newSubmissionId || !results) {
          throw new Error("No submission ID or results returned");
        }

        setResult(newSubmissionId, results);
        router.push(`/results/${newSubmissionId}`);
      } finally {
        setIsResubmitting(false);
      }
    },
    [getResult, isResubmitting, router, setResult, storeResults],
  );

  return { handleResubmit, isResubmitting };
}
