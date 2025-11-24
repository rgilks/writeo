/**
 * Hook for storing draft data
 */

import { useEffect } from "react";
import { countWords } from "@writeo/shared";
import { useDraftStore } from "@/app/lib/stores/draft-store";
import { extractErrorIds } from "@/app/lib/utils/progress";
import { mapScoreToCEFR } from "./utils";

export function useDraftStorage(
  submissionId: string | undefined,
  overall: number,
  draftNumber: number,
  grammarErrors: any[],
  finalAnswerText: string,
  parentSubmissionId: string | undefined
) {
  const addDraft = useDraftStore((state) => state.addDraft);
  const getDraftHistory = useDraftStore((state) => state.getDraftHistory);
  const getRootSubmissionId = useDraftStore((state) => state.getRootSubmissionId);
  const trackFixedErrors = useDraftStore((state) => state.trackFixedErrors);

  useEffect(() => {
    if (submissionId && overall > 0) {
      try {
        const cefrLevel = mapScoreToCEFR(overall);
        const wordCount = countWords(finalAnswerText);
        const errorIds = extractErrorIds(grammarErrors, finalAnswerText);

        const draftData = {
          draftNumber,
          submissionId,
          timestamp: new Date().toISOString(),
          wordCount,
          errorCount: grammarErrors.length,
          overallScore: overall,
          cefrLevel,
          errorIds: [...errorIds],
        };

        // Find the root submission ID to ensure all drafts are stored under the same key
        let rootSubmissionId = parentSubmissionId;
        if (parentSubmissionId) {
          // Try to find the actual root (draft 1's ID) from the parent
          const foundRoot = getRootSubmissionId(parentSubmissionId);
          if (foundRoot) {
            rootSubmissionId = foundRoot;
          }
        }

        // If we still don't have a root and this is draft 1, use submissionId
        if (!rootSubmissionId && draftNumber === 1) {
          rootSubmissionId = submissionId;
        }

        // Only add draft if we have a root submission ID
        if (!rootSubmissionId) {
          console.warn("Cannot add draft: no root submission ID found");
          return;
        }

        const newAchievements = addDraft(draftData, rootSubmissionId);

        if (newAchievements.length > 0) {
          console.log("New achievements unlocked:", newAchievements.map((a) => a.name).join(", "));
        }

        if (rootSubmissionId) {
          const previousDrafts = getDraftHistory(rootSubmissionId);
          if (previousDrafts.length > 0) {
            const previousDraft = previousDrafts[previousDrafts.length - 1];
            const previousErrorIds = Array.isArray(previousDraft.errorIds)
              ? [...previousDraft.errorIds]
              : [];
            const currentErrorIds = [...errorIds];
            trackFixedErrors(submissionId, previousErrorIds, currentErrorIds);
          }
        }
      } catch (error) {
        console.error("Error storing draft:", error);
      }
    }
    // Store functions (addDraft, getDraftHistory, getRootSubmissionId, trackFixedErrors) are stable
    // and don't need to be in the dependency array
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    submissionId,
    overall,
    draftNumber,
    grammarErrors.length,
    finalAnswerText,
    parentSubmissionId,
  ]);
}
