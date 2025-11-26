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
    // Store draft even if overall score is 0 (essay scoring may have failed)
    // This ensures draft history works correctly regardless of scoring service availability
    if (submissionId) {
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
        let rootSubmissionId: string | undefined = undefined;

        if (draftNumber === 1) {
          // Draft 1 is always the root
          rootSubmissionId = submissionId;
        } else if (parentSubmissionId) {
          // For drafts 2+, try to find the root from the parent
          const foundRoot = getRootSubmissionId(parentSubmissionId);
          if (foundRoot) {
            rootSubmissionId = foundRoot;
          } else {
            // If we can't find the root in the store, try to find it by checking parent's drafts
            const parentDrafts = getDraftHistory(parentSubmissionId);
            if (parentDrafts.length > 0) {
              // Find draft 1 in the parent's draft history
              const parentDraft1 = parentDrafts.find((d) => d.draftNumber === 1);
              if (parentDraft1) {
                rootSubmissionId = parentDraft1.submissionId;
              } else {
                // If parent doesn't have draft 1, use parentSubmissionId as the key
                // This handles cases where drafts might be stored under intermediate parents
                rootSubmissionId = parentSubmissionId;
              }
            } else {
              // No parent drafts found - use parentSubmissionId as root
              // This handles the case where we're storing draft 2 and draft 1 hasn't been stored yet
              rootSubmissionId = parentSubmissionId;
            }
          }
        }

        // Fallback: if we still don't have a root, use submissionId (shouldn't happen)
        if (!rootSubmissionId) {
          console.warn(
            "Cannot add draft: no root submission ID found, using submissionId as fallback"
          );
          rootSubmissionId = submissionId;
        }

        const newAchievements = addDraft(draftData, rootSubmissionId);

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
