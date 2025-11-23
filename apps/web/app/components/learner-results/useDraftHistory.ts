/**
 * Hook for managing draft history
 */

import { useMemo } from "react";
import { useDraftStore } from "@/app/lib/stores/draft-store";
import { countWords } from "@writeo/shared";
import { mapScoreToCEFR } from "./utils";
import type { AssessmentResults } from "@writeo/shared";

export function useDraftHistory(
  data: AssessmentResults,
  submissionId: string | undefined,
  overall: number,
  grammarErrors: any[],
  finalAnswerText: string,
  parentSubmissionId?: string
) {
  // Subscribe to drafts state so component re-renders when drafts change
  const drafts = useDraftStore((state) => state.drafts);
  // Store functions are stable
  const getDraftHistory = useDraftStore((state) => state.getDraftHistory);
  const getRootSubmissionId = useDraftStore((state) => state.getRootSubmissionId);

  const draftNumber = (data.meta?.draftNumber as number) || 1;
  const draftHistory =
    (data.meta?.draftHistory as Array<{
      draftNumber: number;
      timestamp: string;
      wordCount: number;
      errorCount: number;
      overallScore?: number;
    }>) || [];

  // Memoize expensive draft history computation
  const displayDraftHistory = useMemo(() => {
    // Find the root submission ID - this is critical for getting all drafts
    let rootSubmissionId = parentSubmissionId || submissionId || "";

    // If we have a submissionId, try to find the actual root using the store helper
    // This handles the case where parentSubmissionId might be a non-root parent
    if (submissionId) {
      const foundRoot = getRootSubmissionId(submissionId);
      if (foundRoot) {
        rootSubmissionId = foundRoot;
      }
    }

    // If we still don't have a root from the store, try parentSubmissionId
    if (!rootSubmissionId && parentSubmissionId) {
      const foundRoot = getRootSubmissionId(parentSubmissionId);
      if (foundRoot) {
        rootSubmissionId = foundRoot;
      }
    }

    // Get all drafts stored under the root submission ID
    const storedDraftHistory = rootSubmissionId ? getDraftHistory(rootSubmissionId) : [];

    // Also try to get drafts by current submissionId in case they're stored differently
    const currentDraftHistory =
      submissionId && submissionId !== rootSubmissionId ? getDraftHistory(submissionId) : [];

    // Merge both, preferring stored drafts with submissionIds
    const allStoredDrafts = [
      ...storedDraftHistory,
      ...currentDraftHistory.filter(
        (d) => !storedDraftHistory.some((sd) => sd.submissionId === d.submissionId)
      ),
    ];

    const draftMap = new Map<number, (typeof allStoredDrafts)[0]>();

    allStoredDrafts.forEach((draft) => {
      if (draft.draftNumber) {
        const existing = draftMap.get(draft.draftNumber);
        if (!existing) {
          draftMap.set(draft.draftNumber, draft);
        } else if (
          (!existing.submissionId && draft.submissionId) ||
          (existing.timestamp < draft.timestamp && draft.submissionId)
        ) {
          draftMap.set(draft.draftNumber, draft);
        }
      }
    });

    draftHistory.forEach((d, idx) => {
      const draftNum = d.draftNumber || idx + 1;
      if (!draftMap.has(draftNum)) {
        const isCurrentDraft = draftNum === draftNumber;
        draftMap.set(draftNum, {
          draftNumber: draftNum,
          submissionId: isCurrentDraft ? submissionId || "" : "",
          timestamp: d.timestamp,
          wordCount: d.wordCount,
          errorCount: d.errorCount,
          overallScore: d.overallScore,
          cefrLevel: d.overallScore ? mapScoreToCEFR(d.overallScore) : undefined,
          errorIds: [],
        });
      }
    });

    let result = Array.from(draftMap.values()).sort((a, b) => a.draftNumber - b.draftNumber);

    const hasCurrentDraft = result.some((d) => d.draftNumber === draftNumber);
    if (!hasCurrentDraft && submissionId && overall > 0) {
      const wordCount = countWords(finalAnswerText);
      result.push({
        draftNumber,
        submissionId,
        timestamp: new Date().toISOString(),
        wordCount,
        errorCount: grammarErrors.length,
        overallScore: overall,
        cefrLevel: mapScoreToCEFR(overall),
        errorIds: [],
      });
      result.sort((a, b) => a.draftNumber - b.draftNumber);
    }

    const finalDraftMap = new Map<number, (typeof result)[0]>();
    result.forEach((draft) => {
      const existing = finalDraftMap.get(draft.draftNumber);
      if (!existing || (!existing.submissionId && draft.submissionId)) {
        finalDraftMap.set(draft.draftNumber, draft);
      }
    });

    return Array.from(finalDraftMap.values()).sort((a, b) => a.draftNumber - b.draftNumber);
  }, [
    draftNumber,
    draftHistory,
    submissionId,
    parentSubmissionId,
    overall,
    grammarErrors.length,
    finalAnswerText,
    getDraftHistory,
    getRootSubmissionId,
    drafts, // Add drafts to dependency array
  ]);

  return { displayDraftHistory, draftNumber, parentSubmissionId };
}
