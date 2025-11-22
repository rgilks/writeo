/**
 * Hook for managing draft history
 */

import { useDraftStore } from "@/app/lib/stores/draft-store";
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
  const getDraftHistory = useDraftStore((state) => state.getDraftHistory);

  const draftNumber = (data.meta?.draftNumber as number) || 1;
  const draftHistory =
    (data.meta?.draftHistory as Array<{
      draftNumber: number;
      timestamp: string;
      wordCount: number;
      errorCount: number;
      overallScore?: number;
    }>) || [];

  const storedDraftHistory = submissionId ? getDraftHistory(submissionId) : [];
  const parentDraftHistory = parentSubmissionId ? getDraftHistory(parentSubmissionId) : [];

  const allStoredDrafts = [
    ...storedDraftHistory,
    ...parentDraftHistory.filter(
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

  let displayDraftHistory = Array.from(draftMap.values()).sort(
    (a, b) => a.draftNumber - b.draftNumber
  );

  const hasCurrentDraft = displayDraftHistory.some((d) => d.draftNumber === draftNumber);
  if (!hasCurrentDraft && submissionId && overall > 0) {
    const wordCount = finalAnswerText.split(/\s+/).filter((w) => w.length > 0).length;
    displayDraftHistory.push({
      draftNumber,
      submissionId,
      timestamp: new Date().toISOString(),
      wordCount,
      errorCount: grammarErrors.length,
      overallScore: overall,
      cefrLevel: mapScoreToCEFR(overall),
      errorIds: [],
    });
    displayDraftHistory.sort((a, b) => a.draftNumber - b.draftNumber);
  }

  const finalDraftMap = new Map<number, (typeof displayDraftHistory)[0]>();
  displayDraftHistory.forEach((draft) => {
    const existing = finalDraftMap.get(draft.draftNumber);
    if (!existing || (!existing.submissionId && draft.submissionId)) {
      finalDraftMap.set(draft.draftNumber, draft);
    }
  });
  displayDraftHistory = Array.from(finalDraftMap.values()).sort(
    (a, b) => a.draftNumber - b.draftNumber
  );

  return { displayDraftHistory, draftNumber, parentSubmissionId };
}
