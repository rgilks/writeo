/**
 * Hook for storing draft data
 */

import { useEffect } from "react";
import { countWords } from "@writeo/shared";
import { useDraftStore } from "@/app/lib/stores/draft-store";
import { extractErrorIds } from "@/app/lib/utils/progress";
import { mapScoreToCEFR } from "./utils";

type StoredDraft = {
  draftNumber: number;
  submissionId?: string;
  timestamp: string;
  wordCount: number;
  errorCount: number;
  overallScore?: number;
  cefrLevel?: string;
  errorIds?: string[];
};

export function useDraftStorage(
  submissionId: string | undefined,
  overall: number,
  draftNumber: number,
  grammarErrors: any[],
  finalAnswerText: string,
  parentSubmissionId: string | undefined,
) {
  const addDraft = useDraftStore((state) => state.addDraft);
  const getDraftHistory = useDraftStore((state) => state.getDraftHistory);
  const getRootSubmissionId = useDraftStore((state) => state.getRootSubmissionId);
  const trackFixedErrors = useDraftStore((state) => state.trackFixedErrors);

  useEffect(() => {
    if (!submissionId) {
      return;
    }

    try {
      // Store draft even if overall score is 0 so draft history always reflects user edits
      const wordCount = countWords(finalAnswerText);
      const errorIds = extractErrorIds(grammarErrors, finalAnswerText);
      const cefrLevel = overall > 0 ? mapScoreToCEFR(overall) : undefined;

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

      const rootSubmissionId = resolveRootSubmissionId({
        draftNumber,
        submissionId,
        parentSubmissionId,
        getRootSubmissionId,
        getDraftHistory,
      });

      const previousDrafts = getDraftHistory(rootSubmissionId);
      const previousDraft = previousDrafts[previousDrafts.length - 1];
      const previousErrorIds = Array.isArray(previousDraft?.errorIds)
        ? [...previousDraft.errorIds]
        : [];

      addDraft(draftData, rootSubmissionId);
      trackFixedErrors(submissionId, previousErrorIds, [...errorIds]);
    } catch (error) {
      console.error("Error storing draft:", error);
    }
    // Store functions are stable and intentionally omitted from deps.
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

function resolveRootSubmissionId({
  draftNumber,
  submissionId,
  parentSubmissionId,
  getRootSubmissionId,
  getDraftHistory,
}: {
  draftNumber: number;
  submissionId: string;
  parentSubmissionId?: string;
  getRootSubmissionId: (id: string) => string | undefined | null;
  getDraftHistory: (id: string) => StoredDraft[];
}) {
  if (draftNumber === 1) {
    return submissionId;
  }

  const tryResolve = (id?: string) => {
    if (!id) return undefined;
    const resolved = getRootSubmissionId(id);
    return resolved ?? id;
  };

  const parentRoot = tryResolve(parentSubmissionId);
  if (parentRoot) {
    return parentRoot;
  }

  const parentDrafts = parentSubmissionId ? getDraftHistory(parentSubmissionId) : [];
  const parentDraftOne = parentDrafts.find((draft) => draft.draftNumber === 1);

  return parentDraftOne?.submissionId ?? parentSubmissionId ?? submissionId;
}
