/**
 * Hook for calculating draft navigation URLs
 */

import type { DraftHistory } from "@/app/lib/stores/draft-store";

export function useDraftNavigation(
  draft: DraftHistory,
  draftNumber: number,
  submissionId: string | undefined,
  parentSubmissionId: string | undefined,
  rootDraft: DraftHistory | undefined,
  getDraftHistory: (id: string) => DraftHistory[]
) {
  let draftSubmissionId = draft.submissionId;

  if (draft.draftNumber === 1 && (!draftSubmissionId || draftSubmissionId.length === 0)) {
    if (parentSubmissionId && draftNumber > 1) {
      draftSubmissionId = parentSubmissionId;
    } else {
      const storedHistory = submissionId ? getDraftHistory(submissionId) : [];
      const storedDraft = storedHistory.find((d) => d.draftNumber === 1);
      draftSubmissionId = storedDraft?.submissionId || "";
    }
  }

  if ((!draftSubmissionId || draftSubmissionId.length === 0) && draft.draftNumber !== 1) {
    const storedHistory = submissionId ? getDraftHistory(submissionId) : [];
    const storedDraft = storedHistory.find((d) => d.draftNumber === draft.draftNumber);
    draftSubmissionId = storedDraft?.submissionId || "";
  }

  const hasValidSubmissionId = Boolean(draftSubmissionId && draftSubmissionId.length > 0);
  const isFirstDraft = draft.draftNumber === 1;
  const rootSubmissionId = rootDraft?.submissionId || parentSubmissionId || draftSubmissionId;

  const navigateUrl = hasValidSubmissionId
    ? isFirstDraft
      ? `/results/${draftSubmissionId}`
      : rootSubmissionId && rootSubmissionId !== draftSubmissionId
        ? `/results/${draftSubmissionId}?parent=${rootSubmissionId}`
        : `/results/${draftSubmissionId}`
    : "#";

  return { navigateUrl, hasValidSubmissionId };
}
