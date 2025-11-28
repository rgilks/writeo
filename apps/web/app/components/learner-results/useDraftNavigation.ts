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
  getDraftHistory: (id: string) => DraftHistory[],
) {
  const fallbackRootId = parentSubmissionId ?? rootDraft?.submissionId ?? submissionId;
  const draftSubmissionId =
    draft.submissionId ||
    findSubmissionId(draft.draftNumber ?? draftNumber, fallbackRootId, getDraftHistory);

  const hasValidSubmissionId = Boolean(draftSubmissionId);
  const navigateUrl = hasValidSubmissionId ? `/results/${draftSubmissionId}` : "#";

  return { navigateUrl, hasValidSubmissionId };
}

function findSubmissionId(
  draftNumber: number | undefined,
  rootId: string | undefined,
  getDraftHistory: (id: string) => DraftHistory[],
) {
  if (!draftNumber || !rootId) {
    return undefined;
  }

  const storedHistory = getDraftHistory(rootId);
  return storedHistory.find((storedDraft) => storedDraft.draftNumber === draftNumber)?.submissionId;
}
