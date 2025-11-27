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
  // Use the draft's submissionId directly if available
  let draftSubmissionId = draft.submissionId;

  // If draft doesn't have a submissionId, try to find it from stored history
  if (!draftSubmissionId || draftSubmissionId.length === 0) {
    // Try to find the root submission ID first
    const rootId = parentSubmissionId || rootDraft?.submissionId || submissionId;
    if (rootId) {
      const storedHistory = getDraftHistory(rootId);
      const storedDraft = storedHistory.find((d) => d.draftNumber === draft.draftNumber);
      draftSubmissionId = storedDraft?.submissionId || "";
    }
  }

  const hasValidSubmissionId = Boolean(draftSubmissionId && draftSubmissionId.length > 0);
  const navigateUrl = hasValidSubmissionId ? `/results/${draftSubmissionId}` : "#";

  return { navigateUrl, hasValidSubmissionId };
}
