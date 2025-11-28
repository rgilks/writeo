/**
 * Draft history section component
 */

import { DraftComparisonTable } from "./DraftComparisonTable";
import { DraftButton } from "./DraftButton";
import { useDraftNavigation } from "./useDraftNavigation";
import type { DraftHistory } from "@/app/lib/stores/draft-store";

export function DraftHistorySection({
  displayDraftHistory,
  draftNumber,
  submissionId,
  parentSubmissionId,
  getDraftHistory,
  onDraftSwitch,
}: {
  displayDraftHistory: DraftHistory[];
  draftNumber: number;
  submissionId?: string;
  parentSubmissionId?: string;
  getDraftHistory: (id: string) => DraftHistory[];
  onDraftSwitch?: (submissionId: string, parentId?: string) => boolean;
}) {
  if (displayDraftHistory.length <= 1) return null;

  const rootDraft = displayDraftHistory.find((d) => d.draftNumber === 1);
  // Determine root submission ID: use parentSubmissionId if it exists, otherwise use draft 1's submissionId
  const rootSubmissionId = parentSubmissionId || rootDraft?.submissionId || submissionId;

  const cardStyle = { padding: "var(--spacing-md)" } as const;
  const headingStyle = {
    fontSize: "16px",
    marginBottom: "var(--spacing-sm)",
    fontWeight: 600,
  } as const;
  const buttonsContainerStyle = {
    display: "flex",
    flexWrap: "wrap",
    gap: "var(--spacing-sm)",
    marginBottom: "var(--spacing-md)",
    width: "100%",
  } as const;

  return (
    <div className="card" lang="en" style={cardStyle} data-testid="draft-history">
      <h2 style={headingStyle} lang="en">
        Draft History
      </h2>
      <div style={buttonsContainerStyle} lang="en" data-testid="draft-buttons-container">
        {displayDraftHistory.map((draft) => {
          const { navigateUrl, hasValidSubmissionId } = useDraftNavigation(
            draft,
            draftNumber,
            submissionId,
            parentSubmissionId,
            rootDraft,
            getDraftHistory,
          );

          return (
            <DraftButton
              key={`draft-${draft.draftNumber}-${draft.submissionId || draft.timestamp}`}
              draft={draft}
              isCurrent={draft.draftNumber === draftNumber}
              navigateUrl={navigateUrl}
              hasValidSubmissionId={hasValidSubmissionId}
              rootSubmissionId={rootSubmissionId}
              onDraftSwitch={onDraftSwitch}
            />
          );
        })}
      </div>
      <DraftComparisonTable draftHistory={displayDraftHistory} currentDraftNumber={draftNumber} />
    </div>
  );
}
