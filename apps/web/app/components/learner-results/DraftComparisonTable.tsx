/**
 * Draft comparison table component
 */

import { DraftTableHeader } from "./DraftTableHeader";
import { DraftTableRow } from "./DraftTableRow";
import type { DraftHistory } from "@/app/lib/stores/draft-store";

export function DraftComparisonTable({
  draftHistory,
  currentDraftNumber,
}: {
  draftHistory: DraftHistory[];
  currentDraftNumber: number;
}) {
  if (draftHistory.length <= 1) return null;

  const containerStyle = {
    marginTop: "var(--spacing-md)",
    padding: "var(--spacing-md)",
    backgroundColor: "var(--bg-primary)",
    borderRadius: "var(--border-radius)",
    border: "1px solid var(--border-color)",
  } as const;

  const headingStyle = {
    fontSize: "16px",
    fontWeight: 600,
    marginBottom: "var(--spacing-md)",
    color: "var(--text-primary)",
  } as const;

  const tableStyle = {
    width: "100%",
    borderCollapse: "collapse",
    fontSize: "14px",
  } as const;

  return (
    <div style={containerStyle} lang="en">
      <h3 style={headingStyle} lang="en">
        Draft Comparison
      </h3>
      <div style={{ overflowX: "auto" }}>
        <table style={tableStyle} lang="en" data-testid="draft-comparison-table">
          <DraftTableHeader />
          <tbody>
            {draftHistory.map((draft, index, drafts) => (
              <DraftTableRow
                key={draft.draftNumber}
                draft={draft}
                index={index}
                prevDraft={index > 0 ? drafts[index - 1] : null}
                currentDraftNumber={currentDraftNumber}
              />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
