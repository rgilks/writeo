/**
 * Draft comparison table component
 */

import { DraftTableHeader } from "./DraftTableHeader";
import { DraftTableRow } from "./DraftTableRow";

interface Draft {
  draftNumber: number;
  submissionId?: string;
  timestamp: string;
  wordCount: number;
  errorCount: number;
  overallScore?: number;
}

export function DraftComparisonTable({
  draftHistory,
  currentDraftNumber,
}: {
  draftHistory: Draft[];
  currentDraftNumber: number;
}) {
  if (draftHistory.length <= 1) return null;

  return (
    <div
      style={{
        marginTop: "var(--spacing-md)",
        padding: "var(--spacing-md)",
        backgroundColor: "var(--bg-primary)",
        borderRadius: "var(--border-radius)",
        border: "1px solid var(--border-color)",
      }}
      lang="en"
    >
      <h3
        style={{
          fontSize: "16px",
          fontWeight: 600,
          marginBottom: "var(--spacing-md)",
          color: "var(--text-primary)",
        }}
        lang="en"
      >
        Draft Comparison
      </h3>
      <div style={{ overflowX: "auto" }}>
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            fontSize: "14px",
          }}
          lang="en"
        >
          <DraftTableHeader />
          <tbody>
            {draftHistory.map((draft, index) => {
              const prevDraft = index > 0 ? draftHistory[index - 1] : null;
              return (
                <DraftTableRow
                  key={draft.draftNumber}
                  draft={draft}
                  index={index}
                  prevDraft={prevDraft}
                  currentDraftNumber={currentDraftNumber}
                />
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
