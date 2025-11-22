/**
 * Draft table row component
 */

import { getScoreColor } from "./utils";
import { ChangeCell } from "./ChangeCell";
import type { DraftHistory } from "@/app/lib/stores/draft-store";

export function DraftTableRow({
  draft,
  index,
  prevDraft,
  currentDraftNumber,
}: {
  draft: DraftHistory;
  index: number;
  prevDraft: DraftHistory | null;
  currentDraftNumber: number;
}) {
  const scoreChange =
    prevDraft && draft.overallScore && prevDraft.overallScore
      ? draft.overallScore - prevDraft.overallScore
      : null;
  const wordChange =
    prevDraft && draft.wordCount && prevDraft.wordCount
      ? draft.wordCount - prevDraft.wordCount
      : null;
  const errorChange =
    prevDraft && draft.errorCount !== undefined && prevDraft.errorCount !== undefined
      ? draft.errorCount - prevDraft.errorCount
      : null;
  const isCurrent = draft.draftNumber === currentDraftNumber;

  return (
    <tr
      key={`draft-row-${draft.draftNumber}-${draft.submissionId || draft.timestamp}`}
      style={{
        borderBottom: "1px solid var(--border-color)",
        backgroundColor: isCurrent ? "rgba(59, 130, 246, 0.05)" : "transparent",
      }}
    >
      <td
        style={{
          padding: "var(--spacing-sm)",
          fontWeight: isCurrent ? 600 : 500,
          color: isCurrent ? "var(--primary-color)" : "var(--text-primary)",
        }}
        lang="en"
      >
        Draft {draft.draftNumber}
        {isCurrent && (
          <span
            style={{
              marginLeft: "var(--spacing-xs)",
              fontSize: "12px",
              color: "var(--text-secondary)",
            }}
            lang="en"
          >
            (current)
          </span>
        )}
      </td>
      <td
        style={{
          textAlign: "right",
          padding: "var(--spacing-sm)",
          fontWeight: 600,
          color: draft.overallScore ? getScoreColor(draft.overallScore) : "var(--text-secondary)",
        }}
        lang="en"
      >
        {draft.overallScore ? draft.overallScore.toFixed(1) : "-"}
      </td>
      <td
        style={{
          textAlign: "right",
          padding: "var(--spacing-sm)",
          color: "var(--text-primary)",
        }}
        lang="en"
      >
        {draft.wordCount || "-"}
      </td>
      <td
        style={{
          textAlign: "right",
          padding: "var(--spacing-sm)",
          color: "var(--text-primary)",
        }}
        lang="en"
      >
        {draft.errorCount !== undefined ? draft.errorCount : "-"}
      </td>
      <td
        style={{
          padding: "var(--spacing-sm)",
          color: "var(--text-secondary)",
          fontSize: "12px",
        }}
        lang="en"
      >
        {index === 0 ? (
          <span style={{ fontStyle: "italic" }}>Baseline</span>
        ) : (
          <ChangeCell scoreChange={scoreChange} wordChange={wordChange} errorChange={errorChange} />
        )}
      </td>
    </tr>
  );
}
