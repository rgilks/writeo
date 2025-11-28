/**
 * Draft table row component
 */

import { getScoreColor } from "./utils";
import { ChangeCell } from "./ChangeCell";
import type { DraftHistory } from "@/app/lib/stores/draft-store";

const cellBaseStyle = {
  padding: "var(--spacing-sm)",
  color: "var(--text-primary)",
} as const;

const numericCellStyle = {
  ...cellBaseStyle,
  textAlign: "right",
} as const;

type DraftTableRowProps = {
  draft: DraftHistory;
  index: number;
  prevDraft: DraftHistory | null;
  currentDraftNumber: number;
};

export function DraftTableRow({ draft, index, prevDraft, currentDraftNumber }: DraftTableRowProps) {
  const hasPrevScore = prevDraft?.overallScore !== undefined && prevDraft?.overallScore !== null;
  const hasScore = draft.overallScore !== undefined && draft.overallScore !== null;
  const hasPrevWords = typeof prevDraft?.wordCount === "number";
  const hasWords = typeof draft.wordCount === "number";
  const hasPrevErrors = typeof prevDraft?.errorCount === "number";
  const hasErrors = typeof draft.errorCount === "number";

  const scoreChange =
    hasPrevScore && hasScore ? draft.overallScore! - prevDraft!.overallScore! : null;
  const wordChange = hasPrevWords && hasWords ? draft.wordCount! - prevDraft!.wordCount! : null;
  const errorChange =
    hasPrevErrors && hasErrors ? draft.errorCount! - prevDraft!.errorCount! : null;
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
          ...numericCellStyle,
          fontWeight: 600,
          color: hasScore ? getScoreColor(draft.overallScore!) : "var(--text-secondary)",
        }}
        lang="en"
      >
        {hasScore ? draft.overallScore!.toFixed(1) : "-"}
      </td>
      <td style={numericCellStyle} lang="en">
        {hasWords ? draft.wordCount : "-"}
      </td>
      <td style={numericCellStyle} lang="en">
        {hasErrors ? draft.errorCount : "-"}
      </td>
      <td style={{ ...cellBaseStyle, color: "var(--text-secondary)", fontSize: "12px" }} lang="en">
        {index === 0 ? (
          <span style={{ fontStyle: "italic" }}>Baseline</span>
        ) : (
          <ChangeCell scoreChange={scoreChange} wordChange={wordChange} errorChange={errorChange} />
        )}
      </td>
    </tr>
  );
}
