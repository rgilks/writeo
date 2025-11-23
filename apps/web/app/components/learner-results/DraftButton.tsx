/**
 * Draft button component
 */

import { useRouter } from "next/navigation";
import type { DraftHistory } from "@/app/lib/stores/draft-store";

export function DraftButton({
  draft,
  isCurrent,
  navigateUrl,
  hasValidSubmissionId,
  rootSubmissionId,
  onDraftSwitch,
}: {
  draft: DraftHistory;
  isCurrent: boolean;
  navigateUrl: string;
  hasValidSubmissionId: boolean;
  rootSubmissionId?: string;
  onDraftSwitch?: (submissionId: string, parentId?: string) => boolean;
}) {
  const router = useRouter();

  const handleClick = () => {
    if (!hasValidSubmissionId || isCurrent) return;
    
    // Try client-side switch first (if callback provided)
    if (onDraftSwitch && draft.submissionId) {
      const switched = onDraftSwitch(draft.submissionId, rootSubmissionId);
      if (switched) {
        return; // Successfully switched without navigation
      }
    }
    
    // Fall back to navigation if client-side switch failed or callback not provided
    router.push(navigateUrl);
  };

  return (
    <div
      key={`draft-${draft.draftNumber}-${draft.submissionId || draft.timestamp}`}
      style={{
        padding: "var(--spacing-sm) var(--spacing-md)",
        backgroundColor: isCurrent ? "var(--primary-color)" : "var(--bg-secondary)",
        color: isCurrent ? "white" : "var(--text-primary)",
        borderRadius: "var(--border-radius)",
        fontSize: "14px",
        fontWeight: isCurrent ? 600 : 500,
        cursor: hasValidSubmissionId && !isCurrent ? "pointer" : "default",
        transition: "all 0.2s ease",
        opacity: hasValidSubmissionId ? 1 : 0.6,
        textAlign: "center",
        minWidth: "100px",
        border: isCurrent ? "2px solid var(--primary-color)" : "1px solid var(--border-color)",
      }}
      onClick={handleClick}
      onMouseEnter={(e) => {
        if (!isCurrent && hasValidSubmissionId) {
          e.currentTarget.style.backgroundColor = "var(--bg-primary)";
          e.currentTarget.style.transform = "scale(1.05)";
          e.currentTarget.style.boxShadow = "var(--shadow-sm)";
        }
      }}
      onMouseLeave={(e) => {
        if (!isCurrent) {
          e.currentTarget.style.backgroundColor = "var(--bg-secondary)";
          e.currentTarget.style.transform = "scale(1)";
          e.currentTarget.style.boxShadow = "none";
        }
      }}
      lang="en"
    >
      <div style={{ fontWeight: 600, marginBottom: "2px" }}>Draft {draft.draftNumber}</div>
      {draft.overallScore && (
        <div style={{ fontSize: "12px", opacity: 0.9 }}>{draft.overallScore.toFixed(1)}</div>
      )}
    </div>
  );
}
