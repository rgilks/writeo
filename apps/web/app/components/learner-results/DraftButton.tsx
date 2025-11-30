/**
 * Draft button component
 */

import { useState } from "react";
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
  const [isHovered, setIsHovered] = useState(false);
  const isInteractive = hasValidSubmissionId && !isCurrent;
  const overallScore = draft.overallScore;
  const showScore = typeof overallScore === "number";

  const handleClick = () => {
    if (!isInteractive) return;

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

  const backgroundColor = isCurrent
    ? "var(--primary-color)"
    : isHovered && isInteractive
      ? "var(--bg-primary)"
      : "var(--bg-secondary)";
  const border = isCurrent ? "2px solid var(--primary-color)" : "1px solid var(--border-color)";
  const transform = isHovered && isInteractive ? "scale(1.05)" : "scale(1)";
  const boxShadow = isHovered && isInteractive ? "var(--shadow-sm)" : "none";

  return (
    <div lang="en">
      <button
        type="button"
        onClick={handleClick}
        onMouseEnter={() => isInteractive && setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        disabled={!isInteractive}
        aria-pressed={isCurrent}
        data-testid={`draft-button-${draft.draftNumber}`}
        style={{
          width: "100%",
          padding: "var(--spacing-sm) var(--spacing-md)",
          backgroundColor,
          color: isCurrent ? "white" : "var(--text-primary)",
          borderRadius: "var(--border-radius)",
          fontSize: "14px",
          fontWeight: isCurrent ? 600 : 500,
          cursor: isInteractive ? "pointer" : "default",
          transition: "all 0.2s ease",
          opacity: hasValidSubmissionId ? 1 : 0.6,
          textAlign: "center",
          minWidth: "100px",
          border,
          transform,
          boxShadow,
        }}
      >
        <div style={{ fontWeight: 600, marginBottom: "2px" }}>Draft {draft.draftNumber}</div>
        {showScore && (
          <div style={{ fontSize: "12px", opacity: 0.9 }}>{overallScore!.toFixed(1)}</div>
        )}
      </button>
    </div>
  );
}
