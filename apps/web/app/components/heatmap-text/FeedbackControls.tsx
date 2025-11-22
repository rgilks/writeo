/**
 * Feedback controls component - shown when feedback is revealed
 */

interface FeedbackControlsProps {
  mediumConfidenceErrors: any[];
  lowConfidenceErrors: any[];
  showMediumConfidenceErrors: boolean;
  showExperimentalSuggestions: boolean;
  onToggleMediumConfidence: () => void;
  onToggleExperimental: () => void;
  onHideFeedback: () => void;
}

export function FeedbackControls({
  mediumConfidenceErrors,
  lowConfidenceErrors,
  showMediumConfidenceErrors,
  showExperimentalSuggestions,
  onToggleMediumConfidence,
  onToggleExperimental,
  onHideFeedback,
}: FeedbackControlsProps) {
  return (
    <div
      lang="en"
      style={{
        marginBottom: "var(--spacing-md)",
        padding: "var(--spacing-sm) var(--spacing-md)",
        backgroundColor: "var(--bg-secondary)",
        border: "1px solid rgba(0, 0, 0, 0.1)",
        borderRadius: "var(--border-radius)",
        fontSize: "13px",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        flexWrap: "wrap",
        gap: "var(--spacing-sm)",
      }}
    >
      <div
        style={{
          display: "flex",
          gap: "var(--spacing-sm)",
          flexWrap: "wrap",
          alignItems: "center",
        }}
      >
        {mediumConfidenceErrors.length > 0 && (
          <button
            onClick={onToggleMediumConfidence}
            className="btn btn-secondary"
            style={{ fontSize: "13px", padding: "var(--spacing-xs) var(--spacing-sm)" }}
            lang="en"
          >
            {showMediumConfidenceErrors ? "Hide" : "Show"} {mediumConfidenceErrors.length} more
            suggestion{mediumConfidenceErrors.length !== 1 ? "s" : ""}
          </button>
        )}
        {lowConfidenceErrors.length > 0 && (
          <button
            onClick={onToggleExperimental}
            className="btn btn-secondary"
            style={{ fontSize: "13px", padding: "var(--spacing-xs) var(--spacing-sm)" }}
            lang="en"
          >
            {showExperimentalSuggestions ? "Hide" : "Show"} {lowConfidenceErrors.length}{" "}
            experimental suggestion{lowConfidenceErrors.length !== 1 ? "s" : ""}
          </button>
        )}
      </div>
      <button
        onClick={onHideFeedback}
        className="btn btn-secondary"
        style={{ fontSize: "13px", padding: "var(--spacing-xs) var(--spacing-sm)" }}
        lang="en"
      >
        Hide Feedback
      </button>
    </div>
  );
}
