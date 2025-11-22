/**
 * Reveal prompt component - shown before feedback is revealed
 */

interface RevealPromptProps {
  onReveal: () => void;
  mediumConfidenceErrors: any[];
  lowConfidenceErrors: any[];
  showMediumConfidenceErrors: boolean;
  showExperimentalSuggestions: boolean;
  onShowMediumConfidence: () => void;
  onShowExperimental: () => void;
}

export function RevealPrompt({
  onReveal,
  mediumConfidenceErrors,
  lowConfidenceErrors,
  showMediumConfidenceErrors,
  showExperimentalSuggestions,
  onShowMediumConfidence,
  onShowExperimental,
}: RevealPromptProps) {
  return (
    <div
      lang="en"
      style={{
        marginBottom: "var(--spacing-md)",
        padding: "var(--spacing-md)",
        backgroundColor: "rgba(59, 130, 246, 0.1)",
        border: "1px solid rgba(59, 130, 246, 0.2)",
        borderRadius: "var(--border-radius)",
        fontSize: "14px",
        lineHeight: "1.5",
      }}
    >
      <p
        style={{
          marginBottom: "var(--spacing-sm)",
          fontWeight: 600,
          color: "var(--text-primary)",
        }}
      >
        💡 Find the Mistakes
      </p>
      <p
        style={{
          marginBottom: "var(--spacing-md)",
          color: "var(--text-secondary)",
          fontSize: "14px",
        }}
      >
        Look for the red highlights in your text. Can you spot what needs fixing? Click "Show
        Feedback" when you're ready to see suggestions.
      </p>
      <div
        style={{
          display: "flex",
          gap: "var(--spacing-sm)",
          flexWrap: "wrap",
          alignItems: "center",
        }}
      >
        <button
          onClick={onReveal}
          className="btn btn-primary"
          style={{ fontSize: "14px", padding: "var(--spacing-sm) var(--spacing-md)" }}
          lang="en"
        >
          Show Feedback
        </button>
        {(mediumConfidenceErrors.length > 0 || lowConfidenceErrors.length > 0) && (
          <span style={{ fontSize: "13px", color: "var(--text-secondary)" }} lang="en">
            {mediumConfidenceErrors.length > 0 && !showMediumConfidenceErrors && (
              <button
                onClick={onShowMediumConfidence}
                className="btn btn-secondary"
                style={{
                  fontSize: "13px",
                  padding: "var(--spacing-xs) var(--spacing-sm)",
                  marginLeft: "var(--spacing-xs)",
                }}
                lang="en"
              >
                +{mediumConfidenceErrors.length} more
              </button>
            )}
            {lowConfidenceErrors.length > 0 && !showExperimentalSuggestions && (
              <button
                onClick={onShowExperimental}
                className="btn btn-secondary"
                style={{
                  fontSize: "13px",
                  padding: "var(--spacing-xs) var(--spacing-sm)",
                  marginLeft: "var(--spacing-xs)",
                }}
                lang="en"
              >
                +{lowConfidenceErrors.length} experimental
              </button>
            )}
          </span>
        )}
      </div>
    </div>
  );
}
