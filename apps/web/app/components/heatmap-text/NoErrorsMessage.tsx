/**
 * No errors message component
 */

interface NoErrorsMessageProps {
  mediumConfidenceErrors: any[];
  lowConfidenceErrors: any[];
  showMediumConfidenceErrors: boolean;
  showExperimentalSuggestions: boolean;
  onShowMediumConfidence: () => void;
  onShowExperimental: () => void;
}

export function NoErrorsMessage({
  mediumConfidenceErrors,
  lowConfidenceErrors,
  showMediumConfidenceErrors,
  showExperimentalSuggestions,
  onShowMediumConfidence,
  onShowExperimental,
}: NoErrorsMessageProps) {
  return (
    <div>
      <div
        lang="en"
        style={{
          marginBottom: "var(--spacing-md)",
          padding: "var(--spacing-md)",
          backgroundColor: "rgba(16, 185, 129, 0.1)",
          border: "1px solid rgba(16, 185, 129, 0.3)",
          borderRadius: "var(--border-radius)",
          fontSize: "14px",
          lineHeight: "1.5",
        }}
      >
        <p style={{ marginBottom: "var(--spacing-sm)", fontWeight: 600 }} lang="en">
          ✅ Great work! No issues found
        </p>
        {(mediumConfidenceErrors.length > 0 || lowConfidenceErrors.length > 0) && (
          <div
            style={{
              display: "flex",
              gap: "var(--spacing-sm)",
              flexWrap: "wrap",
              alignItems: "center",
              marginTop: "var(--spacing-sm)",
            }}
          >
            {mediumConfidenceErrors.length > 0 && !showMediumConfidenceErrors && (
              <button
                onClick={onShowMediumConfidence}
                className="btn btn-secondary"
                style={{ fontSize: "13px", padding: "var(--spacing-xs) var(--spacing-sm)" }}
                lang="en"
              >
                Show {mediumConfidenceErrors.length} more suggestion
                {mediumConfidenceErrors.length !== 1 ? "s" : ""}
              </button>
            )}
            {lowConfidenceErrors.length > 0 && !showExperimentalSuggestions && (
              <button
                onClick={onShowExperimental}
                className="btn btn-secondary"
                style={{ fontSize: "13px", padding: "var(--spacing-xs) var(--spacing-sm)" }}
                lang="en"
              >
                Show {lowConfidenceErrors.length} experimental suggestion
                {lowConfidenceErrors.length !== 1 ? "s" : ""}
              </button>
            )}
          </div>
        )}
      </div>
      <div className="prose max-w-none" translate="no" lang="en">
        {""}
      </div>
    </div>
  );
}
