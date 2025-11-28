import type { ErrorDetailProps } from "./types";

const CLOSE_BUTTON_STYLES = {
  position: "absolute" as const,
  top: "-8px",
  right: "-8px",
  background: "var(--bg-primary)",
  border: "1px solid var(--border-color)",
  borderRadius: "50%",
  width: "24px",
  height: "24px",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  cursor: "pointer",
  fontSize: "18px",
  lineHeight: "1",
  padding: 0,
  color: "var(--text-secondary)",
};

const SIMPLE_MESSAGE_STYLES = {
  marginTop: "var(--spacing-xs)",
  fontSize: "14px",
  color: "var(--text-primary)",
  lineHeight: "1.5",
  position: "relative" as const,
};

const ERROR_TYPE_BADGE_STYLES = {
  padding: "var(--spacing-xs) var(--spacing-sm)",
  backgroundColor: "var(--primary-color)",
  color: "white",
  borderRadius: "var(--spacing-xs)",
  fontSize: "14px",
  fontWeight: 600,
};

const CONFIDENCE_LABEL_STYLES = {
  fontSize: "10px",
  color: "var(--text-secondary)",
  fontWeight: 400,
};

const EXPERIMENTAL_WARNING_STYLES = {
  margin: "0 0 var(--spacing-xs) 0",
  padding: "var(--spacing-xs)",
  backgroundColor: "rgba(245, 158, 11, 0.1)",
  borderRadius: "var(--spacing-xs)",
  fontSize: "13px",
  color: "#92400e",
  fontStyle: "italic" as const,
};

function CloseButton({
  onClose,
  withShadow = false,
}: {
  onClose?: () => void;
  withShadow?: boolean;
}) {
  if (!onClose) return null;

  return (
    <button
      onClick={onClose}
      style={{
        ...CLOSE_BUTTON_STYLES,
        ...(withShadow && { zIndex: 1, boxShadow: "0 2px 4px rgba(0,0,0,0.1)" }),
      }}
      aria-label="Close"
    >
      ×
    </button>
  );
}

function getConfidenceInfo(error: ErrorDetailProps["error"]) {
  const isMediumConfidence = error.mediumConfidence === true;
  const isExperimental = error.highConfidence === false && error.mediumConfidence !== true;

  return {
    isMediumConfidence,
    isExperimental,
    label: isExperimental
      ? "Experimental"
      : isMediumConfidence
        ? "Medium Confidence"
        : "High Confidence",
    tooltip: isExperimental
      ? "This is an experimental suggestion - you may want to check with a teacher"
      : "This is a medium-confidence suggestion - likely correct but less certain",
  };
}

export function ErrorDetail({ error, errorText: _errorText, onClose }: ErrorDetailProps) {
  const hasStructuredFeedback = !!(error.errorType || error.explanation || error.example);

  if (!hasStructuredFeedback) {
    return (
      <div style={SIMPLE_MESSAGE_STYLES}>
        <CloseButton onClose={onClose} />
        {error.message}
        {error.suggestions?.[0] && ` → "${error.suggestions[0]}"`}
      </div>
    );
  }

  const confidence = getConfidenceInfo(error);

  return (
    <div style={{ marginTop: "var(--spacing-sm)", position: "relative" }}>
      <CloseButton onClose={onClose} withShadow />
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--spacing-sm)",
          marginBottom: "var(--spacing-sm)",
          flexWrap: "wrap",
        }}
      >
        <span style={ERROR_TYPE_BADGE_STYLES}>{error.errorType || "Error"}</span>
        {(confidence.isMediumConfidence || confidence.isExperimental) && (
          <span style={CONFIDENCE_LABEL_STYLES} title={confidence.tooltip}>
            ({confidence.label})
          </span>
        )}
      </div>

      <div style={{ marginBottom: "var(--spacing-sm)", fontSize: "14px", lineHeight: "1.5" }}>
        {confidence.isExperimental && (
          <p style={EXPERIMENTAL_WARNING_STYLES}>
            💡 This is an experimental suggestion - you may want to check with a teacher.
          </p>
        )}
        <p style={{ margin: "0 0 var(--spacing-xs) 0", color: "var(--text-primary)" }}>
          {error.explanation || error.message}
        </p>
        {error.example && (
          <p
            style={{
              margin: "0",
              color: "var(--text-primary)",
              fontFamily: "monospace",
              fontSize: "14px",
            }}
          >
            Example: {error.example}
          </p>
        )}
      </div>
    </div>
  );
}
