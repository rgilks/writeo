/**
 * Error detail component - displays error information
 */

import type { ErrorDetailProps } from "./types";

export function ErrorDetail({ error, errorText, onClose }: ErrorDetailProps) {
  const hasStructuredFeedback = !!(error.errorType || error.explanation || error.example);

  if (!hasStructuredFeedback) {
    return (
      <div
        style={{
          marginTop: "var(--spacing-xs)",
          fontSize: "14px",
          color: "var(--text-primary)",
          lineHeight: "1.5",
          position: "relative",
        }}
        lang="en"
      >
        {onClose && (
          <button
            onClick={onClose}
            style={{
              position: "absolute",
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
            }}
            aria-label="Close"
            lang="en"
          >
            ×
          </button>
        )}
        {error.message}
        {error.suggestions?.[0] && ` → "${error.suggestions[0]}"`}
      </div>
    );
  }

  const isMediumConfidence = error.mediumConfidence === true;
  const isExperimental = error.highConfidence === false && error.mediumConfidence !== true;
  const confidenceLabel = isExperimental
    ? "Experimental"
    : isMediumConfidence
      ? "Medium Confidence"
      : "High Confidence";
  const confidenceColor = isExperimental ? "#f59e0b" : isMediumConfidence ? "#f97316" : "#10b981";

  return (
    <div style={{ marginTop: "var(--spacing-sm)", position: "relative" }} lang="en">
      {onClose && (
        <button
          onClick={onClose}
          style={{
            position: "absolute",
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
            zIndex: 1,
            boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
          }}
          aria-label="Close"
          lang="en"
        >
          ×
        </button>
      )}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--spacing-sm)",
          marginBottom: "var(--spacing-sm)",
          flexWrap: "wrap",
        }}
        lang="en"
      >
        <span
          style={{
            padding: "var(--spacing-xs) var(--spacing-sm)",
            backgroundColor: "var(--primary-color)",
            color: "white",
            borderRadius: "var(--spacing-xs)",
            fontSize: "14px",
            fontWeight: 600,
          }}
          lang="en"
        >
          {error.errorType || "Error"}
        </span>
        {(isMediumConfidence || isExperimental) && (
          <span
            style={{
              fontSize: "10px",
              color: "var(--text-secondary)",
              fontWeight: 400,
            }}
            lang="en"
            title={
              isExperimental
                ? "This is an experimental suggestion - you may want to check with a teacher"
                : "This is a medium-confidence suggestion - likely correct but less certain"
            }
          >
            ({confidenceLabel})
          </span>
        )}
      </div>

      <div
        style={{ marginBottom: "var(--spacing-sm)", fontSize: "14px", lineHeight: "1.5" }}
        lang="en"
      >
        {isExperimental && (
          <p
            style={{
              margin: "0 0 var(--spacing-xs) 0",
              padding: "var(--spacing-xs)",
              backgroundColor: "rgba(245, 158, 11, 0.1)",
              borderRadius: "var(--spacing-xs)",
              fontSize: "13px",
              color: "#92400e",
              fontStyle: "italic",
            }}
            lang="en"
          >
            💡 This is an experimental suggestion - you may want to check with a teacher.
          </p>
        )}
        <p style={{ margin: "0 0 var(--spacing-xs) 0", color: "var(--text-primary)" }} lang="en">
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
            lang="en"
          >
            Example: {error.example}
          </p>
        )}
      </div>
    </div>
  );
}
