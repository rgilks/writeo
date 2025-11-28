import type { NoErrorsMessageProps } from "./types";

const SUCCESS_BOX_STYLES = {
  marginBottom: "var(--spacing-md)",
  padding: "var(--spacing-md)",
  backgroundColor: "rgba(16, 185, 129, 0.1)",
  border: "1px solid rgba(16, 185, 129, 0.3)",
  borderRadius: "var(--border-radius)",
  fontSize: "14px",
  lineHeight: "1.5",
} as const;

const MESSAGE_STYLES = {
  marginBottom: "var(--spacing-sm)",
  fontWeight: 600,
} as const;

const BUTTON_GROUP_STYLES = {
  display: "flex",
  gap: "var(--spacing-sm)",
  flexWrap: "wrap",
  alignItems: "center",
  marginTop: "var(--spacing-sm)",
} as const;

const BUTTON_STYLES = {
  fontSize: "13px",
  padding: "var(--spacing-xs) var(--spacing-sm)",
} as const;

function pluralize(count: number, singular: string): string {
  return count === 1 ? singular : `${singular}s`;
}

interface ShowButtonProps {
  count: number;
  isVisible: boolean;
  label: string;
  onClick: () => void;
}

function ShowButton({ count, isVisible, label, onClick }: ShowButtonProps) {
  if (count === 0 || isVisible) return null;

  return (
    <button onClick={onClick} className="btn btn-secondary" style={BUTTON_STYLES}>
      Show {count} {label}
    </button>
  );
}

export function NoErrorsMessage({
  mediumConfidenceErrors,
  lowConfidenceErrors,
  showMediumConfidenceErrors,
  showExperimentalSuggestions,
  onShowMediumConfidence,
  onShowExperimental,
}: NoErrorsMessageProps) {
  const hasAdditionalErrors = mediumConfidenceErrors.length > 0 || lowConfidenceErrors.length > 0;

  return (
    <div style={SUCCESS_BOX_STYLES}>
      <p style={MESSAGE_STYLES}>✅ Great work! No issues found</p>
      {hasAdditionalErrors && (
        <div style={BUTTON_GROUP_STYLES}>
          <ShowButton
            count={mediumConfidenceErrors.length}
            isVisible={showMediumConfidenceErrors}
            label={`more ${pluralize(mediumConfidenceErrors.length, "suggestion")}`}
            onClick={onShowMediumConfidence}
          />
          <ShowButton
            count={lowConfidenceErrors.length}
            isVisible={showExperimentalSuggestions}
            label={`experimental ${pluralize(lowConfidenceErrors.length, "suggestion")}`}
            onClick={onShowExperimental}
          />
        </div>
      )}
    </div>
  );
}
