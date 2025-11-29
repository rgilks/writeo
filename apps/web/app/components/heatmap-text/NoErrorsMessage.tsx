import type { NoErrorsMessageProps } from "./types";
import { pluralize } from "@/app/lib/utils/text-utils";

const SUCCESS_BOX_STYLES = {
  marginBottom: "var(--spacing-md)",
} as const;

const MESSAGE_STYLES = {
  marginBottom: "var(--spacing-sm)",
  fontWeight: 600,
} as const;

// BUTTON_GROUP_STYLES removed - use .button-group class instead

const BUTTON_STYLES = {
  fontSize: "13px",
  padding: "var(--spacing-xs) var(--spacing-sm)",
} as const;

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
    <div className="info-box info-box-success" style={SUCCESS_BOX_STYLES}>
      <p style={MESSAGE_STYLES}>✅ Great work! No issues found</p>
      {hasAdditionalErrors && (
        <div className="button-group" style={{ marginTop: "var(--spacing-sm)" }}>
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
