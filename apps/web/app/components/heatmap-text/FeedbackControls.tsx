import type { FeedbackControlsProps } from "./types";
import { pluralize } from "@/app/lib/utils/text-utils";

const CONTAINER_STYLES = {
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
} as const;

const BUTTON_GROUP_STYLES = {
  display: "flex",
  gap: "var(--spacing-sm)",
  flexWrap: "wrap",
  alignItems: "center",
} as const;

const BUTTON_STYLES = {
  fontSize: "13px",
  padding: "var(--spacing-xs) var(--spacing-sm)",
} as const;

interface ToggleButtonProps {
  count: number;
  isVisible: boolean;
  label: string;
  onClick: () => void;
}

function ToggleButton({ count, isVisible, label, onClick }: ToggleButtonProps) {
  if (count === 0) return null;

  return (
    <button onClick={onClick} className="btn btn-secondary" style={BUTTON_STYLES}>
      {isVisible ? "Hide" : "Show"} {count} {label}
    </button>
  );
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
    <div style={CONTAINER_STYLES}>
      <div style={BUTTON_GROUP_STYLES}>
        <ToggleButton
          count={mediumConfidenceErrors.length}
          isVisible={showMediumConfidenceErrors}
          label={`more ${pluralize(mediumConfidenceErrors.length, "suggestion")}`}
          onClick={onToggleMediumConfidence}
        />
        <ToggleButton
          count={lowConfidenceErrors.length}
          isVisible={showExperimentalSuggestions}
          label={`experimental ${pluralize(lowConfidenceErrors.length, "suggestion")}`}
          onClick={onToggleExperimental}
        />
      </div>
      <button onClick={onHideFeedback} className="btn btn-secondary" style={BUTTON_STYLES}>
        Hide Feedback
      </button>
    </div>
  );
}
