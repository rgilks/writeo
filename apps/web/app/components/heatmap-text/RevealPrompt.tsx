import type { RevealPromptProps } from "./types";

const PROMPT_BOX_STYLES = {
  marginBottom: "var(--spacing-md)",
} as const;

const TITLE_STYLES = {
  marginBottom: "var(--spacing-sm)",
  fontWeight: 600,
  color: "var(--text-primary)",
} as const;

const DESCRIPTION_STYLES = {
  marginBottom: "var(--spacing-md)",
  color: "var(--text-secondary)",
  fontSize: "14px",
} as const;

// BUTTON_GROUP_STYLES removed - use .button-group class instead

const PRIMARY_BUTTON_STYLES = {
  fontSize: "14px",
  padding: "var(--spacing-sm) var(--spacing-md)",
} as const;

const SECONDARY_BUTTON_STYLES = {
  fontSize: "13px",
  padding: "var(--spacing-xs) var(--spacing-sm)",
} as const;

interface AdditionalButtonProps {
  count: number;
  isVisible: boolean;
  label: string;
  onClick: () => void;
}

function AdditionalButton({ count, isVisible, label, onClick }: AdditionalButtonProps) {
  if (count === 0 || isVisible) return null;

  return (
    <button onClick={onClick} className="btn btn-secondary" style={SECONDARY_BUTTON_STYLES}>
      +{count} {label}
    </button>
  );
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
  const hasAdditionalErrors = mediumConfidenceErrors.length > 0 || lowConfidenceErrors.length > 0;

  return (
    <div className="info-box info-box-primary" style={PROMPT_BOX_STYLES}>
      <p style={TITLE_STYLES}>💡 Find the Mistakes</p>
      <p style={DESCRIPTION_STYLES}>
        Look for the red highlights in your text. Can you spot what needs fixing? Click "Show
        Feedback" when you're ready to see suggestions.
      </p>
      <div className="button-group">
        <button onClick={onReveal} className="btn btn-primary" style={PRIMARY_BUTTON_STYLES}>
          Show Feedback
        </button>
        {hasAdditionalErrors && (
          <>
            <AdditionalButton
              count={mediumConfidenceErrors.length}
              isVisible={showMediumConfidenceErrors}
              label="more"
              onClick={onShowMediumConfidence}
            />
            <AdditionalButton
              count={lowConfidenceErrors.length}
              isVisible={showExperimentalSuggestions}
              label="experimental"
              onClick={onShowExperimental}
            />
          </>
        )}
      </div>
    </div>
  );
}
