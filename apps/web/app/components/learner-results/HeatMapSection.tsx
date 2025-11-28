/**
 * Heat map section component
 */

import { HeatMapText } from "../HeatMapText";
import type { LanguageToolError } from "@writeo/shared";

export function HeatMapSection({
  grammarErrors,
  finalAnswerText,
  isFeedbackRevealed,
  onReveal,
}: {
  grammarErrors: LanguageToolError[];
  finalAnswerText: string;
  isFeedbackRevealed: boolean;
  onReveal: () => void;
}) {
  if (!grammarErrors?.length || !finalAnswerText) {
    return null;
  }

  const headingStyle = {
    fontSize: "20px",
    marginBottom: "var(--spacing-md)",
    fontWeight: 600,
  } as const;

  const instructionsStyle = {
    marginBottom: "var(--spacing-md)",
    fontSize: "14px",
    color: "var(--text-secondary)",
  } as const;

  return (
    <div className="card notranslate" translate="no" lang="en">
      <h2 style={headingStyle} lang="en">
        Your Writing with Feedback
      </h2>
      {isFeedbackRevealed && (
        <p style={instructionsStyle} lang="en">
          Click on highlighted text to see suggestions for improvement.
        </p>
      )}
      <HeatMapText text={finalAnswerText} errors={grammarErrors} onReveal={onReveal} />
    </div>
  );
}
