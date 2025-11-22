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
  if (!grammarErrors || grammarErrors.length === 0 || !finalAnswerText) {
    return null;
  }

  return (
    <div className="card notranslate" translate="no" lang="en">
      <h2
        style={{
          fontSize: "20px",
          marginBottom: "var(--spacing-md)",
          fontWeight: 600,
        }}
        lang="en"
      >
        Your Writing with Feedback
      </h2>
      {isFeedbackRevealed && (
        <p
          style={{
            marginBottom: "var(--spacing-md)",
            fontSize: "14px",
            color: "var(--text-secondary)",
          }}
          lang="en"
        >
          Click on highlighted text to see suggestions for improvement.
        </p>
      )}
      <HeatMapText text={finalAnswerText} errors={grammarErrors} onReveal={onReveal} />
    </div>
  );
}
