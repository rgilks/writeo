/**
 * Editable essay section component
 */

import { EditableEssay } from "../EditableEssay";

export function EditableEssaySection({
  finalAnswerText,
  questionText,
  answerId,
  submissionId,
  onSubmit,
}: {
  finalAnswerText: string;
  questionText: string;
  answerId?: string;
  submissionId?: string;
  onSubmit: (editedText: string) => Promise<void>;
}) {
  if (!finalAnswerText) return null;

  if (!questionText) {
    return (
      <div
        lang="en"
        style={{
          marginTop: "var(--spacing-lg)",
          padding: "var(--spacing-lg)",
          backgroundColor: "var(--bg-secondary)",
          border: "2px solid rgba(139, 69, 19, 0.2)",
          borderRadius: "var(--border-radius-lg)",
        }}
      >
        <p
          lang="en"
          style={{
            fontSize: "14px",
            color: "var(--text-secondary)",
            fontStyle: "italic",
          }}
        >
          Note: Question text is not available for editing. You can still view your essay and
          feedback above.
        </p>
      </div>
    );
  }

  return (
    <EditableEssay
      initialText={finalAnswerText}
      questionId={answerId}
      questionText={questionText}
      parentSubmissionId={submissionId}
      onSubmit={onSubmit}
    />
  );
}
