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

  // Always render EditableEssay when finalAnswerText exists, even if questionText is empty
  // This allows users to edit and resubmit in free writing mode
  // EditableEssay component handles empty questionText gracefully
  return (
    <EditableEssay
      initialText={finalAnswerText}
      questionId={answerId}
      questionText={questionText || ""} // Pass empty string if questionText is not provided
      parentSubmissionId={submissionId}
      onSubmit={onSubmit}
    />
  );
}
