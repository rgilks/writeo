"use client";

interface QuestionCardProps {
  isCustom: boolean;
  customQuestion: string;
  prompt: string;
  onCustomQuestionChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  disabled?: boolean;
}

export function QuestionCard({
  isCustom,
  customQuestion,
  prompt,
  onCustomQuestionChange,
  disabled = false,
}: QuestionCardProps) {
  return (
    <div className="card question-card">
      <h2
        style={{
          fontSize: "20px",
          marginBottom: "var(--spacing-md)",
          display: "flex",
          alignItems: "center",
          gap: "var(--spacing-sm)",
        }}
      >
        <span>📝</span> {isCustom ? "Your Question (Optional)" : "Question"}
      </h2>
      {isCustom ? (
        <>
          <textarea
            className="textarea notranslate"
            value={customQuestion}
            onChange={onCustomQuestionChange}
            placeholder="Enter your question here, or leave blank for free writing practice..."
            rows={4}
            disabled={disabled}
            translate="no"
            style={{
              width: "100%",
              minHeight: "80px",
              resize: "vertical",
            }}
          />
          {!customQuestion.trim() && (
            <p
              style={{
                marginTop: "var(--spacing-sm)",
                fontSize: "14px",
                color: "var(--text-secondary)",
                fontStyle: "italic",
              }}
            >
              💡 Leave blank to practice free writing without answering a specific question.
            </p>
          )}
        </>
      ) : (
        <div className="prompt-box notranslate" style={{ whiteSpace: "pre-wrap" }} translate="no">
          {prompt}
        </div>
      )}
    </div>
  );
}
