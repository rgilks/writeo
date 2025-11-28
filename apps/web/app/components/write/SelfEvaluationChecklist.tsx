"use client";

interface SelfEvaluationChecklistProps {
  selfEval: {
    answeredAllParts: boolean;
    supportedOpinion: boolean;
    variedStructure: boolean;
  };
  onSelfEvalChange: (updates: Partial<SelfEvaluationChecklistProps["selfEval"]>) => void;
  showAnsweredAllParts: boolean;
}

export function SelfEvaluationChecklist({
  selfEval,
  onSelfEvalChange,
  showAnsweredAllParts,
}: SelfEvaluationChecklistProps) {
  return (
    <div
      style={{
        marginTop: "var(--spacing-md)",
        padding: "var(--spacing-md)",
        backgroundColor: "rgba(102, 126, 234, 0.1)",
        borderRadius: "var(--border-radius)",
      }}
    >
      <p
        style={{
          marginBottom: "var(--spacing-sm)",
          fontSize: "14px",
          fontWeight: 600,
        }}
      >
        ✓ Self-Evaluation Checklist (optional)
      </p>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "var(--spacing-sm)",
        }}
      >
        {showAnsweredAllParts && (
          <label
            style={{
              display: "flex",
              alignItems: "center",
              gap: "var(--spacing-sm)",
              fontSize: "14px",
              cursor: "pointer",
              lineHeight: "1.5",
            }}
          >
            <input
              type="checkbox"
              checked={selfEval.answeredAllParts}
              onChange={(e) => onSelfEvalChange({ answeredAllParts: e.target.checked })}
              style={{ cursor: "pointer" }}
            />
            Did I answer all parts of the question?
          </label>
        )}
        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: "var(--spacing-sm)",
            fontSize: "14px",
            cursor: "pointer",
            lineHeight: "1.5",
          }}
        >
          <input
            type="checkbox"
            checked={selfEval.supportedOpinion}
            onChange={(e) => onSelfEvalChange({ supportedOpinion: e.target.checked })}
            style={{ cursor: "pointer" }}
          />
          Did I support my opinion with at least two reasons?
        </label>
        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: "var(--spacing-sm)",
            fontSize: "14px",
            cursor: "pointer",
            lineHeight: "1.5",
          }}
        >
          <input
            type="checkbox"
            checked={selfEval.variedStructure}
            onChange={(e) => onSelfEvalChange({ variedStructure: e.target.checked })}
            style={{ cursor: "pointer" }}
          />
          Did I vary my sentence structure?
        </label>
      </div>
    </div>
  );
}
