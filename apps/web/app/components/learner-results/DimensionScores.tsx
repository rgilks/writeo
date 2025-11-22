/**
 * Dimension scores display component
 */

import { getScoreColor, getScoreLabel } from "./utils";

export function DimensionScores({
  dimensions,
  lowestDim,
}: {
  dimensions: {
    TA: number;
    CC: number;
    Vocab: number;
    Grammar: number;
  };
  lowestDim?: [string, number];
}) {
  return (
    <div
      style={{
        marginTop: "var(--spacing-md)",
        padding: "var(--spacing-sm) var(--spacing-md)",
        backgroundColor: "var(--bg-secondary)",
        borderRadius: "var(--border-radius)",
      }}
      lang="en"
    >
      <h3
        style={{
          fontSize: "16px",
          fontWeight: 600,
          marginBottom: "var(--spacing-sm)",
          color: "var(--text-primary)",
        }}
        lang="en"
      >
        How You Did
      </h3>
      <div className="dimensions-grid-responsive" lang="en">
        {[
          { key: "TA", label: "Answering the Question", score: dimensions.TA },
          { key: "CC", label: "Organization", score: dimensions.CC },
          { key: "Vocab", label: "Vocabulary", score: dimensions.Vocab },
          { key: "Grammar", label: "Grammar", score: dimensions.Grammar },
        ].map(({ key, label, score }) => {
          const isWeakest = lowestDim && lowestDim[0] === key;
          const scoreLabel = getScoreLabel(score);
          const scoreColor = getScoreColor(score);

          return (
            <div
              key={key}
              style={{
                padding: "10px 8px",
                backgroundColor: isWeakest ? "rgba(239, 68, 68, 0.08)" : "var(--bg-primary)",
                border: isWeakest
                  ? "2px solid rgba(239, 68, 68, 0.4)"
                  : "1px solid rgba(0, 0, 0, 0.08)",
                borderRadius: "var(--border-radius)",
                transition: "all 0.2s ease",
                position: "relative",
              }}
              lang="en"
            >
              <div
                style={{
                  fontSize: "28px",
                  fontWeight: 700,
                  color: scoreColor,
                  marginBottom: "4px",
                  lineHeight: "1",
                }}
                lang="en"
              >
                {score.toFixed(1)}
              </div>
              <div
                style={{
                  fontSize: "10px",
                  fontWeight: 600,
                  color: scoreColor,
                  marginBottom: "4px",
                  textTransform: "uppercase",
                  letterSpacing: "0.5px",
                }}
                lang="en"
              >
                {scoreLabel}
              </div>
              <div
                style={{
                  fontSize: "12px",
                  color: "var(--text-primary)",
                  lineHeight: "1.3",
                  fontWeight: 500,
                  marginBottom: "6px",
                }}
                lang="en"
              >
                {label}
              </div>
              <div
                style={{
                  height: "3px",
                  backgroundColor: "rgba(0, 0, 0, 0.1)",
                  borderRadius: "2px",
                  overflow: "hidden",
                }}
                lang="en"
              >
                <div
                  style={{
                    height: "100%",
                    width: `${(score / 9) * 100}%`,
                    backgroundColor: scoreColor,
                    borderRadius: "2px",
                    transition: "width 0.3s ease",
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
