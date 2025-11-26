/**
 * Dimension scores display component
 */

import { getScoreColor, getScoreLabel } from "./utils";

export function DimensionScores({
  dimensions,
  lowestDim,
  questionText,
}: {
  dimensions: {
    TA: number;
    CC: number;
    Vocab: number;
    Grammar: number;
  };
  lowestDim?: [string, number];
  questionText?: string;
}) {
  // Build dimensions array, excluding TA if there's no question
  const hasQuestion = questionText && questionText.trim().length > 0;
  const dimensionItems = [
    ...(hasQuestion ? [{ key: "TA", label: "Answering the Question", score: dimensions.TA }] : []),
    { key: "CC", label: "Organization", score: dimensions.CC },
    { key: "Vocab", label: "Vocabulary", score: dimensions.Vocab },
    { key: "Grammar", label: "Grammar", score: dimensions.Grammar },
  ];

  const columnCount = dimensionItems.length;

  // Check if all scores are 0 (essay grading service may have failed)
  const allScoresZero = dimensionItems.every((item) => item.score === 0);

  if (allScoresZero) {
    return null; // Don't show dimension scores when scoring service failed
  }

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
      <div
        className="dimensions-grid-responsive"
        style={{
          gridTemplateColumns:
            columnCount === 3
              ? "repeat(3, 1fr)"
              : columnCount === 4
                ? "repeat(4, 1fr)"
                : `repeat(${columnCount}, 1fr)`,
        }}
        lang="en"
      >
        {dimensionItems.map(({ key, label, score }) => {
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
