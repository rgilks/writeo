/**
 * Score display component
 */

import { CEFRBadge } from "../CEFRInfo";
import { getScoreColor, getScoreLabel, mapScoreToCEFR, getCEFRDescriptor } from "./utils";

export function ScoreDisplay({ overall }: { overall: number }) {
  // Check if we have a valid score (essay grading service may have failed)
  const hasValidScore = typeof overall === "number" && overall > 0;
  const scoreColor = hasValidScore ? getScoreColor(overall) : "var(--text-secondary)";
  const scoreLabelStyle = {
    marginBottom: "var(--spacing-md)",
    fontSize: hasValidScore ? "28px" : "20px",
    fontWeight: hasValidScore ? 700 : 600,
    color: hasValidScore ? "var(--text-primary)" : "var(--text-secondary)",
    marginTop: "var(--spacing-xs)",
  } as const;
  const infoBoxStyle = {
    marginTop: "var(--spacing-md)",
    padding: "var(--spacing-sm) var(--spacing-md)",
    borderRadius: "var(--border-radius)",
    fontSize: "14px",
    lineHeight: "1.5",
  } as const;

  if (!hasValidScore) {
    return (
      <div className="overall-score-section">
        <div className="overall-score-main">
          <div className="overall-score-value" style={{ color: scoreColor, fontSize: "2.5rem" }}>
            —
          </div>
          <div className="overall-score-label" lang="en">
            <div className="score-label-main">Your Writing Level</div>
            <div className="score-label-sub" style={scoreLabelStyle}>
              Score Unavailable
            </div>
            <div
              style={{
                ...infoBoxStyle,
                backgroundColor: "rgba(239, 68, 68, 0.08)",
                color: scoreColor,
              }}
              lang="en"
            >
              The essay scoring service is temporarily unavailable. Your grammar feedback is still
              available below. Please try again later for a full score.
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="overall-score-section">
      <div className="overall-score-main">
        <div className="overall-score-value" style={{ color: scoreColor }}>
          {overall.toFixed(1)}
        </div>
        <div className="overall-score-label" lang="en">
          <div className="score-label-main">Your Writing Level</div>
          <div className="score-label-sub" style={scoreLabelStyle}>
            {getScoreLabel(overall)}
          </div>
          <div
            style={{
              ...infoBoxStyle,
              backgroundColor: "rgba(59, 130, 246, 0.1)",
              display: "flex",
              alignItems: "center",
              gap: "var(--spacing-sm)",
              flexWrap: "wrap",
            }}
            lang="en"
          >
            <CEFRBadge level={mapScoreToCEFR(overall)} showLabel={true} />
            <span
              style={{
                fontSize: "14px",
                color: "var(--text-secondary)",
                flex: "1 1 auto",
                minWidth: "200px",
              }}
              lang="en"
            >
              {getCEFRDescriptor(mapScoreToCEFR(overall))}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
