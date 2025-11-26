/**
 * Score display component
 */

import { CEFRBadge } from "../CEFRInfo";
import { getScoreColor, getScoreLabel, mapScoreToCEFR, getCEFRDescriptor } from "./utils";

export function ScoreDisplay({ overall }: { overall: number }) {
  // Check if we have a valid score (essay grading service may have failed)
  const hasValidScore = overall > 0;

  if (!hasValidScore) {
    return (
      <div className="overall-score-section">
        <div className="overall-score-main">
          <div
            className="overall-score-value"
            style={{ color: "var(--text-secondary)", fontSize: "2.5rem" }}
          >
            —
          </div>
          <div className="overall-score-label" lang="en">
            <div className="score-label-main">Your Writing Level</div>
            <div
              className="score-label-sub"
              style={{
                marginBottom: "var(--spacing-md)",
                fontSize: "20px",
                fontWeight: 600,
                color: "var(--text-secondary)",
                marginTop: "var(--spacing-xs)",
              }}
            >
              Score Unavailable
            </div>
            <div
              style={{
                marginTop: "var(--spacing-md)",
                padding: "var(--spacing-sm) var(--spacing-md)",
                backgroundColor: "rgba(239, 68, 68, 0.08)",
                borderRadius: "var(--border-radius)",
                color: "var(--text-secondary)",
                fontSize: "14px",
                lineHeight: "1.5",
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
        <div className="overall-score-value" style={{ color: getScoreColor(overall) }}>
          {overall.toFixed(1)}
        </div>
        <div className="overall-score-label" lang="en">
          <div className="score-label-main">Your Writing Level</div>
          <div
            className="score-label-sub"
            style={{
              marginBottom: "var(--spacing-md)",
              fontSize: "28px",
              fontWeight: 700,
              color: "var(--text-primary)",
              marginTop: "var(--spacing-xs)",
            }}
          >
            {getScoreLabel(overall)}
          </div>
          <div
            style={{
              marginTop: "var(--spacing-md)",
              padding: "var(--spacing-sm) var(--spacing-md)",
              backgroundColor: "rgba(59, 130, 246, 0.1)",
              borderRadius: "var(--border-radius)",
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
