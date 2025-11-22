/**
 * Score display component
 */

import { CEFRBadge } from "../CEFRInfo";
import { getScoreColor, getScoreLabel, mapScoreToCEFR, getCEFRDescriptor } from "./utils";

export function ScoreDisplay({ overall }: { overall: number }) {
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
