"use client";

import { motion } from "framer-motion";
import type { DraftHistory } from "@/app/lib/stores/draft-store";

interface ProgressChartProps {
  draftHistory: DraftHistory[];
  type?: "score" | "errors" | "cefr";
}

/**
 * ProgressChart - Visualizes progress across drafts
 */
export function ProgressChart({ draftHistory, type = "score" }: ProgressChartProps) {
  if (draftHistory.length < 2) {
    return null; // Need at least 2 drafts to show progress
  }

  const maxScore = Math.max(...draftHistory.map((d) => d.overallScore || 0).filter((s) => s > 0));
  const minScore = Math.min(...draftHistory.map((d) => d.overallScore || 0).filter((s) => s > 0));
  const scoreRange = maxScore - minScore || 1;

  const maxErrors = Math.max(...draftHistory.map((d) => d.errorCount || 0));

  if (type === "score") {
    // Calculate score differences for trend indicators
    const scoresWithTrends = draftHistory.map((draft, index) => {
      const score = draft.overallScore || 0;
      const prevScore = index > 0 ? draftHistory[index - 1].overallScore || 0 : null;
      const diff = prevScore !== null ? score - prevScore : null;
      return { draft, score, diff, index };
    });

    // Use a fixed scale from 0 to 9 for better comparison
    const chartMin = 0;
    const chartMax = 9;
    const chartRange = chartMax - chartMin;

    return (
      <div
        lang="en"
        style={{
          padding: "var(--spacing-md)",
          backgroundColor: "var(--bg-secondary)",
          borderRadius: "var(--border-radius)",
        }}
      >
        <h3
          style={{
            fontSize: "16px",
            fontWeight: 600,
            marginBottom: "var(--spacing-md)",
            color: "var(--text-primary)",
          }}
          lang="en"
        >
          Score Progress Across Drafts
        </h3>
        <div
          style={{
            position: "relative",
            height: "200px",
            padding: "var(--spacing-md)",
            paddingBottom: "40px",
          }}
        >
          {/* Y-axis labels */}
          <div
            style={{
              position: "absolute",
              left: 0,
              top: 0,
              bottom: "40px",
              width: "30px",
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between",
              fontSize: "11px",
              color: "var(--text-secondary)",
            }}
          >
            <span>{chartMax.toFixed(1)}</span>
            <span>{((chartMax + chartMin) / 2).toFixed(1)}</span>
            <span>{chartMin.toFixed(1)}</span>
          </div>

          {/* Chart area */}
          <div
            style={{
              marginLeft: "40px",
              position: "relative",
              height: "100%",
              display: "flex",
              alignItems: "flex-end",
            }}
          >
            {/* Grid lines */}
            <div
              style={{
                position: "absolute",
                inset: 0,
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between",
              }}
            >
              {[0, 0.5, 1].map((ratio) => (
                <div
                  key={ratio}
                  style={{
                    width: "100%",
                    height: "1px",
                    backgroundColor: "rgba(0, 0, 0, 0.1)",
                  }}
                />
              ))}
            </div>

            {/* Bars and line */}
            <div
              style={{
                position: "relative",
                width: "100%",
                height: "100%",
                display: "flex",
                alignItems: "flex-end",
                gap: "var(--spacing-xs)",
              }}
            >
              {/* Bars with scores */}
              {scoresWithTrends.map(({ draft, score, diff, index }) => {
                const height = score > 0 ? ((score - chartMin) / chartRange) * 100 : 0;
                const isImproving = diff !== null && diff > 0;
                const isDeclining = diff !== null && diff < 0;

                return (
                  <motion.div
                    key={`draft-${draft.draftNumber}-${draft.submissionId || draft.timestamp || index}`}
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: `${height}%`, opacity: 1 }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                    style={{
                      flex: 1,
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      justifyContent: "flex-end",
                      position: "relative",
                    }}
                  >
                    {/* Score bar */}
                    <motion.div
                      style={{
                        width: "100%",
                        backgroundColor: isImproving
                          ? "var(--secondary-accent)"
                          : isDeclining
                            ? "var(--error-color)"
                            : "var(--primary-color)",
                        borderRadius: "var(--spacing-xs) var(--spacing-xs) 0 0",
                        minHeight: "8px",
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        justifyContent: "flex-end",
                        padding: "var(--spacing-xs)",
                        position: "relative",
                      }}
                    >
                      {/* Score value */}
                      <span
                        style={{
                          fontSize: "12px",
                          color: "white",
                          fontWeight: 600,
                          marginBottom: "2px",
                          whiteSpace: "nowrap",
                        }}
                        lang="en"
                      >
                        {score > 0 ? score.toFixed(1) : "-"}
                      </span>
                    </motion.div>

                    {/* Trend indicator */}
                    {diff !== null && diff !== 0 && (
                      <div
                        style={{
                          fontSize: "10px",
                          color: isImproving ? "var(--secondary-accent)" : "var(--error-color)",
                          fontWeight: 600,
                          marginTop: "4px",
                        }}
                        lang="en"
                      >
                        {diff > 0 ? "+" : ""}
                        {diff.toFixed(1)}
                      </div>
                    )}

                    {/* Draft label */}
                    <div
                      style={{
                        fontSize: "12px",
                        color: "var(--text-secondary)",
                        marginTop: "8px",
                        fontWeight: 500,
                      }}
                      lang="en"
                    >
                      Draft {draft.draftNumber}
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (type === "errors") {
    return (
      <div
        lang="en"
        style={{
          padding: "var(--spacing-md)",
          backgroundColor: "var(--bg-secondary)",
          borderRadius: "var(--border-radius)",
        }}
      >
        <h3
          style={{
            fontSize: "16px",
            fontWeight: 600,
            marginBottom: "var(--spacing-md)",
            color: "var(--text-primary)",
          }}
          lang="en"
        >
          Error Reduction
        </h3>
        <div
          style={{
            display: "flex",
            alignItems: "flex-end",
            gap: "var(--spacing-sm)",
            height: "120px",
            padding: "var(--spacing-sm)",
          }}
        >
          {draftHistory.map((draft, index) => {
            const errorCount = draft.errorCount || 0;
            const height = maxErrors > 0 ? (errorCount / maxErrors) * 100 : 0;
            const isImproving = index > 0 && errorCount < (draftHistory[index - 1].errorCount || 0);

            return (
              <motion.div
                key={draft.submissionId}
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: `${height}%`, opacity: 1 }}
                transition={{ duration: 0.6, delay: index * 0.1, ease: [0.4, 0, 0.2, 1] }}
                style={{
                  flex: 1,
                  backgroundColor: isImproving ? "var(--secondary-accent)" : "var(--error-color)",
                  borderRadius: "var(--spacing-xs) var(--spacing-xs) 0 0",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "flex-end",
                  padding: "var(--spacing-xs)",
                  minHeight: "24px",
                }}
              >
                <span
                  style={{
                    fontSize: "14px",
                    color: "white",
                    fontWeight: 600,
                    marginBottom: "var(--spacing-xs)",
                  }}
                  lang="en"
                >
                  {errorCount}
                </span>
                <span
                  style={{
                    fontSize: "14px",
                    color: "rgba(255, 255, 255, 0.8)",
                  }}
                  lang="en"
                >
                  D{draft.draftNumber}
                </span>
              </motion.div>
            );
          })}
        </div>
      </div>
    );
  }

  // CEFR level progress
  const cefrLevels = ["A2", "B1", "B2", "C1", "C2"];
  const getCefrIndex = (level?: string): number => {
    if (!level) return 0;
    return cefrLevels.indexOf(level) + 1;
  };

  return (
    <div
      lang="en"
      style={{
        padding: "var(--spacing-md)",
        backgroundColor: "var(--bg-secondary)",
        borderRadius: "var(--border-radius)",
      }}
    >
      <h3
        style={{
          fontSize: "16px",
          fontWeight: 600,
          marginBottom: "var(--spacing-md)",
          color: "var(--text-primary)",
        }}
        lang="en"
      >
        CEFR Level Progress
      </h3>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--spacing-sm)",
          padding: "var(--spacing-sm)",
        }}
      >
        {cefrLevels.map((level, index) => {
          const hasReached = draftHistory.some((d) => getCefrIndex(d.cefrLevel) >= index + 1);
          const isCurrent = draftHistory[draftHistory.length - 1]?.cefrLevel === level;

          return (
            <div
              key={level}
              style={{
                flex: 1,
                textAlign: "center",
                padding: "var(--spacing-sm)",
                backgroundColor: isCurrent
                  ? "var(--primary-color)"
                  : hasReached
                    ? "var(--secondary-accent)"
                    : "var(--bg-tertiary)",
                color: isCurrent || hasReached ? "white" : "var(--text-secondary)",
                borderRadius: "var(--border-radius)",
                fontWeight: isCurrent ? 600 : 400,
                fontSize: "14px",
              }}
              lang="en"
            >
              {level}
            </div>
          );
        })}
      </div>
    </div>
  );
}
