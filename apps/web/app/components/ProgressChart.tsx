"use client";

import { useMemo } from "react";
import { motion } from "framer-motion";
import type { DraftHistory } from "@/app/lib/stores/draft-store";

const CEFR_LEVELS = ["A2", "B1", "B2", "C1", "C2"] as const;

interface ProgressChartProps {
  draftHistory: DraftHistory[];
  type?: "score" | "errors" | "cefr";
}

const MIN_DRAFTS_FOR_PROGRESS = 2;
const CHART_MIN = 0;
const CHART_MAX = 9;
const CHART_RANGE = CHART_MAX - CHART_MIN;
const SCORE_CHART_HEIGHT = "200px";
const ERROR_CHART_HEIGHT = "120px";
const ANIMATION_DELAY_STEP = 0.1;
const GRID_LINE_RATIOS = [0, 0.5, 1] as const;

const CONTAINER_STYLE = {
  padding: "var(--spacing-md)",
  backgroundColor: "var(--bg-secondary)",
  borderRadius: "var(--border-radius)",
} as const;

const HEADING_STYLE = {
  fontSize: "16px",
  fontWeight: 600,
  marginBottom: "var(--spacing-md)",
  color: "var(--text-primary)",
} as const;

const GRID_LINE_STYLE = {
  width: "100%",
  height: "1px",
  backgroundColor: "rgba(0, 0, 0, 0.1)",
} as const;

/**
 * ProgressChart - Visualizes progress across drafts
 */
export function ProgressChart({ draftHistory, type = "score" }: ProgressChartProps) {
  if (draftHistory.length < MIN_DRAFTS_FOR_PROGRESS) {
    return null;
  }

  const maxErrors = useMemo(
    () => Math.max(...draftHistory.map((d) => d.errorCount || 0)),
    [draftHistory],
  );

  if (type === "score") {
    const scoresWithTrends = useMemo(
      () =>
        draftHistory.map((draft, index) => {
          const score = draft.overallScore || 0;
          const prevScore = index > 0 ? draftHistory[index - 1].overallScore || 0 : null;
          const diff = prevScore !== null ? score - prevScore : null;
          return { draft, score, diff, index };
        }),
      [draftHistory],
    );

    return (
      <div lang="en" style={CONTAINER_STYLE}>
        <h3 style={HEADING_STYLE}>Score Progress Across Drafts</h3>
        <div
          style={{
            position: "relative",
            height: SCORE_CHART_HEIGHT,
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
            <span>{CHART_MAX.toFixed(1)}</span>
            <span>{((CHART_MAX + CHART_MIN) / 2).toFixed(1)}</span>
            <span>{CHART_MIN.toFixed(1)}</span>
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
              {GRID_LINE_RATIOS.map((ratio) => (
                <div key={ratio} style={GRID_LINE_STYLE} />
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
                const height = score > 0 ? ((score - CHART_MIN) / CHART_RANGE) * 100 : 0;
                const isImproving = diff !== null && diff > 0;
                const isDeclining = diff !== null && diff < 0;

                return (
                  <div
                    key={`draft-${draft.draftNumber}-${draft.submissionId || draft.timestamp || index}`}
                    style={{
                      flex: 1,
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      justifyContent: "flex-end",
                      position: "relative",
                      height: "100%",
                    }}
                  >
                    {/* Score bar */}
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: `${height}%`, opacity: 1 }}
                      transition={{ duration: 0.5, delay: index * ANIMATION_DELAY_STEP }}
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
                    >
                      Draft {draft.draftNumber}
                    </div>
                  </div>
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
      <div lang="en" style={CONTAINER_STYLE}>
        <h3 style={HEADING_STYLE}>Error Reduction</h3>
        <div
          style={{
            display: "flex",
            alignItems: "flex-end",
            gap: "var(--spacing-sm)",
            height: ERROR_CHART_HEIGHT,
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
                transition={{
                  duration: 0.6,
                  delay: index * ANIMATION_DELAY_STEP,
                  ease: [0.4, 0, 0.2, 1],
                }}
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
                >
                  {errorCount}
                </span>
                <span
                  style={{
                    fontSize: "14px",
                    color: "rgba(255, 255, 255, 0.8)",
                  }}
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
  const getCefrIndex = (level?: string): number => {
    if (!level) return 0;
    return CEFR_LEVELS.indexOf(level as (typeof CEFR_LEVELS)[number]) + 1;
  };

  const lastDraft = useMemo(() => draftHistory[draftHistory.length - 1], [draftHistory]);

  return (
    <div lang="en" style={CONTAINER_STYLE}>
      <h3 style={HEADING_STYLE}>CEFR Level Progress</h3>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--spacing-sm)",
          padding: "var(--spacing-sm)",
        }}
      >
        {CEFR_LEVELS.map((level, index) => {
          const hasReached = draftHistory.some((d) => getCefrIndex(d.cefrLevel) >= index + 1);
          const isCurrent = lastDraft?.cefrLevel === level;

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
            >
              {level}
            </div>
          );
        })}
      </div>
    </div>
  );
}
