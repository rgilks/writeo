"use client";

import { motion } from "framer-motion";
import { useMemo } from "react";
import { pluralize } from "@/app/lib/utils/text-utils";

interface CelebrationMessageProps {
  scoreDiff?: number | null;
  errorCountDiff?: number | null;
  draftNumber: number;
}

const ENCOURAGEMENT_MESSAGES: Record<number, string> = {
  2: " Keep revising to improve further.",
  3: " Continue practicing to maintain your progress.",
};

/**
 * CelebrationMessage - Shows celebratory feedback when learners improve between drafts
 */
export function CelebrationMessage({
  scoreDiff,
  errorCountDiff,
  draftNumber,
}: CelebrationMessageProps) {
  const hasScoreImprovement = (scoreDiff ?? 0) > 0;
  const hasErrorReduction = (errorCountDiff ?? 0) < 0;

  if (!hasScoreImprovement && !hasErrorReduction) {
    return null;
  }

  const { message, emoji } = useMemo(() => {
    const errorReductionCount = errorCountDiff ? Math.abs(errorCountDiff) : 0;
    const scoreImprovement = scoreDiff || 0;

    if (hasScoreImprovement && hasErrorReduction) {
      return {
        message: `Great! You fixed ${errorReductionCount} ${pluralize(errorReductionCount, "mistake")} and improved your score by ${scoreImprovement.toFixed(1)} ${pluralize(scoreImprovement, "point")}!`,
        emoji: "🌟",
      };
    }

    if (hasErrorReduction) {
      return {
        message: `Great! You fixed ${errorReductionCount} ${pluralize(errorReductionCount, "mistake")} in this draft!`,
        emoji: "🎉",
      };
    }

    return {
      message: `Your score improved by ${scoreImprovement.toFixed(1)} ${pluralize(scoreImprovement, "point")}!`,
      emoji: "📈",
    };
  }, [hasScoreImprovement, hasErrorReduction, scoreDiff, errorCountDiff]);

  const encouragement = useMemo(() => {
    return (
      ENCOURAGEMENT_MESSAGES[draftNumber] || (draftNumber >= 3 ? ENCOURAGEMENT_MESSAGES[3] : "")
    );
  }, [draftNumber]);

  return (
    <motion.div
      initial={{ opacity: 0, y: -20, scale: 0.9 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.5, type: "spring", stiffness: 300, damping: 25 }}
      style={{
        marginBottom: "var(--spacing-lg)",
        padding: "var(--spacing-lg)",
        backgroundColor: "rgba(16, 185, 129, 0.1)",
        border: "2px solid rgba(16, 185, 129, 0.3)",
        borderRadius: "var(--border-radius-lg)",
        textAlign: "center",
      }}
      lang="en"
    >
      <div
        style={{
          fontSize: "48px",
          marginBottom: "var(--spacing-sm)",
        }}
      >
        {emoji}
      </div>
      <h3
        style={{
          fontSize: "20px",
          fontWeight: 600,
          marginBottom: "var(--spacing-xs)",
          color: "var(--text-primary)",
        }}
      >
        {message}
      </h3>
      {encouragement && (
        <p
          style={{
            fontSize: "16px",
            color: "var(--text-secondary)",
            marginTop: "var(--spacing-sm)",
          }}
        >
          {encouragement}
        </p>
      )}
    </motion.div>
  );
}
