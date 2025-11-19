"use client";

import { motion } from "framer-motion";

interface CelebrationMessageProps {
  scoreDiff?: number | null;
  errorCountDiff?: number | null;
  draftNumber: number;
}

/**
 * CelebrationMessage - Shows celebratory feedback when learners improve between drafts
 */
export function CelebrationMessage({
  scoreDiff,
  errorCountDiff,
  draftNumber,
}: CelebrationMessageProps) {
  // Only show if there's improvement
  const hasScoreImprovement = scoreDiff !== null && scoreDiff !== undefined && scoreDiff > 0;
  const hasErrorReduction =
    errorCountDiff !== null && errorCountDiff !== undefined && errorCountDiff < 0;

  if (!hasScoreImprovement && !hasErrorReduction) {
    return null;
  }

  const errorReductionCount = errorCountDiff ? Math.abs(errorCountDiff) : 0;
  const scoreImprovement = scoreDiff || 0;

  // Determine celebration message
  let message = "";
  let emoji = "🎉";

  if (hasScoreImprovement && hasErrorReduction) {
    // Both improved
    message = `Great! You fixed ${errorReductionCount} mistake${errorReductionCount !== 1 ? "s" : ""} and improved your score by ${scoreImprovement.toFixed(1)} point${scoreImprovement !== 1 ? "s" : ""}!`;
    emoji = "🌟";
  } else if (hasErrorReduction) {
    // Only errors fixed
    message = `🎉 Great! You fixed ${errorReductionCount} mistake${errorReductionCount !== 1 ? "s" : ""} in this draft!`;
    emoji = "🎉";
  } else if (hasScoreImprovement) {
    // Only score improved
    message = `📈 Your score improved by ${scoreImprovement.toFixed(1)} point${scoreImprovement !== 1 ? "s" : ""}!`;
    emoji = "📈";
  }

  // Additional encouragement for multiple drafts
  let encouragement = "";
  if (draftNumber >= 3) {
    encouragement = " Continue practicing to maintain your progress.";
  } else if (draftNumber === 2) {
    encouragement = " Keep revising to improve further.";
  }

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
        lang="en"
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
        lang="en"
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
          lang="en"
        >
          {encouragement}
        </p>
      )}
    </motion.div>
  );
}
