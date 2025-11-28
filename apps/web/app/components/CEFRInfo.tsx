"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

type CEFRLevel = "A2" | "B1" | "B2" | "C1" | "C2";

interface CEFRInfoProps {
  level: string;
  showFullInfo?: boolean;
}

interface CEFRDescription {
  label: string;
  description: string;
  tips: string[];
}

const CEFR_LEVELS: readonly CEFRLevel[] = ["A2", "B1", "B2", "C1", "C2"] as const;

const CEFR_DESCRIPTIONS: Record<CEFRLevel, CEFRDescription> = {
  A2: {
    label: "Elementary",
    description: "You can write simple connected text on familiar topics.",
    tips: [
      "Use simple sentence structures",
      "Focus on basic grammar accuracy",
      "Practice writing about everyday topics",
      "Build your vocabulary gradually",
    ],
  },
  B1: {
    label: "Intermediate",
    description:
      "You can write simple connected text on topics which are familiar or of personal interest.",
    tips: [
      "Try using more complex sentences",
      "Vary your sentence structure",
      "Use linking words to connect ideas",
      "Expand your vocabulary with synonyms",
    ],
  },
  B2: {
    label: "Upper Intermediate",
    description: "You can write clear, detailed text on a wide range of subjects.",
    tips: [
      "Use more sophisticated vocabulary",
      "Write longer, more complex sentences",
      "Organize ideas into clear paragraphs",
      "Express opinions with supporting reasons",
    ],
  },
  C1: {
    label: "Advanced",
    description: "You can write clear, well-structured text on complex subjects.",
    tips: [
      "Use advanced grammar structures",
      "Write with precision and nuance",
      "Develop arguments with evidence",
      "Adapt your writing style to the context",
    ],
  },
  C2: {
    label: "Proficient",
    description: "You can write clear, smoothly flowing text in an appropriate style.",
    tips: [
      "Master complex grammar structures",
      "Write with natural fluency",
      "Use idiomatic expressions appropriately",
      "Adapt style for different audiences",
    ],
  },
};

export function CEFRInfo({ level, showFullInfo = false }: CEFRInfoProps) {
  const [isExpanded, setIsExpanded] = useState(showFullInfo);
  const info = CEFR_DESCRIPTIONS[level.toUpperCase() as CEFRLevel];

  if (!info) {
    return null;
  }

  return (
    <div
      lang="en"
      style={{
        padding: "var(--spacing-md)",
        backgroundColor: "var(--bg-secondary)",
        borderRadius: "var(--border-radius)",
        border: "1px solid var(--border-color)",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: isExpanded ? "var(--spacing-md)" : 0,
        }}
      >
        <div>
          <h3
            style={{
              fontSize: "18px",
              fontWeight: 600,
              color: "var(--text-primary)",
              marginBottom: "var(--spacing-xs)",
            }}
          >
            CEFR {level} - {info.label}
          </h3>
          <p
            style={{
              fontSize: "14px",
              color: "var(--text-secondary)",
              lineHeight: "1.5",
            }}
          >
            {info.description}
          </p>
        </div>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          style={{
            padding: "var(--spacing-sm) var(--spacing-md)",
            minHeight: "44px",
            backgroundColor: "var(--primary-color)",
            color: "white",
            border: "none",
            borderRadius: "var(--border-radius)",
            cursor: "pointer",
            fontSize: "14px",
            fontWeight: 600,
            marginLeft: "var(--spacing-md)",
            whiteSpace: "nowrap",
            flexShrink: 0,
          }}
        >
          {isExpanded ? "Show Less" : "Show Tips"}
        </button>
      </div>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
          >
            <div
              style={{
                marginTop: "var(--spacing-md)",
                paddingTop: "var(--spacing-md)",
                borderTop: "1px solid var(--border-color)",
              }}
            >
              <h4
                style={{
                  fontSize: "16px",
                  fontWeight: 600,
                  color: "var(--text-primary)",
                  marginBottom: "var(--spacing-sm)",
                }}
              >
                Tips to Reach {getNextLevel(level)}
              </h4>
              <ul
                style={{
                  margin: 0,
                  paddingLeft: "var(--spacing-lg)",
                  color: "var(--text-secondary)",
                  fontSize: "14px",
                  lineHeight: "1.5",
                }}
              >
                {info.tips.map((tip: string, index: number) => (
                  <li key={index}>{tip}</li>
                ))}
              </ul>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function getNextLevel(level: string): string {
  const index = CEFR_LEVELS.indexOf(level.toUpperCase() as CEFRLevel);
  return index >= 0 && index < CEFR_LEVELS.length - 1 ? CEFR_LEVELS[index + 1] : "the next level";
}

interface CEFRBadgeProps {
  level: string;
  showLabel?: boolean;
}

/**
 * CEFRBadge - Simple badge showing CEFR level
 */
export function CEFRBadge({ level, showLabel = true }: CEFRBadgeProps) {
  const info = CEFR_DESCRIPTIONS[level.toUpperCase() as CEFRLevel];

  return (
    <div
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "var(--spacing-xs)",
        padding: "var(--spacing-xs) var(--spacing-sm)",
        backgroundColor: "var(--bg-secondary)",
        borderRadius: "var(--border-radius)",
        border: "1px solid var(--border-color)",
      }}
      lang="en"
    >
      <span
        style={{
          fontSize: "14px",
          fontWeight: 600,
          color: "var(--primary-color)",
        }}
      >
        {level}
      </span>
      {showLabel && info && (
        <span
          style={{
            fontSize: "14px",
            color: "var(--text-secondary)",
          }}
        >
          {info.label}
        </span>
      )}
    </div>
  );
}
