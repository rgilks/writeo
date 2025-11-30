"use client";

import { MIN_ESSAY_WORDS, MAX_ESSAY_WORDS } from "@writeo/shared";

interface WordCountDisplayProps {
  wordCount: number;
}

export function WordCountDisplay({ wordCount }: WordCountDisplayProps) {
  return (
    <div
      data-testid="word-count-display"
      style={{
        display: "flex",
        gap: "var(--spacing-md)",
        alignItems: "center",
        fontSize: "14px",
        color: "var(--text-secondary)",
      }}
    >
      <span aria-live="polite" aria-atomic="true" data-testid="word-count-value">
        {wordCount} {wordCount === 1 ? "word" : "words"}
      </span>
      {wordCount < MIN_ESSAY_WORDS && (
        <span
          style={{ color: "var(--error-color)", fontWeight: 600 }}
          role="status"
          aria-live="polite"
        >
          (Need at least {MIN_ESSAY_WORDS} words)
        </span>
      )}
      {wordCount >= MIN_ESSAY_WORDS && wordCount <= MAX_ESSAY_WORDS && (
        <span style={{ color: "var(--secondary-accent)" }} aria-label="Word count valid">
          ✓
        </span>
      )}
      {wordCount > MAX_ESSAY_WORDS && (
        <span
          style={{ color: "var(--error-color)", fontWeight: 600 }}
          role="status"
          aria-live="polite"
        >
          (Too long - maximum {MAX_ESSAY_WORDS} words)
        </span>
      )}
    </div>
  );
}
