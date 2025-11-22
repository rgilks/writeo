/**
 * Utility functions for learner results
 */

import type { LanguageToolError } from "@writeo/shared";

export function getScoreColor(score: number): string {
  if (score >= 7.5) return "#10b981";
  if (score >= 6.5) return "#3b82f6";
  if (score >= 5.5) return "#f59e0b";
  return "#ef4444";
}

export function getScoreLabel(score: number): string {
  if (score >= 7.5) return "Excellent";
  if (score >= 6.5) return "Good";
  if (score >= 5.5) return "Fair";
  return "Needs Improvement";
}

export function getCEFRDescriptor(level: string): string {
  const descriptors: Record<string, string> = {
    A2: "Can write simple connected text on familiar topics.",
    B1: "Can write simple connected text on topics which are familiar or of personal interest.",
    B2: "Can write clear, detailed text on a wide range of subjects.",
    C1: "Can write clear, well-structured text on complex subjects.",
    C2: "Can write clear, smoothly flowing text in an appropriate style.",
  };
  return descriptors[level] || "Writing proficiency level.";
}

export function mapScoreToCEFR(score: number): string {
  if (score >= 8.5) return "C2";
  if (score >= 7.0) return "C1";
  if (score >= 5.5) return "B2";
  if (score >= 4.0) return "B1";
  return "A2";
}

export function getCEFRThresholds(): Record<string, { min: number; max: number }> {
  return {
    A2: { min: 0, max: 4.0 },
    B1: { min: 4.0, max: 5.5 },
    B2: { min: 5.5, max: 7.0 },
    C1: { min: 7.0, max: 8.5 },
    C2: { min: 8.5, max: 9.0 },
  };
}

export function calculateCEFRProgress(score: number): {
  current: string;
  next: string;
  progress: number;
  scoreToNext: number;
} {
  const thresholds = getCEFRThresholds();
  const current = mapScoreToCEFR(score);
  const cefrLevels = ["A2", "B1", "B2", "C1", "C2"];
  const currentIndex = cefrLevels.indexOf(current);
  const nextIndex = currentIndex < cefrLevels.length - 1 ? currentIndex + 1 : currentIndex;
  const next = cefrLevels[nextIndex];

  if (current === "C2") {
    return { current, next: current, progress: 100, scoreToNext: 0 };
  }

  const currentThreshold = thresholds[current];
  const nextThreshold = thresholds[next];
  const range = nextThreshold.min - currentThreshold.min;
  const position = score - currentThreshold.min;
  const progress = Math.min(100, Math.max(0, (position / range) * 100));
  const scoreToNext = nextThreshold.min - score;

  return { current, next, progress, scoreToNext };
}

export function getCEFRLabel(level: string): string {
  const labels: Record<string, string> = {
    A2: "Elementary",
    B1: "Intermediate",
    B2: "Upper Intermediate",
    C1: "Advanced",
    C2: "Proficient",
  };
  return labels[level] || level;
}

export function getErrorExplanation(errorType: string, count: number): string {
  const explanations: Record<string, string> = {
    "Subject-verb agreement":
      "The subject and verb must agree in number (singular/plural). Example: 'He go' should be 'He goes'.",
    "Verb tense":
      "Use consistent verb tenses. Check if actions happened in the past, present, or future.",
    "Article use":
      "Use 'a' before consonant sounds, 'an' before vowel sounds, and 'the' for specific things.",
    Preposition:
      "Prepositions show relationships (in, on, at, with, etc.). Choose the correct one for the context.",
    Spelling:
      "Check spelling carefully. Common mistakes include homophones (words that sound the same but are spelled differently).",
    Punctuation: "Use punctuation marks correctly: periods, commas, question marks, etc.",
    "Word order": "English follows a specific word order: Subject-Verb-Object.",
    "Grammar error": "A grammatical mistake that affects clarity or correctness.",
  };
  return (
    explanations[errorType] ||
    `This type of error appears ${count} ${count === 1 ? "time" : "times"} in your essay.`
  );
}
