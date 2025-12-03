/**
 * Utility functions for learner results
 */

// LanguageToolError type available from @writeo/shared if needed

const SCORE_BANDS = [
  { min: 7.5, color: "#10b981", label: "Excellent" },
  { min: 6.5, color: "#3b82f6", label: "Good" },
  { min: 5.5, color: "#f59e0b", label: "Fair" },
  { min: Number.NEGATIVE_INFINITY, color: "#ef4444", label: "Needs Improvement" },
] as const;

export function getScoreColor(score: number): string {
  return SCORE_BANDS.find((band) => score >= band.min)!.color;
}

export function getScoreLabel(score: number): string {
  return SCORE_BANDS.find((band) => score >= band.min)!.label;
}

const CEFR_DESCRIPTORS: Record<string, string> = {
  A2: "Can write simple connected text on familiar topics.",
  B1: "Can write simple connected text on topics which are familiar or of personal interest.",
  B2: "Can write clear, detailed text on a wide range of subjects.",
  C1: "Can write clear, well-structured text on complex subjects.",
  C2: "Can write clear, smoothly flowing text in an appropriate style.",
};

export function getCEFRDescriptor(level: string): string {
  return CEFR_DESCRIPTORS[level] || "Writing proficiency level.";
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

export function calculateCEFRProgress(score: number) {
  const thresholds = getCEFRThresholds();
  const levels = Object.keys(thresholds);
  const current = mapScoreToCEFR(score);
  const currentIndex = levels.indexOf(current);
  const nextIndex = Math.min(currentIndex + 1, levels.length - 1);
  const next = levels[nextIndex];

  if (current === "C2") {
    return { current, next, progress: 100, scoreToNext: 0 };
  }

  const currentThreshold = thresholds[current];
  const nextThreshold = thresholds[next];
  const range = nextThreshold.min - currentThreshold.min;
  const position = score - currentThreshold.min;
  const progress = Math.min(100, Math.max(0, (position / range) * 100));
  const scoreToNext = nextThreshold.min - score;

  return { current, next, progress, scoreToNext };
}

const CEFR_LABELS: Record<string, string> = {
  A2: "Elementary",
  B1: "Intermediate",
  B2: "Upper Intermediate",
  C1: "Advanced",
  C2: "Proficient",
};

export function getCEFRLabel(level: string): string {
  return CEFR_LABELS[level] || level;
}

const ERROR_EXPLANATIONS: Record<string, string> = {
  "Subject-verb agreement":
    "The subject and verb must agree in number (singular/plural). Example: 'He go' should be 'He goes'.",
  "Verb tense":
    "Use consistent verb tenses. Check if actions happened in the past, present, or future.",
  "Article use":
    "Use 'a' before consonant sounds, 'an' before vowel sounds, and 'the' for specific things.",
  "Article usage":
    "Use 'a' before consonant sounds, 'an' before vowel sounds, and 'the' for specific things. Articles help define nouns.",
  Preposition:
    "Prepositions show relationships (in, on, at, with, etc.). Choose the correct one for the context.",
  Spelling:
    "Check spelling carefully. Common mistakes include homophones (words that sound the same but are spelled differently).",
  Punctuation: "Use punctuation marks correctly: periods, commas, question marks, etc.",
  "Word order": "English follows a specific word order: Subject-Verb-Object.",
  "Grammar error": "A grammatical mistake that affects clarity or correctness.",
  Grammar:
    "A grammatical mistake that affects clarity or correctness. Check sentence structure and word forms.",
  "Possible Typo": "This looks like a typing error. Check the spelling.",
  Capitalization:
    "Sentences should start with a capital letter. Proper nouns (names, places) should also be capitalized.",
  Compound: "Check if these words should be written as one word, two words, or with a hyphen.",
  Redundancy: "Avoid using extra words that don't add meaning. Conciseness improves clarity.",
  Style: "Consider a different word choice to improve the flow or tone of your writing.",
};

export function getErrorExplanation(errorType: string, count: number): string {
  return (
    ERROR_EXPLANATIONS[errorType] ||
    `This type of error appears ${count} ${count === 1 ? "time" : "times"} in your essay.`
  );
}
