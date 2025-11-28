/**
 * Shared types for assessment data
 */

import type { LanguageToolError } from "@writeo/shared";

export interface EssayScoreDimensions {
  TA?: number;
  CC?: number;
  Vocab?: number;
  Grammar?: number;
  Overall?: number;
}

export interface EssayScores {
  overall?: number;
  dimensions?: EssayScoreDimensions;
}

export interface RelevanceCheck {
  addressesQuestion: boolean;
  score: number;
  threshold?: number;
}

export interface AssessmentData {
  essayScores?: EssayScores;
  ltErrors?: LanguageToolError[];
  llmErrors?: LanguageToolError[];
  relevanceCheck?: RelevanceCheck;
}
