import type { LanguageToolError, EssayScores, RelevanceCheck } from "@writeo/shared";

export interface AssessmentData {
  essayScores?: EssayScores;
  ltErrors?: LanguageToolError[];
  llmErrors?: LanguageToolError[];
  relevanceCheck?: RelevanceCheck;
}
