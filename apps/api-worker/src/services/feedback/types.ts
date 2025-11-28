import type { EssayScoreDimensions, EssayScores, RelevanceCheck } from "@writeo/shared";

export type { EssayScoreDimensions, EssayScores, RelevanceCheck };

export interface FeedbackError {
  message: string;
  category: string;
  suggestions?: string[];
  start?: number;
  end?: number;
  errorType?: string;
}

export interface AIFeedback {
  relevance: {
    addressesQuestion: boolean;
    score: number;
    explanation: string;
  };
  feedback: {
    strengths: string[];
    improvements: string[];
    overall: string;
  };
}

export interface TeacherFeedback {
  message: string;
  focusArea?: string;
  clues?: string;
  explanation?: string;
}

export interface CombinedFeedback {
  detailed: AIFeedback;
  teacher: TeacherFeedback;
}
