/**
 * Feedback types
 */

export interface FeedbackError {
  message: string;
  category: string;
  suggestions?: string[];
  start?: number;
  end?: number;
  errorType?: string;
}

export interface EssayScoreDimensions {
  TA?: number;
  CC?: number;
  Vocab?: number;
  Grammar?: number;
}

export interface EssayScores {
  overall?: number;
  dimensions?: EssayScoreDimensions;
  label?: string;
}

export interface RelevanceCheck {
  addressesQuestion: boolean;
  score: number;
  threshold?: number;
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
