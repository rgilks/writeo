/**
 * Storage data retrieval utilities for feedback routes
 */

import { StorageService } from "../../services/storage";
import type { AssessmentResults, LanguageToolError } from "@writeo/shared";

export interface FeedbackData {
  questionText: string;
  essayScores?: {
    overall?: number;
    dimensions?: {
      TA?: number;
      CC?: number;
      Vocab?: number;
      Grammar?: number;
      Overall?: number;
    };
  };
  ltErrors?: LanguageToolError[];
  llmErrors?: LanguageToolError[];
  relevanceCheck?: {
    addressesQuestion: boolean;
    score: number;
    threshold: number;
  };
}

export async function loadFeedbackDataFromStorage(
  storage: StorageService,
  submissionId: string,
  answerId: string,
  providedQuestionText?: string,
  providedAssessmentData?: any,
): Promise<FeedbackData | null> {
  const results = await storage.getResults(submissionId);
  if (!results) return null;

  let questionText = providedQuestionText || "";
  if (!questionText) {
    const questionTexts = results.meta?.questionTexts as Record<string, string> | undefined;
    if (questionTexts && questionTexts[answerId]) {
      questionText = questionTexts[answerId];
    } else {
      const answer = await storage.getAnswer(answerId);
      if (answer) {
        const question = await storage.getQuestion(answer["question-id"]);
        if (question) {
          questionText = question.text;
        }
      }
    }
  }

  let essayScores = providedAssessmentData?.essayScores;
  let ltErrors = providedAssessmentData?.ltErrors;
  let llmErrors = providedAssessmentData?.llmErrors;
  let relevanceCheck = providedAssessmentData?.relevanceCheck;

  if (!essayScores || !ltErrors || !llmErrors || !relevanceCheck) {
    for (const part of results.results?.parts || []) {
      for (const answer of part.answers || []) {
        for (const assessor of answer["assessor-results"] || []) {
          if (!essayScores && assessor.id === "T-AES-ESSAY") {
            essayScores = { overall: assessor.overall, dimensions: assessor.dimensions };
          }
          if (!ltErrors && assessor.id === "T-GEC-LT") {
            ltErrors = assessor.errors as LanguageToolError[] | undefined;
          }
          if (!llmErrors && assessor.id === "T-GEC-LLM") {
            llmErrors = assessor.errors as LanguageToolError[] | undefined;
          }
          if (!relevanceCheck && assessor.id === "T-RELEVANCE-CHECK" && assessor.meta) {
            const meta = assessor.meta as any;
            relevanceCheck = {
              addressesQuestion: meta.addressesQuestion ?? false,
              score: meta.similarityScore ?? 0,
              threshold: meta.threshold ?? 0.5,
            };
          }
        }
      }
    }
  }

  return {
    questionText,
    essayScores,
    ltErrors,
    llmErrors,
    relevanceCheck,
  };
}

export function getCachedTeacherFeedback(
  results: AssessmentResults | null,
  mode: "clues" | "explanation",
): string | undefined {
  if (!results) return undefined;
  const firstPart = results.results?.parts?.[0];
  const teacherAssessor = firstPart?.answers?.[0]?.["assessor-results"]?.find(
    (a: any) => a.id === "T-TEACHER-FEEDBACK",
  );
  if (!teacherAssessor) return undefined;
  const meta = (teacherAssessor.meta || {}) as Record<string, any>;
  return mode === "clues" ? meta.cluesMessage : meta.explanationMessage;
}
