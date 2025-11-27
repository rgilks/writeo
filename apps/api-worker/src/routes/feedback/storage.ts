/**
 * Storage data retrieval utilities for feedback routes
 */

import { StorageService } from "../../services/storage";
import type { AssessmentResults, LanguageToolError, AssessorResult } from "@writeo/shared";
import {
  getEssayAssessorResult,
  getLanguageToolAssessorResult,
  getLLMAssessorResult,
  getRelevanceCheckAssessorResult,
  findAssessorResultById,
} from "@writeo/shared";
import { AssessmentDataSchema, type AssessmentDataInput } from "./validation";

export type FeedbackData = { questionText: string } & AssessmentDataInput;

/**
 * Loads feedback data (question text, assessment data) from storage.
 * Falls back to provided data when available.
 */
export async function loadFeedbackDataFromStorage(
  storage: StorageService,
  submissionId: string,
  answerId: string,
  providedQuestionText?: string,
  providedAssessmentData?: AssessmentDataInput,
): Promise<FeedbackData | null> {
  const results = await storage.getResults(submissionId);
  if (!results) return null;

  const questionText = await resolveQuestionText(storage, results, answerId, providedQuestionText);

  const safeAssessmentData = providedAssessmentData
    ? AssessmentDataSchema.parse(providedAssessmentData)
    : undefined;

  const assessmentData = mergeAssessmentData(results, safeAssessmentData);

  return {
    questionText,
    ...assessmentData,
  };
}

/**
 * Attempts to resolve question text from provided data, metadata, or StorageService lookup.
 */
async function resolveQuestionText(
  storage: StorageService,
  results: AssessmentResults,
  answerId: string,
  providedQuestionText?: string,
): Promise<string> {
  if (providedQuestionText && providedQuestionText.trim().length > 0) {
    return providedQuestionText;
  }

  const questionTexts = results.meta?.questionTexts as Record<string, string> | undefined;
  if (questionTexts?.[answerId]) {
    return questionTexts[answerId];
  }

  const answer = await storage.getAnswer(answerId);
  if (answer) {
    const question = await storage.getQuestion(answer["question-id"]);
    if (question) {
      return question.text;
    }
  }

  return "";
}

/**
 * Merges provided assessment data with data extracted from assessment results.
 */
function mergeAssessmentData(
  results: AssessmentResults,
  providedAssessmentData?: AssessmentDataInput,
): AssessmentDataInput {
  let essayScores = providedAssessmentData?.essayScores;
  let ltErrors = providedAssessmentData?.ltErrors;
  let llmErrors = providedAssessmentData?.llmErrors;
  let relevanceCheck = providedAssessmentData?.relevanceCheck;

  if (essayScores && ltErrors && llmErrors && relevanceCheck) {
    return { essayScores, ltErrors, llmErrors, relevanceCheck };
  }

  for (const part of results.results?.parts || []) {
    for (const answer of part.answers || []) {
      const assessorResults = (answer["assessor-results"] || []) as AssessorResult[];

      if (!essayScores) {
        const essayAssessor = getEssayAssessorResult(assessorResults);
        if (essayAssessor) {
          essayScores = {
            overall: essayAssessor.overall,
            dimensions: essayAssessor.dimensions,
          };
        }
      }

      if (!ltErrors) {
        const ltAssessor = getLanguageToolAssessorResult(assessorResults);
        if (ltAssessor) {
          ltErrors = ltAssessor.errors;
        }
      }

      if (!llmErrors) {
        const llmAssessor = getLLMAssessorResult(assessorResults);
        if (llmAssessor) {
          llmErrors = llmAssessor.errors;
        }
      }

      if (!relevanceCheck) {
        const relevanceAssessor = getRelevanceCheckAssessorResult(assessorResults);
        if (relevanceAssessor) {
          const meta = relevanceAssessor.meta;
          relevanceCheck = {
            addressesQuestion: meta.addressesQuestion,
            score: meta.similarityScore,
            threshold: meta.threshold,
          };
        }
      }

      if (essayScores && ltErrors && llmErrors && relevanceCheck) {
        break;
      }
    }

    if (essayScores && ltErrors && llmErrors && relevanceCheck) {
      break;
    }
  }

  return { essayScores, ltErrors, llmErrors, relevanceCheck };
}

export function getCachedTeacherFeedback(
  results: AssessmentResults | null,
  mode: "clues" | "explanation",
): string | undefined {
  if (!results) return undefined;
  const firstPart = results.results?.parts?.[0];
  const teacherAssessor = findAssessorResultById(
    firstPart?.answers?.[0]?.["assessor-results"] || [],
    "T-TEACHER-FEEDBACK",
  );
  if (!teacherAssessor) return undefined;
  const meta = (teacherAssessor.meta || {}) as Record<string, any>;
  return mode === "clues" ? meta.cluesMessage : meta.explanationMessage;
}
