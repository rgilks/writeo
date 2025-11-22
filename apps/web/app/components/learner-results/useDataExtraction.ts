/**
 * Hook for extracting data from assessment results
 */

import type { AssessmentResults, LanguageToolError } from "@writeo/shared";
import {
  getEssayAssessorResult,
  getLanguageToolAssessorResult,
  getLLMAssessorResult,
  getTeacherFeedbackAssessorResult,
  getRelevanceCheckAssessorResult,
} from "@writeo/shared";
import { mapScoreToCEFR } from "./utils";

export function useDataExtraction(data: AssessmentResults) {
  const parts = data.results?.parts || [];
  const firstPart = parts[0];
  const firstAnswer = firstPart?.answers?.[0];
  const assessorResults = firstAnswer?.["assessor-results"] || [];

  const essayAssessor = getEssayAssessorResult(assessorResults);
  const overall = essayAssessor?.overall || 0;
  const rawDimensions = essayAssessor?.dimensions || {};
  const dimensions = {
    TA: rawDimensions.TA ?? 0,
    CC: rawDimensions.CC ?? 0,
    Vocab: rawDimensions.Vocab ?? 0,
    Grammar: rawDimensions.Grammar ?? 0,
    Overall: rawDimensions.Overall ?? 0,
  };

  // Extract questionText early to determine if TA should be included in lowestDim
  const answerTexts = data.meta?.answerTexts as Record<string, string> | undefined;
  const answerId = answerTexts ? Object.keys(answerTexts)[0] : undefined;
  const questionTexts = data.meta?.questionTexts as Record<string, string> | undefined;
  const questionText = questionTexts && answerId ? questionTexts[answerId] : "";
  const hasQuestion = questionText && questionText.trim().length > 0;

  // Calculate lowestDim excluding TA if there's no question
  const lowestDim = Object.entries(dimensions)
    .filter(([k]) => k !== "Overall" && (hasQuestion || k !== "TA"))
    .sort(([, a], [, b]) => (a ?? 0) - (b ?? 0))[0] as [string, number] | undefined;

  const ltAssessor = getLanguageToolAssessorResult(assessorResults);
  const ltErrors: LanguageToolError[] = ltAssessor?.errors ?? [];

  const llmAssessor = getLLMAssessorResult(assessorResults);
  const llmErrors: LanguageToolError[] = llmAssessor?.errors ?? [];

  const grammarErrors: LanguageToolError[] = [...ltErrors, ...llmErrors];

  const teacherAssessor = getTeacherFeedbackAssessorResult(assessorResults);
  const teacherFeedback = teacherAssessor?.meta
    ? {
        message: teacherAssessor.meta.message,
        focusArea: teacherAssessor.meta.focusArea,
        cluesMessage: teacherAssessor.meta.cluesMessage,
        explanationMessage: teacherAssessor.meta.explanationMessage,
      }
    : undefined;

  const submissionId =
    typeof window !== "undefined"
      ? window.location.pathname.split("/results/")[1]?.split("/")[0]
      : undefined;

  const draftNumber = (data.meta?.draftNumber as number) || 1;
  const parentSubmissionId = data.meta?.parentSubmissionId as string | undefined;
  const draftHistory =
    (data.meta?.draftHistory as Array<{
      draftNumber: number;
      timestamp: string;
      wordCount: number;
      errorCount: number;
      overallScore?: number;
    }>) || [];

  const previousDraft = draftHistory.length > 1 ? draftHistory[draftHistory.length - 2] : null;
  const currentDraft = draftHistory.length > 0 ? draftHistory[draftHistory.length - 1] : null;

  const wordCountDiff =
    previousDraft && currentDraft ? currentDraft.wordCount - previousDraft.wordCount : null;
  const errorCountDiff =
    previousDraft && currentDraft ? currentDraft.errorCount - previousDraft.errorCount : null;
  const scoreDiff =
    previousDraft && currentDraft && previousDraft.overallScore && currentDraft.overallScore
      ? currentDraft.overallScore - previousDraft.overallScore
      : null;

  const relevanceAssessor = getRelevanceCheckAssessorResult(assessorResults);
  const relevanceCheck = relevanceAssessor?.meta
    ? {
        addressesQuestion: relevanceAssessor.meta.addressesQuestion,
        score: relevanceAssessor.meta.similarityScore,
        threshold: relevanceAssessor.meta.threshold,
      }
    : undefined;

  return {
    overall,
    dimensions,
    lowestDim,
    grammarErrors,
    ltErrors,
    llmErrors,
    teacherFeedback,
    submissionId,
    draftNumber,
    parentSubmissionId,
    draftHistory,
    scoreDiff,
    errorCountDiff,
    wordCountDiff,
    answerId,
    questionText,
    relevanceCheck,
  };
}
