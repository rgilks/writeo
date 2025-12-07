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
  getGECSeq2seqAssessorResult,
  getGECGectorAssessorResult,
  convertGECEditsToErrors,
} from "@writeo/shared";

export function useDataExtraction(data: AssessmentResults, submissionId?: string) {
  const parts = data.results?.parts ?? [];
  const [firstPart] = parts;
  const [firstAnswer] = firstPart?.answers ?? [];
  const assessorResults = firstAnswer?.assessorResults ?? [];

  const essayAssessor = getEssayAssessorResult(assessorResults);
  const overall = essayAssessor?.overall ?? 0;
  const rawDimensions = essayAssessor?.dimensions ?? {};
  const dimensions = {
    TA: rawDimensions.TA ?? 0,
    CC: rawDimensions.CC ?? 0,
    Vocab: rawDimensions.Vocab ?? 0,
    Grammar: rawDimensions.Grammar ?? 0,
    Overall: rawDimensions.Overall ?? 0,
  };

  // Extract questionText early to determine if TA should be included in lowestDim
  const meta = data.meta ?? {};
  const answerTexts = meta.answerTexts as Record<string, string> | undefined;
  const [answerId] = answerTexts ? Object.keys(answerTexts) : [];
  const questionTexts = meta.questionTexts as Record<string, string> | undefined;
  const questionText = (answerId && questionTexts?.[answerId]) || "";
  const hasQuestion = questionText.trim().length > 0;

  // Calculate lowestDim excluding TA if there's no question
  const lowestDim = Object.entries(dimensions)
    .filter(([k]) => k !== "Overall" && (hasQuestion || k !== "TA"))
    .sort(([, a], [, b]) => (a ?? 0) - (b ?? 0))[0] as [string, number] | undefined;

  // Collect errors from all GEC sources
  const ltErrors: LanguageToolError[] =
    getLanguageToolAssessorResult(assessorResults)?.errors ?? [];
  const llmErrors: LanguageToolError[] = getLLMAssessorResult(assessorResults)?.errors ?? [];

  // Convert GEC Seq2Seq edits to LanguageToolError format for heatmap display
  const gecSeq2seqAssessor = getGECSeq2seqAssessorResult(assessorResults);
  const gecSeq2seqErrors: LanguageToolError[] = gecSeq2seqAssessor
    ? convertGECEditsToErrors(gecSeq2seqAssessor.meta.edits, "GEC-SEQ2SEQ")
    : [];

  // Convert GEC GECToR edits to LanguageToolError format for heatmap display
  const gecGectorAssessor = getGECGectorAssessorResult(assessorResults);
  const gecGectorErrors: LanguageToolError[] = gecGectorAssessor
    ? convertGECEditsToErrors(gecGectorAssessor.meta.edits, "GEC-GECTOR")
    : [];

  // Combine all grammar errors: LT + LLM + GEC Seq2Seq + GEC GECToR
  const grammarErrors: LanguageToolError[] = [
    ...ltErrors,
    ...llmErrors,
    ...gecSeq2seqErrors,
    ...gecGectorErrors,
  ];

  const teacherAssessor = getTeacherFeedbackAssessorResult(assessorResults);
  const teacherFeedback = teacherAssessor?.meta
    ? {
        message: teacherAssessor.meta.message,
        focusArea: teacherAssessor.meta.focusArea,
        cluesMessage: teacherAssessor.meta.cluesMessage,
        explanationMessage: teacherAssessor.meta.explanationMessage,
      }
    : undefined;

  // submissionId is now passed as a parameter instead of reading from window.location

  const draftNumber = (meta.draftNumber as number) ?? 1;
  const parentSubmissionId = meta.parentSubmissionId as string | undefined;
  const draftHistory =
    (meta.draftHistory as Array<{
      draftNumber: number;
      timestamp: string;
      wordCount: number;
      errorCount: number;
      overallScore?: number;
    }>) ?? [];

  const previousDraft = draftHistory.length > 1 ? draftHistory[draftHistory.length - 2] : null;
  const currentDraft = draftHistory.length > 0 ? draftHistory[draftHistory.length - 1] : null;

  const hasNumber = (value: unknown): value is number =>
    typeof value === "number" && !Number.isNaN(value);

  const wordCountDiff =
    previousDraft &&
    currentDraft &&
    hasNumber(previousDraft.wordCount) &&
    hasNumber(currentDraft.wordCount)
      ? currentDraft.wordCount - previousDraft.wordCount
      : null;
  const errorCountDiff =
    previousDraft &&
    currentDraft &&
    hasNumber(previousDraft.errorCount) &&
    hasNumber(currentDraft.errorCount)
      ? currentDraft.errorCount - previousDraft.errorCount
      : null;
  const scoreDiff =
    previousDraft &&
    currentDraft &&
    hasNumber(previousDraft.overallScore) &&
    hasNumber(currentDraft.overallScore)
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
