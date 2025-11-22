/**
 * Hook for extracting data from assessment results
 */

import type { AssessmentResults, LanguageToolError } from "@writeo/shared";
import { mapScoreToCEFR } from "./utils";

export function useDataExtraction(data: AssessmentResults) {
  const parts = data.results?.parts || [];
  const firstPart = parts[0];
  const firstAnswer = firstPart?.answers?.[0];
  const assessorResults = firstAnswer?.["assessor-results"] || [];

  const essayAssessor = assessorResults.find((a: any) => a.id === "T-AES-ESSAY");
  const overall = essayAssessor?.overall || 0;
  const rawDimensions = essayAssessor?.dimensions || {};
  const dimensions = {
    TA: rawDimensions.TA ?? 0,
    CC: rawDimensions.CC ?? 0,
    Vocab: rawDimensions.Vocab ?? 0,
    Grammar: rawDimensions.Grammar ?? 0,
    Overall: rawDimensions.Overall ?? 0,
  };

  const lowestDim = Object.entries(dimensions)
    .filter(([k]) => k !== "Overall")
    .sort(([, a], [, b]) => (a ?? 0) - (b ?? 0))[0] as [string, number] | undefined;

  const ltAssessor = assessorResults.find((a: any) => a.id === "T-GEC-LT");
  const ltErrors: LanguageToolError[] = Array.isArray(ltAssessor?.errors) ? ltAssessor.errors : [];

  const llmAssessor = assessorResults.find((a: any) => a.id === "T-GEC-LLM");
  const llmErrors: LanguageToolError[] = Array.isArray(llmAssessor?.errors)
    ? llmAssessor.errors
    : [];

  const grammarErrors: LanguageToolError[] = [...ltErrors, ...llmErrors];

  const teacherAssessor = assessorResults.find((a: any) => a.id === "T-TEACHER-FEEDBACK");
  const teacherFeedback = teacherAssessor?.meta
    ? {
        message: teacherAssessor.meta.message as string,
        focusArea: teacherAssessor.meta.focusArea as string | undefined,
        cluesMessage: (teacherAssessor.meta as any).cluesMessage as string | undefined,
        explanationMessage: (teacherAssessor.meta as any).explanationMessage as string | undefined,
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

  const answerTexts = data.meta?.answerTexts as Record<string, string> | undefined;
  const answerId = answerTexts ? Object.keys(answerTexts)[0] : undefined;

  const questionTexts = data.meta?.questionTexts as Record<string, string> | undefined;
  const questionText = questionTexts && answerId ? questionTexts[answerId] : "";

  const relevanceAssessor = assessorResults.find((a: any) => a.id === "T-RELEVANCE-CHECK");
  const relevanceCheck = relevanceAssessor?.meta
    ? {
        addressesQuestion: Boolean(relevanceAssessor.meta.addressesQuestion ?? false),
        score: Number(relevanceAssessor.meta.similarityScore ?? 0),
        threshold: Number(relevanceAssessor.meta.threshold ?? 0.5),
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
