"use server";

import type { AssessmentResults } from "@writeo/shared";
import { countWords, validateWordCount } from "@writeo/shared";
import { makeSerializableError } from "./utils/error-handling";
import {
  createSubmission,
  getSubmissionResults,
  pollSubmissionResults,
} from "./actions/submission";
import { getDraftInfo, getSubmissionResultsWithDraftTracking } from "./actions/draft";
import { getTeacherFeedback } from "./actions/teacher-feedback";
import { streamAIFeedback } from "./actions/streaming";

interface SubmitEssayResult {
  submissionId: string;
  results: unknown;
}

interface ResultsWithMeta {
  meta?: Record<string, unknown>;
}

function hasMeta(results: unknown): results is ResultsWithMeta {
  return typeof results === "object" && results !== null;
}

function validateAnswer(answerText: string): number {
  if (!answerText?.trim()) {
    throw new Error("Answer text is required");
  }

  const wordCount = countWords(answerText);
  const validation = validateWordCount(wordCount);
  if (!validation.valid) {
    throw new Error(validation.error);
  }

  return wordCount;
}

function ensureMeta(results: ResultsWithMeta): void {
  if (!results.meta) {
    results.meta = {};
  }
}

async function setFollowUpDraftMetadata(
  results: ResultsWithMeta,
  parentSubmissionId: string,
  storeResults: boolean,
  parentResults: AssessmentResults | undefined,
  wordCount: number,
): Promise<void> {
  try {
    const draftInfo = await getDraftInfo(parentSubmissionId, storeResults, parentResults);
    results.meta = {
      ...results.meta,
      draftNumber: draftInfo.draftNumber,
      parentSubmissionId: draftInfo.parentSubmissionId,
      draftHistory: draftInfo.draftHistory,
      wordCount,
    };
  } catch (error) {
    console.warn("[submitEssay] Failed to get draft info:", error);
    // Fallback to default follow-up draft metadata
    results.meta = {
      ...results.meta,
      draftNumber: 2,
      parentSubmissionId,
      draftHistory: [],
      wordCount,
    };
  }
}

function setFirstDraftMetadata(results: ResultsWithMeta, wordCount: number): void {
  results.meta = {
    ...results.meta,
    draftNumber: 1,
    draftHistory: [],
    wordCount,
  };
}

export async function submitEssay(
  questionText: string,
  answerText: string,
  parentSubmissionId?: string,
  storeResults: boolean = false,
  parentResults?: AssessmentResults,
): Promise<SubmitEssayResult> {
  try {
    const finalQuestionText = questionText?.trim() || "";
    const wordCount = validateAnswer(answerText);

    const { submissionId, results } = await createSubmission(
      finalQuestionText,
      answerText,
      storeResults,
    );

    if (hasMeta(results)) {
      ensureMeta(results);

      if (parentSubmissionId) {
        await setFollowUpDraftMetadata(
          results,
          parentSubmissionId,
          storeResults,
          parentResults,
          wordCount,
        );
      } else {
        setFirstDraftMetadata(results, wordCount);
      }
    }

    return { submissionId, results };
  } catch (error) {
    console.error("[submitEssay] Error:", error instanceof Error ? error.message : String(error));
    throw makeSerializableError(error);
  }
}

export { getSubmissionResults, pollSubmissionResults } from "./actions/submission";
export { getSubmissionResultsWithDraftTracking } from "./actions/draft";
export { getTeacherFeedback } from "./actions/teacher-feedback";
export { streamAIFeedback } from "./actions/streaming";
