/**
 * Server Actions for Writeo API - Main exports
 */

"use server";

import { makeSerializableError } from "./utils/error-handling";
import {
  createSubmission,
  getSubmissionResults,
  pollSubmissionResults,
} from "./actions/submission";
import { getDraftInfo, getSubmissionResultsWithDraftTracking } from "./actions/draft";
import { getTeacherFeedback } from "./actions/teacher-feedback";
import { streamAIFeedback } from "./actions/streaming";

export async function submitEssay(
  questionText: string,
  answerText: string,
  parentSubmissionId?: string,
  storeResults: boolean = false,
  parentResults?: any
): Promise<{ submissionId: string; results: any }> {
  try {
    if (!questionText?.trim()) throw new Error("Question text is required");
    if (!answerText?.trim()) throw new Error("Answer text is required");

    const wordCount = answerText
      .trim()
      .split(/\s+/)
      .filter((w) => w.length > 0).length;
    const MIN_WORDS = 250;
    const MAX_WORDS = 500;
    if (wordCount < MIN_WORDS) {
      throw new Error(
        `Essay is too short. Please write at least ${MIN_WORDS} words (currently ${wordCount} words).`
      );
    }
    if (wordCount > MAX_WORDS) {
      throw new Error(
        `Essay is too long. Please keep it under ${MAX_WORDS} words (currently ${wordCount} words).`
      );
    }

    const { submissionId, results } = await createSubmission(
      questionText,
      answerText,
      storeResults
    );

    if (parentSubmissionId && results) {
      try {
        const draftInfo = await getDraftInfo(parentSubmissionId, storeResults, parentResults);
        results.meta = {
          ...results.meta,
          draftNumber: draftInfo.draftNumber,
          parentSubmissionId: draftInfo.parentSubmissionId,
          draftHistory: draftInfo.draftHistory,
        };
      } catch (error) {
        console.warn("[submitEssay] Failed to get draft info:", error);
      }
    }

    return { submissionId, results };
  } catch (error) {
    console.error("[submitEssay] Error:", error instanceof Error ? error.message : String(error));
    throw makeSerializableError(error);
  }
}

export { getSubmissionResults, pollSubmissionResults };
export { getSubmissionResultsWithDraftTracking };
export { getTeacherFeedback };
export { streamAIFeedback };
