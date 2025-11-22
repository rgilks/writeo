/**
 * Server Actions for Writeo API - Main exports
 */

"use server";

import { makeSerializableError } from "./utils/error-handling";
import { countWords, validateWordCount } from "@writeo/shared";
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
    // Allow empty question text for free writing - send empty string to API
    const finalQuestionText = questionText?.trim() || "";
    if (!answerText?.trim()) throw new Error("Answer text is required");

    const wordCount = countWords(answerText);
    const validation = validateWordCount(wordCount);
    if (!validation.valid) {
      throw new Error(validation.error);
    }

    const { submissionId, results } = await createSubmission(
      finalQuestionText,
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
