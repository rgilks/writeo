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
  parentResults?: any,
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
      storeResults,
    );

    // Always set draft metadata - even for first draft
    if (results) {
      if (!results.meta) {
        results.meta = {};
      }

      if (parentSubmissionId) {
        // This is a follow-up draft
        try {
          const draftInfo = await getDraftInfo(parentSubmissionId, storeResults, parentResults);
          results.meta.draftNumber = draftInfo.draftNumber;
          results.meta.parentSubmissionId = draftInfo.parentSubmissionId;
          results.meta.draftHistory = draftInfo.draftHistory;
        } catch (error) {
          console.warn("[submitEssay] Failed to get draft info:", error);
          // Set fallback values so draft tracking still works
          results.meta.draftNumber = 2;
          results.meta.parentSubmissionId = parentSubmissionId;
          results.meta.draftHistory = [];
        }
      } else {
        // This is the first draft - set draftNumber: 1 explicitly
        results.meta.draftNumber = 1;
        results.meta.draftHistory = [];
      }

      // Store word count in meta for draft history tracking
      results.meta.wordCount = wordCount;
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
