/**
 * LLM result processing
 */

import type { LanguageToolError } from "@writeo/shared";
import { safeLogError } from "../../utils/logging";

export function processLLMResults(
  llmResults: PromiseSettledResult<any[]>,
  llmAssessmentRequests: Array<{ answerId: string }>,
  llmProvider: string,
  aiModel: string,
): Map<string, LanguageToolError[]> {
  const llmErrorsByAnswerId = new Map<string, LanguageToolError[]>();
  if (llmResults.status === "fulfilled" && Array.isArray(llmResults.value)) {
    const llmResponses = llmResults.value;
    console.log(
      `[LLM Assessment] Processing ${llmResponses.length} responses for ${llmAssessmentRequests.length} requests`,
    );
    for (let i = 0; i < llmAssessmentRequests.length; i++) {
      const request = llmAssessmentRequests[i];
      if (!request) continue;
      const { answerId } = request;
      const llmErrors = llmResponses[i] || [];
      llmErrorsByAnswerId.set(answerId, llmErrors);
      if (llmErrors.length === 0) {
        console.log(
          `[LLM Assessment] No errors found for answer ${answerId} (provider: ${llmProvider}, model: ${aiModel})`,
        );
      } else {
        console.log(
          `[LLM Assessment] Found ${llmErrors.length} errors for answer ${answerId} (provider: ${llmProvider}, model: ${aiModel})`,
        );
      }
    }
  } else {
    const errorMsg =
      llmResults.status === "rejected"
        ? llmResults.reason instanceof Error
          ? llmResults.reason.message
          : String(llmResults.reason)
        : "Invalid response format";
    console.error(`[LLM Assessment] Failed:`, {
      status: llmResults.status,
      error: errorMsg,
      provider: llmProvider,
      model: aiModel,
      hasRequests: llmAssessmentRequests.length > 0,
    });
    safeLogError("LLM assessment failed", {
      status: llmResults.status,
      error: errorMsg,
      provider: llmProvider,
      model: aiModel,
    });
  }
  return llmErrorsByAnswerId;
}
