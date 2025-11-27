/**
 * LLM result processing
 */

import type { LanguageToolError } from "@writeo/shared";
import { safeLogError } from "../../utils/logging";

const DEBUG_LLM_ASSESSMENT =
  typeof process !== "undefined" && process.env?.DEBUG_LLM_ASSESSMENT === "true";

type LLMResultSettle = PromiseSettledResult<LanguageToolError[][]>;

function logLLMProcessingSummary(
  llmResponses: LanguageToolError[][],
  llmAssessmentRequests: Array<{ answerId: string }>,
  llmProvider: string,
  aiModel: string,
) {
  const mismatchedCounts =
    llmResponses.length !== llmAssessmentRequests.length
      ? ` (mismatch: ${llmResponses.length} responses vs ${llmAssessmentRequests.length} requests)`
      : "";
  console.log(
    `[LLM Assessment] Processing ${llmResponses.length} responses${mismatchedCounts} for ${llmAssessmentRequests.length} requests (provider: ${llmProvider}, model: ${aiModel})`,
  );
}

function logPerAnswerDetails(
  answerId: string,
  llmErrors: LanguageToolError[],
  llmProvider: string,
  aiModel: string,
) {
  if (!DEBUG_LLM_ASSESSMENT) {
    return;
  }

  const message =
    llmErrors.length === 0
      ? `[LLM Assessment][debug] No errors found for answer ${answerId}`
      : `[LLM Assessment][debug] Found ${llmErrors.length} errors for answer ${answerId}`;
  console.log(message, { provider: llmProvider, model: aiModel });
}

function logLLMFailure(
  llmResults: LLMResultSettle,
  errorMsg: string,
  llmProvider: string,
  aiModel: string,
  requestCount: number,
) {
  console.error(`[LLM Assessment] Failed`, {
    status: llmResults.status,
    error: errorMsg,
    provider: llmProvider,
    model: aiModel,
    hasRequests: requestCount > 0,
  });
  safeLogError("LLM assessment failed", {
    status: llmResults.status,
    error: errorMsg,
    provider: llmProvider,
    model: aiModel,
  });
}

export function processLLMResults(
  llmResults: LLMResultSettle,
  llmAssessmentRequests: Array<{ answerId: string }>,
  llmProvider: string,
  aiModel: string,
): Map<string, LanguageToolError[]> {
  const llmErrorsByAnswerId = new Map<string, LanguageToolError[]>();

  if (llmResults.status !== "fulfilled" || !Array.isArray(llmResults.value)) {
    const errorMsg =
      llmResults.status === "rejected"
        ? llmResults.reason instanceof Error
          ? llmResults.reason.message
          : String(llmResults.reason)
        : "Invalid response format";
    logLLMFailure(llmResults, errorMsg, llmProvider, aiModel, llmAssessmentRequests.length);
    return llmErrorsByAnswerId;
  }

  const llmResponses = llmResults.value;
  logLLMProcessingSummary(llmResponses, llmAssessmentRequests, llmProvider, aiModel);

  llmAssessmentRequests.forEach(({ answerId }, index) => {
    const llmErrors = llmResponses[index] ?? [];
    llmErrorsByAnswerId.set(answerId, llmErrors);
    logPerAnswerDetails(answerId, llmErrors, llmProvider, aiModel);
  });

  return llmErrorsByAnswerId;
}
