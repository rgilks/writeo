/**
 * Relevance result processing
 */

import type { RelevanceCheck } from "../relevance";
import { safeLogError } from "../../utils/logging";

type RelevanceResultSettle = PromiseSettledResult<Array<RelevanceCheck | null>>;

export function processRelevanceResults(
  relevanceResults: RelevanceResultSettle,
  relevanceRequests: Array<{ answerId: string }>,
): Map<string, RelevanceCheck> {
  const relevanceByAnswerId = new Map<string, RelevanceCheck>();

  if (relevanceResults.status !== "fulfilled" || !Array.isArray(relevanceResults.value)) {
    const errorMsg =
      relevanceResults.status === "rejected"
        ? relevanceResults.reason instanceof Error
          ? relevanceResults.reason.message
          : String(relevanceResults.reason)
        : "Invalid relevance response format";
    safeLogError("Relevance assessment failed", {
      status: relevanceResults.status,
      error: errorMsg,
      requestCount: relevanceRequests.length,
    });
    return relevanceByAnswerId;
  }

  const relevanceValues = relevanceResults.value;
  if (relevanceValues.length !== relevanceRequests.length) {
    safeLogError("Relevance assessment count mismatch", {
      responseCount: relevanceValues.length,
      requestCount: relevanceRequests.length,
    });
  }

  relevanceRequests.forEach(({ answerId }, index) => {
    const relevance = relevanceValues[index];
    if (relevance) {
      relevanceByAnswerId.set(answerId, relevance);
    }
  });

  return relevanceByAnswerId;
}
