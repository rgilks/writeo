/**
 * Relevance result processing
 */

import type { RelevanceCheck } from "../relevance";

export function processRelevanceResults(
  relevanceResults: PromiseSettledResult<(RelevanceCheck | null)[]>,
  relevanceRequests: Array<{ answerId: string }>,
): Map<string, RelevanceCheck> {
  const relevanceByAnswerId = new Map<string, RelevanceCheck>();
  if (relevanceResults.status === "fulfilled" && Array.isArray(relevanceResults.value)) {
    for (let i = 0; i < relevanceRequests.length; i++) {
      const request = relevanceRequests[i];
      if (!request) continue;
      const { answerId } = request;
      const relevance = relevanceResults.value[i];
      if (relevance) {
        relevanceByAnswerId.set(answerId, relevance);
      }
    }
  }
  return relevanceByAnswerId;
}
