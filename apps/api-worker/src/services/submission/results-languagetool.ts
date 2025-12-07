/**
 * LanguageTool result processing
 */

import type { LanguageToolError, LanguageToolResponse } from "@writeo/shared";
import type { ModalRequest } from "@writeo/shared";
import { safeLogError } from "../../utils/logging";
import { transformLanguageToolResponse } from "../../utils/text-processing";
import { buildAnswerLookup, iterateAnswers } from "./utils";

export async function processLanguageToolResults(
  ltResults: PromiseSettledResult<Response[]>,
  ltRequests: Array<{ answerId: string; request: Promise<Response> }>,
  modalParts: ModalRequest["parts"],
  languageToolEnabled: boolean,
): Promise<{
  ltErrorsByAnswerId: Map<string, LanguageToolError[]>;
  answerTextsByAnswerId: Map<string, string>;
}> {
  const ltErrorsByAnswerId = new Map<string, LanguageToolError[]>();
  const answerTextsByAnswerId = new Map<string, string>();
  const answersById = buildAnswerLookup(modalParts);

  if (languageToolEnabled) {
    if (ltResults.status !== "fulfilled" || !Array.isArray(ltResults.value)) {
      const errorMsg =
        ltResults.status === "rejected"
          ? ltResults.reason instanceof Error
            ? ltResults.reason.message
            : String(ltResults.reason)
          : "Invalid response";
      safeLogError("LanguageTool service failed", { error: errorMsg });
    } else {
      const ltResponses = await Promise.allSettled(
        ltResults.value.map(async (res) => {
          if (!res.ok) {
            const errorText = await res.text().catch(() => res.statusText);
            throw new Error(
              `LanguageTool request failed: ${res.status} ${res.statusText} - ${errorText}`,
            );
          }
          return await res.json();
        }),
      );

      for (let i = 0; i < ltResponses.length; i++) {
        const result = ltResponses[i];
        if (!result) continue;
        if (result.status === "fulfilled") {
          const answerId = ltRequests[i]?.answerId;
          if (answerId) {
            const answer = answersById.get(answerId);
            if (!answer) {
              safeLogError("LanguageTool response missing matching answer", { answerId });
              continue;
            }
            answerTextsByAnswerId.set(answerId, answer.answer_text);
            const ltResponse = (result as PromiseFulfilledResult<LanguageToolResponse>).value;

            // Debug logging in CI to diagnose test failures
            if (process.env.CI === "true") {
              console.log("[processLanguageToolResults] Processing LT response", {
                answerId,
                textLength: answer.answer_text.length,
                matchCount: ltResponse.matches?.length || 0,
                firstMatch: ltResponse.matches?.[0]
                  ? {
                      offset: ltResponse.matches[0].offset,
                      length: ltResponse.matches[0].length,
                      ruleId: ltResponse.matches[0].rule?.id,
                      ruleType: ltResponse.matches[0].rule?.type,
                      hasContext: !!ltResponse.matches[0].context,
                    }
                  : null,
              });
            }

            const errors = transformLanguageToolResponse(ltResponse, answer.answer_text);

            // Debug logging in CI
            if (process.env.CI === "true" && errors.length > 0) {
              const firstError = errors[0];
              if (firstError) {
                console.log("[processLanguageToolResults] Transformed errors", {
                  answerId,
                  errorCount: errors.length,
                  firstError: {
                    start: firstError.start,
                    end: firstError.end,
                    hasConfidenceScore: "confidenceScore" in firstError,
                    confidenceScore: firstError.confidenceScore,
                    source: firstError.source,
                  },
                });
              }
            }

            ltErrorsByAnswerId.set(answerId, errors);
          }
        } else if (result.status === "rejected") {
          safeLogError(`LanguageTool request ${i} failed`, {
            reason: (result as PromiseRejectedResult).reason,
          });
        }
      }
    }
  } else {
    for (const answer of iterateAnswers(modalParts)) {
      answerTextsByAnswerId.set(answer.id, answer.answer_text);
    }
  }

  return { ltErrorsByAnswerId, answerTextsByAnswerId };
}
