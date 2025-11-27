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
            const errors = transformLanguageToolResponse(
              (result as PromiseFulfilledResult<LanguageToolResponse>).value,
              answer.answer_text,
            );
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
