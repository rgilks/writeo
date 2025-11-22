/**
 * LanguageTool result processing
 */

import type { LanguageToolError, LanguageToolResponse } from "@writeo/shared";
import type { ModalRequest } from "@writeo/shared";
import { safeLogError } from "../../utils/logging";
import { transformLanguageToolResponse } from "../../utils/text-processing";

export async function processLanguageToolResults(
  ltResults: PromiseSettledResult<Response[]>,
  ltRequests: Array<{ answerId: string; request: Promise<Response> }>,
  modalParts: ModalRequest["parts"],
  env: { MODAL_LT_URL?: string }
): Promise<{
  ltErrorsByAnswerId: Map<string, LanguageToolError[]>;
  answerTextsByAnswerId: Map<string, string>;
}> {
  const ltErrorsByAnswerId = new Map<string, LanguageToolError[]>();
  const answerTextsByAnswerId = new Map<string, string>();

  if (env.MODAL_LT_URL) {
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
              `LanguageTool request failed: ${res.status} ${res.statusText} - ${errorText}`
            );
          }
          return await res.json();
        })
      );

      for (let i = 0; i < ltResponses.length; i++) {
        const result = ltResponses[i];
        if (!result) continue;
        if (result.status === "fulfilled") {
          const answerId = ltRequests[i]?.answerId;
          if (answerId) {
            for (const part of modalParts) {
              const answer = part.answers.find((a) => a.id === answerId);
              if (answer) {
                answerTextsByAnswerId.set(answerId, answer.answer_text);
                break;
              }
            }
            let fullText = "";
            for (const part of modalParts) {
              const answer = part.answers.find((a) => a.id === answerId);
              if (answer) {
                fullText = answer.answer_text;
                break;
              }
            }
            const errors = transformLanguageToolResponse(
              (result as PromiseFulfilledResult<LanguageToolResponse>).value,
              fullText
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
    for (const part of modalParts) {
      for (const answer of part.answers) {
        answerTextsByAnswerId.set(answer.id, answer.answer_text);
      }
    }
  }

  return { ltErrorsByAnswerId, answerTextsByAnswerId };
}
