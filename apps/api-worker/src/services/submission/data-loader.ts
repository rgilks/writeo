/**
 * Data loading utilities for building Modal requests
 */

import type { CreateSubmissionRequest, ModalRequest } from "@writeo/shared";
import { StorageService } from "../storage";
import { errorResponse } from "../../utils/errors";
import type { Context } from "hono";
import type { Env } from "../../types/env";

export async function buildModalRequest(
  body: CreateSubmissionRequest,
  storeResults: boolean,
  storage: StorageService,
  c: Context<{ Bindings: Env }>,
): Promise<ModalRequest | Response> {
  const modalParts: ModalRequest["parts"] = [];

  for (const part of body.submission) {
    const modalAnswers: ModalRequest["parts"][0]["answers"] = [];
    for (const answerRef of part.answers) {
      let questionText: string | undefined = answerRef["question-text"];
      if (!questionText && storeResults) {
        const question = await storage.getQuestion(answerRef["question-id"] || "");
        questionText = question?.text;
      }
      // Allow empty string for free writing (no question)
      // If questionText is undefined or null, use empty string
      if (questionText === undefined || questionText === null) {
        questionText = "";
      }

      const answerText = answerRef.text;
      if (!answerText) {
        return errorResponse(400, `Answer text is required for answer ${answerRef.id}`, c);
      }

      modalAnswers.push({
        id: answerRef.id,
        question_id: answerRef["question-id"] || "",
        question_text: questionText,
        answer_text: answerText,
      });
    }
    modalParts.push({ part: part.part, answers: modalAnswers });
  }

  return {
    submission_id: c.req.param("submission_id"),
    template: body.template,
    parts: modalParts,
  };
}
