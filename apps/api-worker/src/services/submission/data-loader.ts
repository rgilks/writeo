/**
 * Data loading utilities for building Modal requests
 */

import type { CreateSubmissionRequest, ModalRequest } from "@writeo/shared";
import { StorageService } from "../storage";
import { errorResponse } from "../../utils/errors";
import type { Context } from "hono";
import type { Env } from "../../types/env";

type SubmissionAnswer = CreateSubmissionRequest["submission"][number]["answers"][number];
type ModalAnswer = ModalRequest["parts"][number]["answers"][number];

async function resolveQuestionText(
  answer: SubmissionAnswer,
  storeResults: boolean,
  storage: StorageService,
  cache: Map<string, string>,
): Promise<string> {
  const providedText = answer["question-text"];
  if (providedText !== undefined && providedText !== null) {
    return providedText;
  }

  if (!storeResults) {
    return "";
  }

  const questionId = answer["question-id"] ?? "";
  if (!questionId) {
    return "";
  }

  if (cache.has(questionId)) {
    return cache.get(questionId)!;
  }

  const question = await storage.getQuestion(questionId);
  const questionText = question?.text ?? "";
  cache.set(questionId, questionText);
  return questionText;
}

export async function buildModalRequest(
  body: CreateSubmissionRequest,
  storeResults: boolean,
  storage: StorageService,
  c: Context<{ Bindings: Env }>,
): Promise<ModalRequest | Response> {
  const modalParts: ModalRequest["parts"] = [];
  const questionTextCache = new Map<string, string>();

  for (const part of body.submission) {
    const modalAnswers: ModalAnswer[] = [];

    for (const answerRef of part.answers) {
      const questionId = answerRef["question-id"] ?? "";
      const questionText = await resolveQuestionText(
        answerRef,
        storeResults,
        storage,
        questionTextCache,
      );

      const answerText = answerRef.text;
      if (!answerText) {
        return errorResponse(400, `Answer text is required for answer ${answerRef.id}`, c);
      }

      modalAnswers.push({
        id: answerRef.id,
        question_id: questionId,
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
