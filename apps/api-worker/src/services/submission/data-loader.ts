/**
 * Data loading utilities for building Modal requests
 */

import type { CreateSubmissionRequest, ModalRequest } from "@writeo/shared";
import { StorageService } from "../storage";
import { errorResponse, ERROR_CODES } from "../../utils/errors";
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
  const providedText = answer.questionText;
  // If questionText is explicitly null, it's free writing (no question)
  if (providedText === null) {
    return "";
  }
  // If questionText is provided as a string, use it
  if (typeof providedText === "string") {
    return providedText;
  }

  // If questionText is omitted, try to load from storage (only if storeResults is true)
  if (!storeResults) {
    return "";
  }

  const questionId = answer.questionId;
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
  submissionId: string,
  c: Context<{ Bindings: Env; Variables: { requestId?: string } }>,
): Promise<ModalRequest | Response> {
  const modalParts: ModalRequest["parts"] = [];
  const questionTextCache = new Map<string, string>();

  for (const part of body.submission) {
    const modalAnswers: ModalAnswer[] = [];

    for (const answerRef of part.answers) {
      const questionId = answerRef.questionId;
      if (!questionId) {
        return errorResponse(
          400,
          `questionId is required for answer ${answerRef.id}`,
          c,
          ERROR_CODES.MISSING_REQUIRED_FIELD,
          `submission[${part.part}].answers[${answerRef.id}].questionId`,
        );
      }

      const questionText = await resolveQuestionText(
        answerRef,
        storeResults,
        storage,
        questionTextCache,
      );

      const answerText = answerRef.text;
      if (!answerText) {
        return errorResponse(
          400,
          `Answer text is required for answer ${answerRef.id}`,
          c,
          ERROR_CODES.MISSING_REQUIRED_FIELD,
          `submission[${part.part}].answers[${answerRef.id}].text`,
        );
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
    submission_id: submissionId,
    template: body.template,
    parts: modalParts,
  };
}
