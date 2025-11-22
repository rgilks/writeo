/**
 * Submission validation utilities
 */

import type { CreateSubmissionRequest, SubmissionPart } from "@writeo/shared";
import { isValidUUID } from "@writeo/shared";
import { validateText } from "../../utils/validation";
import { errorResponse } from "../../utils/errors";
import { MAX_ANSWER_TEXT_LENGTH } from "../../utils/constants";
import type { Context } from "hono";
import type { Env } from "../../types/env";

export interface ValidationResult {
  answerIds: string[];
  questionsToCreate: Array<{ id: string; text: string }>;
  answersToCreate: Array<{
    id: string;
    questionId: string;
    answerText: string;
  }>;
}

function validateAnswer(
  answer: SubmissionPart["answers"][number],
  answerIds: string[],
  questionsToCreate: Array<{ id: string; text: string }>,
  answersToCreate: Array<{ id: string; questionId: string; answerText: string }>,
  c: Context<{ Bindings: Env }>
): Response | null {
  if (!answer.id || !isValidUUID(answer.id)) {
    return errorResponse(400, `Invalid answer id: ${answer.id}`, c);
  }
  answerIds.push(answer.id);

  const answerText = answer["text"];
  if (!answerText) {
    return errorResponse(
      400,
      `Answer text is required. Answers must be sent inline with the submission.`,
      c
    );
  }

  if (!answer["question-id"]) {
    return errorResponse(400, `question-id is required for each answer`, c);
  }

  const questionId = answer["question-id"];
  if (!isValidUUID(questionId)) {
    return errorResponse(400, `Invalid question-id format: ${questionId}`, c);
  }

  const answerTextValidation = validateText(answerText, MAX_ANSWER_TEXT_LENGTH);
  if (!answerTextValidation.valid) {
    return errorResponse(
      400,
      `Invalid answer text: ${answerTextValidation.error || "Invalid content"}`,
      c
    );
  }

  const questionText = answer["question-text"];
  if (questionText) {
    const questionTextValidation = validateText(questionText, 10000);
    if (!questionTextValidation.valid) {
      return errorResponse(
        400,
        `Invalid question-text: ${questionTextValidation.error || "Invalid content"}`,
        c
      );
    }
    questionsToCreate.push({ id: questionId, text: questionText });
  }

  answersToCreate.push({
    id: answer.id,
    questionId: questionId,
    answerText: answerText,
  });

  return null;
}

export function validateSubmissionBody(
  body: CreateSubmissionRequest,
  c: Context<{ Bindings: Env }>
): ValidationResult | Response {
  if (!body.submission || !Array.isArray(body.submission)) {
    return errorResponse(400, "Missing or invalid 'submission' array", c);
  }

  if (!body.template || !body.template.name || typeof body.template.version !== "number") {
    return errorResponse(400, "Missing or invalid 'template' object", c);
  }

  const answerIds: string[] = [];
  const questionsToCreate: Array<{ id: string; text: string }> = [];
  const answersToCreate: Array<{
    id: string;
    questionId: string;
    answerText: string;
  }> = [];

  for (const part of body.submission) {
    if (!part.part || !Array.isArray(part.answers)) {
      return errorResponse(400, "Invalid submission part structure", c);
    }
    for (const answer of part.answers) {
      const result = validateAnswer(answer, answerIds, questionsToCreate, answersToCreate, c);
      if (result) return result;
    }
  }

  return { answerIds, questionsToCreate, answersToCreate };
}
