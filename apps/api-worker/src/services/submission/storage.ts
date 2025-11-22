/**
 * Submission storage operations
 */

import type { CreateSubmissionRequest } from "@writeo/shared";
import { StorageService } from "../storage";
import { errorResponse } from "../../utils/errors";
import type { Context } from "hono";
import type { Env } from "../../types/env";
import type { ValidationResult } from "./validator";

async function storeQuestions(
  storage: StorageService,
  questionsToCreate: Array<{ id: string; text: string }>,
  c: Context<{ Bindings: Env }>
): Promise<Response | null> {
  const questionPromises = questionsToCreate.map(async (question) => {
    const existing = await storage.getQuestion(question.id);
    if (!existing) {
      await storage.putQuestion(question.id, { text: question.text });
    } else {
      if (existing.text !== question.text) {
        throw new Error(`Question ${question.id} already exists with different content`);
      }
    }
  });

  const questionResults = await Promise.allSettled(questionPromises);
  for (const result of questionResults) {
    if (result.status === "rejected") {
      return errorResponse(
        409,
        result.reason instanceof Error ? result.reason.message : String(result.reason),
        c
      );
    }
  }
  return null;
}

async function storeAnswers(
  storage: StorageService,
  answersToCreate: Array<{ id: string; questionId: string; answerText: string }>,
  createdQuestionIds: Set<string>,
  c: Context<{ Bindings: Env }>
): Promise<Response | null> {
  const answerPromises = answersToCreate.map(async (answer) => {
    const existing = await storage.getAnswer(answer.id);
    if (!existing) {
      if (!createdQuestionIds.has(answer.questionId)) {
        const questionExists = await storage.getQuestion(answer.questionId);
        if (!questionExists) {
          throw new Error(`Referenced question does not exist: ${answer.questionId}`);
        }
      }
      await storage.putAnswer(answer.id, {
        "question-id": answer.questionId,
        text: answer.answerText,
      });
    } else {
      if (existing["question-id"] !== answer.questionId || existing.text !== answer.answerText) {
        throw new Error(`Answer ${answer.id} already exists with different content`);
      }
    }
  });

  const answerResults = await Promise.allSettled(answerPromises);
  for (const result of answerResults) {
    if (result.status === "rejected") {
      return errorResponse(
        409,
        result.reason instanceof Error ? result.reason.message : String(result.reason),
        c
      );
    }
  }
  return null;
}

export async function storeSubmissionEntities(
  storage: StorageService,
  validation: ValidationResult,
  submissionId: string,
  body: CreateSubmissionRequest,
  c: Context<{ Bindings: Env }>
): Promise<Response | null> {
  const { questionsToCreate, answersToCreate } = validation;

  const questionResult = await storeQuestions(storage, questionsToCreate, c);
  if (questionResult) return questionResult;

  const answersWithoutQuestionText = answersToCreate.filter((answer) => {
    return !questionsToCreate.some((q) => q.id === answer.questionId);
  });

  for (const answer of answersWithoutQuestionText) {
    const existingQuestion = await storage.getQuestion(answer.questionId);
    if (!existingQuestion) {
      return errorResponse(
        400,
        `question-text is required when question ${answer.questionId} does not exist`,
        c
      );
    }
  }

  const createdQuestionIds = new Set(questionsToCreate.map((q) => q.id));
  const answerResult = await storeAnswers(storage, answersToCreate, createdQuestionIds, c);
  if (answerResult) return answerResult;

  const existing = await storage.getSubmission(submissionId);
  if (existing) {
    if (JSON.stringify(existing) === JSON.stringify(body)) {
      return new Response(null, { status: 204 });
    } else {
      return errorResponse(409, "Submission already exists with different content", c);
    }
  }

  await storage.putSubmission(submissionId, body);
  return null;
}
