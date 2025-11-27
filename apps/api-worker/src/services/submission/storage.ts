/**
 * Submission storage operations
 */

import type { CreateSubmissionRequest } from "@writeo/shared";
import { StorageService } from "../storage";
import { errorResponse } from "../../utils/errors";
import type { Context } from "hono";
import type { Env } from "../../types/env";
import type { ValidationResult } from "./validator";

type StorageConflict = Response & { status: 409 };

function conflictResponse(message: string, c: Context<{ Bindings: Env }>): StorageConflict {
  return errorResponse(409, message, c) as StorageConflict;
}

async function storeQuestions(
  storage: StorageService,
  questionsToCreate: Array<{ id: string; text: string }>,
  c: Context<{ Bindings: Env }>,
): Promise<Response | null> {
  const existingQuestions = new Map<string, Awaited<ReturnType<StorageService["getQuestion"]>>>();

  for (const question of questionsToCreate) {
    if (!existingQuestions.has(question.id)) {
      existingQuestions.set(question.id, await storage.getQuestion(question.id));
    }
    const existing = existingQuestions.get(question.id);
    if (!existing) {
      try {
        await storage.putQuestion(question.id, { text: question.text });
      } catch (error) {
        return errorResponse(
          500,
          `Failed to store question ${question.id}: ${
            error instanceof Error ? error.message : String(error)
          }`,
          c,
        );
      }
    } else if (existing.text !== question.text) {
      return conflictResponse(`Question ${question.id} already exists with different content`, c);
    }
  }
  return null;
}

async function storeAnswers(
  storage: StorageService,
  answersToCreate: Array<{ id: string; questionId: string; answerText: string }>,
  createdQuestionIds: Set<string>,
  c: Context<{ Bindings: Env }>,
): Promise<Response | null> {
  const existingAnswers = new Map<string, Awaited<ReturnType<StorageService["getAnswer"]>>>();
  const questionExistenceCache = new Map<string, boolean>();

  for (const answer of answersToCreate) {
    if (!existingAnswers.has(answer.id)) {
      existingAnswers.set(answer.id, await storage.getAnswer(answer.id));
    }
    const existing = existingAnswers.get(answer.id);
    if (!existing) {
      if (!createdQuestionIds.has(answer.questionId)) {
        if (!questionExistenceCache.has(answer.questionId)) {
          const questionExists = await storage.getQuestion(answer.questionId);
          questionExistenceCache.set(answer.questionId, Boolean(questionExists));
        }
        if (!questionExistenceCache.get(answer.questionId)) {
          return conflictResponse(`Referenced question does not exist: ${answer.questionId}`, c);
        }
      }
      try {
        await storage.putAnswer(answer.id, {
          "question-id": answer.questionId,
          text: answer.answerText,
        });
      } catch (error) {
        return errorResponse(
          500,
          `Failed to store answer ${answer.id}: ${
            error instanceof Error ? error.message : String(error)
          }`,
          c,
        );
      }
    } else if (
      existing["question-id"] !== answer.questionId ||
      existing.text !== answer.answerText
    ) {
      return conflictResponse(`Answer ${answer.id} already exists with different content`, c);
    }
  }
  return null;
}

export async function storeSubmissionEntities(
  storage: StorageService,
  validation: ValidationResult,
  submissionId: string,
  body: CreateSubmissionRequest,
  c: Context<{ Bindings: Env }>,
): Promise<Response | null> {
  const { questionsToCreate, answersToCreate } = validation;

  const questionResult = await storeQuestions(storage, questionsToCreate, c);
  if (questionResult) return questionResult;

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
