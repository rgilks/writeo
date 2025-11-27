/**
 * Submission validation utilities
 */

import type { CreateSubmissionRequest } from "@writeo/shared";
import { validateText } from "../../utils/validation";
import { errorResponse } from "../../utils/errors";
import { MAX_ANSWER_TEXT_LENGTH } from "../../utils/constants";
import type { Context } from "hono";
import type { Env } from "../../types/env";
import { z } from "zod";
import { formatZodMessage } from "../../utils/zod";

export interface ValidationResult {
  answerIds: string[];
  questionsToCreate: Array<{ id: string; text: string }>;
  answersToCreate: Array<{
    id: string;
    questionId: string;
    answerText: string;
  }>;
}

const MAX_QUESTION_TEXT_LENGTH = 10000;

const templateSchema = z.object({
  name: z.string().min(1, "Template name is required"),
  version: z.number({ invalid_type_error: "Template version must be a number" }),
});

const answerTextValidator = (
  value: string,
  ctx: z.RefinementCtx,
  maxLength: number,
  label: string,
) => {
  const validation = validateText(value, maxLength);
  if (!validation.valid) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: `Invalid ${label}: ${validation.error || "Invalid content"}`,
    });
  }
};

const answerSchema = z.object({
  id: z.string().uuid("Invalid answer id"),
  text: z
    .string()
    .superRefine((val, ctx) =>
      answerTextValidator(val, ctx, MAX_ANSWER_TEXT_LENGTH, "answer text"),
    ),
  "question-id": z.string().uuid("Invalid question-id format"),
  "question-text": z
    .string()
    .optional()
    .superRefine((val, ctx) => {
      if (typeof val === "undefined") {
        return;
      }
      answerTextValidator(val, ctx, MAX_QUESTION_TEXT_LENGTH, "question-text");
    }),
});

const submissionPartSchema = z.object({
  part: z.string().min(1, "Each submission part must include a part identifier"),
  answers: z
    .array(answerSchema, { required_error: "Each submission part must include answers" })
    .min(1, "Each submission part must include at least one answer"),
});

const submissionSchema = z
  .object({
    submission: z
      .array(submissionPartSchema, { required_error: "Submission array is required" })
      .min(1, "Submission must include at least one part"),
    template: templateSchema,
  })
  .strict()
  .superRefine((data, ctx) => {
    const answerIds = new Set<string>();
    const questionsById = new Map<string, string>();
    data.submission.forEach((part, partIndex) => {
      part.answers.forEach((answer, answerIndex) => {
        if (answerIds.has(answer.id)) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            message: `Duplicate answer id detected: ${answer.id}`,
            path: ["submission", partIndex, "answers", answerIndex, "id"],
          });
        } else {
          answerIds.add(answer.id);
        }

        const questionId = answer["question-id"];
        const questionText = answer["question-text"];
        if (typeof questionText === "string") {
          const existing = questionsById.get(questionId);
          if (existing && existing !== questionText) {
            ctx.addIssue({
              code: z.ZodIssueCode.custom,
              message: `Conflicting question text provided for question-id ${questionId}`,
              path: ["submission", partIndex, "answers", answerIndex, "question-text"],
            });
          } else if (!existing) {
            questionsById.set(questionId, questionText);
          }
        }
      });
    });
  });

export function validateSubmissionBody(
  body: CreateSubmissionRequest,
  c: Context<{ Bindings: Env }>,
): ValidationResult | Response {
  const parsed = submissionSchema.safeParse(body);
  if (!parsed.success) {
    return errorResponse(400, formatZodMessage(parsed.error, "Invalid submission payload"), c);
  }

  const answerIds: string[] = [];
  const questionsById = new Map<string, string>();
  const questionsToCreate: Array<{ id: string; text: string }> = [];
  const answersToCreate: Array<{
    id: string;
    questionId: string;
    answerText: string;
  }> = [];

  for (const part of parsed.data.submission) {
    for (const answer of part.answers) {
      answerIds.push(answer.id);

      const questionId = answer["question-id"];
      const questionText = answer["question-text"];
      if (typeof questionText === "string" && !questionsById.has(questionId)) {
        questionsById.set(questionId, questionText);
        questionsToCreate.push({ id: questionId, text: questionText });
      }

      answersToCreate.push({
        id: answer.id,
        questionId,
        answerText: answer.text,
      });
    }
  }

  return { answerIds, questionsToCreate, answersToCreate };
}
