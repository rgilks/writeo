/**
 * Submission validation utilities
 */

import type { CreateSubmissionRequest } from "@writeo/shared";
import { validateText } from "../../utils/validation";
import { errorResponse, ERROR_CODES } from "../../utils/errors";
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

// Assessor IDs validation - accepts any strings, invalid ones are filtered later
const assessorsSchema = z.array(z.string()).optional();

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
  questionId: z.string().uuid("Invalid questionId format").optional(),
  questionText: z
    .union([z.string(), z.null()])
    .optional()
    .superRefine((val, ctx) => {
      if (val === null || val === undefined) {
        return; // null or undefined is allowed (null = free writing, undefined = referenced question)
      }
      if (val === "") {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message:
            "questionText cannot be empty string. Use null for free writing or omit for referenced questions.",
        });
        return;
      }
      answerTextValidator(val, ctx, MAX_QUESTION_TEXT_LENGTH, "questionText");
    }),
  text: z
    .string()
    .min(1, "Answer text is required")
    .superRefine((val, ctx) =>
      answerTextValidator(val, ctx, MAX_ANSWER_TEXT_LENGTH, "answer text"),
    ),
});

const submissionPartSchema = z.object({
  part: z.number().int().positive("Part must be a positive integer"),
  answers: z
    .array(answerSchema, { required_error: "Each submission part must include answers" })
    .min(1, "Each submission part must include at least one answer"),
});

const submissionSchema = z
  .object({
    submission: z
      .array(submissionPartSchema, { required_error: "Submission array is required" })
      .min(1, "Submission must include at least one part"),
    assessors: assessorsSchema,
    template: templateSchema.optional(), // Deprecated, kept for backward compat
    storeResults: z.boolean().optional(),
  })
  .passthrough()
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

        const questionId = answer.questionId;
        const questionText = answer.questionText;
        if (questionId && typeof questionText === "string" && questionText !== null) {
          const existing = questionsById.get(questionId);
          if (existing && existing !== questionText) {
            ctx.addIssue({
              code: z.ZodIssueCode.custom,
              message: `Conflicting question text provided for questionId ${questionId}`,
              path: ["submission", partIndex, "answers", answerIndex, "questionText"],
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
  c: Context<{ Bindings: Env; Variables: { requestId?: string } }>,
): ValidationResult | Response {
  const parsed = submissionSchema.safeParse(body);
  if (!parsed.success) {
    // Extract field path from zod error for better error reporting
    const firstError = parsed.error.errors[0];
    const fieldPath = firstError?.path?.join(".") || "unknown";
    return errorResponse(
      400,
      formatZodMessage(parsed.error, "Invalid submission payload"),
      c,
      ERROR_CODES.INVALID_SUBMISSION_FORMAT,
      fieldPath,
    );
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

      const questionId = answer.questionId;
      const questionText = answer.questionText;
      // If questionText is provided (non-null string), create/update the question
      if (
        questionId &&
        typeof questionText === "string" &&
        questionText !== null &&
        !questionsById.has(questionId)
      ) {
        questionsById.set(questionId, questionText);
        questionsToCreate.push({ id: questionId, text: questionText });
      }

      // questionId is required - either provided explicitly or must exist
      if (!questionId) {
        return errorResponse(
          400,
          "questionId is required for all answers",
          c,
          ERROR_CODES.MISSING_REQUIRED_FIELD,
          `submission[${part.part}].answers[${answer.id}].questionId`,
        );
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
