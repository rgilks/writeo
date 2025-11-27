/**
 * Zod schemas for feedback route payloads
 */

import { z } from "zod";
import { MAX_ANSWER_TEXT_LENGTH } from "../../utils/constants";

const EssayScoreDimensionsSchema = z
  .object({
    TA: z.number().min(0).max(9).optional(),
    CC: z.number().min(0).max(9).optional(),
    Vocab: z.number().min(0).max(9).optional(),
    Grammar: z.number().min(0).max(9).optional(),
    Overall: z.number().min(0).max(9).optional(),
  })
  .partial();

const EssayScoresSchema = z.object({
  overall: z.number().min(0).max(9).optional(),
  label: z.string().optional(),
  dimensions: EssayScoreDimensionsSchema.optional(),
});

const LanguageToolErrorSchema = z.object({
  message: z.string(),
  category: z.string(),
  suggestions: z.array(z.string()).optional(),
  start: z.number().optional(),
  end: z.number().optional(),
  errorType: z.string().optional(),
});

const RelevanceCheckSchema = z.object({
  addressesQuestion: z.boolean(),
  score: z.number(),
  threshold: z.number().optional(),
});

export const AssessmentDataSchema = z.object({
  essayScores: EssayScoresSchema.optional(),
  ltErrors: z.array(LanguageToolErrorSchema).optional(),
  llmErrors: z.array(LanguageToolErrorSchema).optional(),
  relevanceCheck: RelevanceCheckSchema.optional(),
});

export const TeacherFeedbackRequestSchema = z.object({
  answerId: z.string().uuid(),
  mode: z.enum(["clues", "explanation"]),
  answerText: z.string().min(1).max(MAX_ANSWER_TEXT_LENGTH),
  questionText: z.string().optional(),
  assessmentData: AssessmentDataSchema.optional(),
});

export const StreamingFeedbackRequestSchema = z.object({
  answerId: z.string().uuid(),
  answerText: z.string().min(1),
  questionText: z.string().optional(),
  assessmentData: AssessmentDataSchema.optional(),
});

export type AssessmentDataInput = z.infer<typeof AssessmentDataSchema>;
export type TeacherFeedbackRequestBody = z.infer<typeof TeacherFeedbackRequestSchema>;
export type StreamingFeedbackRequestBody = z.infer<typeof StreamingFeedbackRequestSchema>;

export function formatZodError(error: z.ZodError): string {
  return error.issues
    .map((issue) => `${issue.path.join(".") || "<root>"}: ${issue.message}`)
    .join("; ");
}
