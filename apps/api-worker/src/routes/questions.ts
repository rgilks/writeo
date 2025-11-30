import { Hono } from "hono";
import type { Env } from "../types/env";
import { errorResponse } from "../utils/errors";
import { safeLogError, sanitizeError } from "../utils/logging";
import { validateText, validateRequestBodySize } from "../utils/validation";
import { getServices } from "../utils/context";
import type { CreateQuestionRequest } from "@writeo/shared";
import { MAX_REQUEST_BODY_SIZE, MAX_QUESTION_LENGTH } from "../utils/constants";
import { z } from "zod";
import { formatZodMessage, uuidStringSchema } from "../utils/zod";

export const questionsRouter = new Hono<{
  Bindings: Env;
  Variables: { requestId?: string };
}>();

const questionIdSchema = uuidStringSchema("question_id");

const questionTextSchema: z.ZodType<CreateQuestionRequest> = z
  .object({
    text: z.string().superRefine((val, ctx) => {
      const validation = validateText(val, MAX_QUESTION_LENGTH);
      if (!validation.valid) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: validation.error || "Invalid text content",
        });
      }
    }),
  })
  .strict();

questionsRouter.put("/v1/text/questions/:question_id", async (c) => {
  const questionId = c.req.param("question_id");
  const parsedQuestionId = questionIdSchema.safeParse(questionId);
  if (!parsedQuestionId.success) {
    return errorResponse(
      400,
      formatZodMessage(parsedQuestionId.error, "Invalid question_id format"),
      c,
    );
  }

  try {
    const sizeValidation = await validateRequestBodySize(c.req.raw, MAX_REQUEST_BODY_SIZE);
    if (!sizeValidation.valid) {
      return errorResponse(413, sizeValidation.error || "Request body too large (max 1MB)", c);
    }

    const parsedBody = questionTextSchema.safeParse(await c.req.json());
    if (!parsedBody.success) {
      return errorResponse(400, formatZodMessage(parsedBody.error, "Invalid question payload"), c);
    }
    const body = parsedBody.data;

    const { storage } = getServices(c);
    const existing = await storage.getQuestion(parsedQuestionId.data);

    if (existing) {
      if (existing.text === body.text) {
        return new Response(null, { status: 204 });
      }
      return c.json(
        {
          error: "Question already exists with different content",
          existingQuestion: {
            id: parsedQuestionId.data,
            text: existing.text,
          },
        },
        { status: 409 },
      );
    }

    await storage.putQuestion(parsedQuestionId.data, body);
    const url = new URL(c.req.url);
    return new Response(null, {
      status: 201,
      headers: {
        Location: `${url.origin}/v1/text/questions/${parsedQuestionId.data}`,
      },
    });
  } catch (error) {
    const sanitized = sanitizeError(error);
    safeLogError("Error creating question", sanitized, c);
    return errorResponse(500, "Internal server error", c);
  }
});
