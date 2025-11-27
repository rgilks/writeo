import { Hono } from "hono";
import type { Env } from "../types/env";
import { isValidUUID } from "@writeo/shared";
import { errorResponse } from "../utils/errors";
import { safeLogError, sanitizeError } from "../utils/logging";
import { validateText, validateRequestBodySize } from "../utils/validation";
import { StorageService } from "../services/storage";
import type { CreateQuestionRequest } from "@writeo/shared";
import { MAX_REQUEST_BODY_SIZE, MAX_QUESTION_LENGTH } from "../utils/constants";

export const questionsRouter = new Hono<{ Bindings: Env }>();

questionsRouter.put("/text/questions/:question_id", async (c) => {
  const questionId = c.req.param("question_id");
  if (!isValidUUID(questionId)) {
    return errorResponse(400, "Invalid question_id format", c);
  }

  try {
    const sizeValidation = await validateRequestBodySize(c.req.raw, MAX_REQUEST_BODY_SIZE);
    if (!sizeValidation.valid) {
      return errorResponse(413, sizeValidation.error || "Request body too large (max 1MB)", c);
    }

    const body = await c.req.json<CreateQuestionRequest>();
    if (!body.text || typeof body.text !== "string") {
      return errorResponse(400, "Missing or invalid 'text' field", c);
    }

    const textValidation = validateText(body.text, MAX_QUESTION_LENGTH);
    if (!textValidation.valid) {
      return errorResponse(400, textValidation.error || "Invalid text content", c);
    }

    const storage = new StorageService(c.env.WRITEO_DATA, c.env.WRITEO_RESULTS);
    const existing = await storage.getQuestion(questionId);

    if (existing) {
      if (existing.text === body.text) {
        return new Response(null, { status: 204 });
      }
      return c.json(
        {
          error: "Question already exists with different content",
          existingQuestion: {
            id: questionId,
            text: existing.text,
          },
        },
        { status: 409 },
      );
    }

    await storage.putQuestion(questionId, body);
    return new Response(null, { status: 201 });
  } catch (error) {
    const sanitized = sanitizeError(error);
    safeLogError("Error creating question", sanitized);
    return errorResponse(500, "Internal server error", c);
  }
});
