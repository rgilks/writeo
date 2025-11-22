import { Hono } from "hono";
import type { Env } from "../types/env";
import { errorResponse } from "../utils/errors";
import { safeLogError, sanitizeError } from "../utils/logging";
import { validateApiKey } from "./feedback/auth";
import { handleStreamingRequest } from "./feedback/handlers";
import { buildStreamingPrompt, createStreamingResponse } from "./feedback/streaming";
import { handleTeacherFeedbackRequest } from "./feedback/handlers";

export const feedbackRouter = new Hono<{ Bindings: Env }>();

feedbackRouter.post("/text/submissions/:submission_id/ai-feedback/stream", async (c) => {
  const authError = validateApiKey(c);
  if (authError) return authError;

  try {
    const requestData = await handleStreamingRequest(c);
    if (requestData instanceof Response) {
      return requestData;
    }

    const prompt = buildStreamingPrompt(
      requestData.questionText,
      requestData.answerText,
      requestData.essayScores,
      requestData.ltErrors
    );

    const stream = await createStreamingResponse(
      requestData.llmProvider,
      requestData.apiKey,
      requestData.aiModel,
      prompt
    );

    return new Response(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  } catch (error) {
    const sanitized = sanitizeError(error);
    safeLogError("Error streaming AI feedback", sanitized);
    return errorResponse(500, "Internal server error", c);
  }
});

feedbackRouter.post("/text/submissions/:submission_id/teacher-feedback", async (c) => {
  const authError = validateApiKey(c);
  if (authError) return authError;

  try {
    const result = await handleTeacherFeedbackRequest(c);
    if (result instanceof Response) {
      return result;
    }

    return c.json(result.feedback);
  } catch (error) {
    const sanitized = sanitizeError(error);
    safeLogError("Error getting Teacher feedback", sanitized);
    return errorResponse(500, "Internal server error", c);
  }
});
