import { Hono } from "hono";
import type { Context } from "hono";
import type { Env } from "../types/env";
import { errorResponse } from "../utils/errors";
import { safeLogError, sanitizeError } from "../utils/logging";
import { validateApiKey } from "./feedback/auth";
import {
  handleStreamingRequest,
  handleTeacherFeedbackRequest,
  buildStreamingPrompt,
  createStreamingResponse,
} from "./feedback/handlers";

export const feedbackRouter = new Hono<{ Bindings: Env }>();

feedbackRouter.post("/text/submissions/:submission_id/ai-feedback/stream", (c) =>
  handleFeedbackRoute(c, "Error streaming AI feedback", async () => {
    const requestData = await handleStreamingRequest(c);
    if (requestData instanceof Response) {
      return requestData;
    }

    const prompt = buildStreamingPrompt(
      requestData.questionText,
      requestData.answerText,
      requestData.essayScores,
      requestData.ltErrors,
    );

    const stream = await createStreamingResponse(
      requestData.llmProvider,
      requestData.apiKey,
      requestData.aiModel,
      prompt,
    );

    return new Response(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  }),
);

feedbackRouter.post("/text/submissions/:submission_id/teacher-feedback", (c) =>
  handleFeedbackRoute(c, "Error getting Teacher feedback", async () => {
    const result = await handleTeacherFeedbackRequest(c);
    if (result instanceof Response) {
      return result;
    }

    return c.json(result.feedback);
  }),
);

async function handleFeedbackRoute(
  c: Context<{ Bindings: Env }>,
  logContext: string,
  handler: () => Promise<Response>,
): Promise<Response> {
  const authError = validateApiKey(c);
  if (authError) return authError;

  try {
    return await handler();
  } catch (error) {
    const sanitized = sanitizeError(error);
    safeLogError(logContext, sanitized);
    return errorResponse(500, "Internal server error", c);
  }
}
