import { Hono } from "hono";
import type { Env } from "../types/env";
import { withErrorHandling } from "../utils/handlers";
import { validateApiKey } from "./feedback/auth";
import {
  handleStreamingRequest,
  handleTeacherFeedbackRequest,
  buildStreamingPrompt,
  createStreamingResponse,
} from "./feedback/handlers";

export const feedbackRouter = new Hono<{
  Bindings: Env;
  Variables: { requestId?: string };
}>();

feedbackRouter.post(
  "/v1/text/submissions/:submission_id/ai-feedback/stream",
  withErrorHandling(async (c) => {
    const authError = validateApiKey(c);
    if (authError) return authError;

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
  }, "Error streaming AI feedback"),
);

feedbackRouter.post(
  "/v1/text/submissions/:submission_id/teacher-feedback",
  withErrorHandling(async (c) => {
    const authError = validateApiKey(c);
    if (authError) return authError;

    const result = await handleTeacherFeedbackRequest(c);
    if (result instanceof Response) {
      return result;
    }

    return c.json(result.feedback);
  }, "Error getting Teacher feedback"),
);
