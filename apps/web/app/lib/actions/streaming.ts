/**
 * Streaming server actions
 */

"use server";

import { getApiBase, getApiKey } from "../api-config";
import { getErrorMessage, makeSerializableError } from "../utils/error-handling";
import { retryWithBackoff } from "@writeo/shared";

export async function streamAIFeedback(
  submissionId: string,
  answerId: string,
  answerText: string,
  questionText?: string,
  assessmentData?: any
): Promise<Response> {
  try {
    if (!submissionId || !answerId || !answerText) {
      throw new Error("Missing required fields: submissionId, answerId, answerText");
    }

    const apiBase = getApiBase();
    const apiKey = getApiKey();

    if (apiKey === "MISSING_API_KEY" || apiBase === "MISSING_API_BASE_URL") {
      throw new Error("Server configuration error: API credentials not set");
    }

    const requestBody: any = {
      answerId,
      answerText,
    };

    if (questionText) {
      requestBody.questionText = questionText;
    }
    if (assessmentData) {
      requestBody.assessmentData = assessmentData;
    }

    const response = await fetch(`${apiBase}/text/submissions/${submissionId}/ai-feedback/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Token ${apiKey}`,
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      throw new Error(`Failed to stream feedback: ${await getErrorMessage(response)}`);
    }

    return response;
  } catch (error) {
    console.error("[streamAIFeedback] Error:", error);
    const errorMessage = error instanceof Error ? error.message : String(error);
    return new Response(`data: ${JSON.stringify({ type: "error", message: errorMessage })}\n\n`, {
      status: 500,
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  }
}
