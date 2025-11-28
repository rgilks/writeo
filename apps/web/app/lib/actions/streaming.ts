"use server";

import type { AssessmentData } from "../types/assessment";
import { getApiBase, getApiKey } from "../api-config";
import { getErrorMessage } from "../utils/error-handling";

interface RequestBody {
  answerId: string;
  answerText: string;
  questionText?: string;
  assessmentData?: AssessmentData;
}

const SSE_ERROR_HEADERS = {
  "Content-Type": "text/event-stream",
  "Cache-Control": "no-cache",
  Connection: "keep-alive",
} as const;

function createErrorResponse(message: string): Response {
  return new Response(`data: ${JSON.stringify({ type: "error", message })}\n\n`, {
    status: 500,
    headers: SSE_ERROR_HEADERS,
  });
}

export async function streamAIFeedback(
  submissionId: string,
  answerId: string,
  answerText: string,
  questionText?: string,
  assessmentData?: AssessmentData,
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

    const requestBody: RequestBody = {
      answerId,
      answerText,
      ...(questionText && { questionText }),
      ...(assessmentData && { assessmentData }),
    };

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
    return createErrorResponse(errorMessage);
  }
}
