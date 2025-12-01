/**
 * Proxies SSE stream from API worker to client
 * (Server Actions can't directly handle SSE from client components)
 */

import { NextRequest } from "next/server";
import { getApiBase, getApiKey } from "@/app/lib/api-config";

interface StreamRequest {
  submissionId: string;
  answerId: string;
  answerText: string;
  questionText?: string;
  assessmentData?: unknown;
}

interface ApiRequestBody {
  answerId: string;
  answerText: string;
  questionText?: string;
  assessmentData?: unknown;
}

const SSE_HEADERS = {
  "Content-Type": "text/event-stream",
  "Cache-Control": "no-cache",
  Connection: "keep-alive",
  "X-Accel-Buffering": "no",
} as const;

function errorResponse(message: string, status: number): Response {
  return new Response(JSON.stringify({ error: message }), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function validateRequest(body: unknown): body is StreamRequest {
  return (
    typeof body === "object" &&
    body !== null &&
    "submissionId" in body &&
    "answerId" in body &&
    "answerText" in body &&
    typeof (body as StreamRequest).submissionId === "string" &&
    typeof (body as StreamRequest).answerId === "string" &&
    typeof (body as StreamRequest).answerText === "string"
  );
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    if (!validateRequest(body)) {
      return errorResponse("Missing required fields: submissionId, answerId, answerText", 400);
    }

    const { submissionId, answerId, answerText, questionText, assessmentData } = body;

    const apiBase = getApiBase();
    const apiKey = getApiKey();

    if (apiKey === "MISSING_API_KEY" || apiBase === "MISSING_API_BASE_URL") {
      return errorResponse("Server configuration error: API credentials not set", 500);
    }

    const requestBody: ApiRequestBody = {
      answerId,
      answerText,
      ...(questionText ? { questionText } : {}),
      ...(assessmentData !== undefined ? { assessmentData } : {}),
    };
    const response = await fetch(`${apiBase}/v1/text/submissions/${submissionId}/ai-feedback/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Token ${apiKey}`,
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorText = await response.text();
      return errorResponse(
        errorText || `Failed to stream feedback: ${response.status}`,
        response.status,
      );
    }

    return new Response(response.body, {
      status: 200,
      headers: SSE_HEADERS,
    });
  } catch (error) {
    console.error("[API Route] Error streaming AI feedback:", error);
    const errorMessage = error instanceof Error ? error.message : String(error);
    return errorResponse(`Internal server error: ${errorMessage}`, 500);
  }
}
