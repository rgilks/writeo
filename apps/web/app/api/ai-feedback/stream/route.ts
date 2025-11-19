/**
 * API Route: Stream AI Feedback
 * Proxies SSE stream from API worker to client
 * This is needed because Server Actions can't directly handle SSE from client components
 */

import { NextRequest } from "next/server";
import { getApiBase, getApiKey } from "@/app/lib/api-config";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { submissionId, answerId, answerText } = body;

    if (!submissionId || !answerId || !answerText) {
      return new Response(
        JSON.stringify({ error: "Missing required fields: submissionId, answerId, answerText" }),
        {
          status: 400,
          headers: { "Content-Type": "application/json" },
        }
      );
    }

    const apiBase = getApiBase();
    const apiKey = getApiKey();

    if (apiKey === "MISSING_API_KEY" || apiBase === "MISSING_API_BASE_URL") {
      return new Response(
        JSON.stringify({ error: "Server configuration error: API credentials not set" }),
        {
          status: 500,
          headers: { "Content-Type": "application/json" },
        }
      );
    }

    // Call the API worker's streaming endpoint
    const response = await fetch(`${apiBase}/text/submissions/${submissionId}/ai-feedback/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Token ${apiKey}`,
      },
      body: JSON.stringify({
        answerId,
        answerText,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      return new Response(
        JSON.stringify({ error: errorText || `Failed to stream feedback: ${response.status}` }),
        {
          status: response.status,
          headers: { "Content-Type": "application/json" },
        }
      );
    }

    // Return the SSE stream with proper headers
    return new Response(response.body, {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
        "X-Accel-Buffering": "no", // Disable buffering for streaming
      },
    });
  } catch (error) {
    console.error("[API Route] Error streaming AI feedback:", error);
    const errorMessage = error instanceof Error ? error.message : String(error);
    return new Response(JSON.stringify({ error: `Internal server error: ${errorMessage}` }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
