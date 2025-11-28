"use client";

import { useState, useCallback, useRef } from "react";
import { z } from "zod";
import type { AssessmentData } from "@/app/lib/types/assessment";

const StreamEventSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("start"),
    message: z.string().optional(),
  }),
  z.object({
    type: z.literal("chunk"),
    text: z.string().min(1),
    message: z.string().optional(),
  }),
  z.object({
    type: z.literal("done"),
    message: z.string().optional(),
  }),
  z.object({
    type: z.literal("error"),
    message: z.string().optional(),
  }),
]);

type StreamEvent = z.infer<typeof StreamEventSchema>;

interface UseAIFeedbackStreamReturn {
  feedback: string;
  isStreaming: boolean;
  error: string | null;
  startStream: (
    submissionId: string,
    answerId: string,
    answerText: string,
    questionText?: string,
    assessmentData?: AssessmentData,
  ) => Promise<void>;
  stopStream: () => void;
}

const SSE_DATA_PREFIX = "data: ";
const SSE_MESSAGE_DELIMITER = "\n\n";

function parseSSEEvent(line: string): StreamEvent | null {
  if (!line.startsWith(SSE_DATA_PREFIX)) {
    return null;
  }

  const jsonStr = line.slice(SSE_DATA_PREFIX.length).trim();
  if (!jsonStr) {
    return null;
  }

  try {
    return StreamEventSchema.parse(JSON.parse(jsonStr));
  } catch {
    return null;
  }
}

export function useAIFeedbackStream(): UseAIFeedbackStreamReturn {
  const [feedback, setFeedback] = useState<string>("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const startStream = useCallback(
    async (
      submissionId: string,
      answerId: string,
      answerText: string,
      questionText?: string,
      assessmentData?: AssessmentData,
    ) => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      setFeedback("");
      setError(null);
      setIsStreaming(true);

      const abortController = new AbortController();
      abortControllerRef.current = abortController;

      try {
        const requestBody = {
          submissionId,
          answerId,
          answerText,
          ...(questionText && { questionText }),
          ...(assessmentData && { assessmentData }),
        };

        const response = await fetch("/api/ai-feedback/stream", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestBody),
          signal: abortController.signal,
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ error: "Unknown error" }));
          throw new Error(errorData.error || `HTTP ${response.status}`);
        }

        if (!response.body) {
          throw new Error("Response body is null");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        try {
          while (true) {
            const { done, value } = await reader.read();

            if (done) {
              break;
            }

            buffer += decoder.decode(value, { stream: true });

            const messages = buffer.split(SSE_MESSAGE_DELIMITER);
            buffer = messages.pop() || "";

            for (const message of messages) {
              if (!message.trim()) continue;

              const event = parseSSEEvent(message);
              if (!event) continue;

              switch (event.type) {
                case "start":
                  setFeedback("");
                  break;

                case "chunk":
                  if (event.text) {
                    setFeedback((prev) => prev + event.text);
                  }
                  break;

                case "done":
                  setIsStreaming(false);
                  return;

                case "error":
                  throw new Error(event.message || "Stream error");
              }
            }
          }
        } finally {
          reader.releaseLock();
        }
      } catch (err) {
        if (err instanceof Error && err.name === "AbortError") {
          return;
        }
        const errorMessage = err instanceof Error ? err.message : String(err);
        setError(errorMessage);
        setIsStreaming(false);
      }
    },
    [],
  );

  const stopStream = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  return {
    feedback,
    isStreaming,
    error,
    startStream,
    stopStream,
  };
}
