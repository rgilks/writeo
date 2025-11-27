"use client";

import { useState, useCallback, useRef } from "react";

interface StreamEvent {
  type: "start" | "chunk" | "done" | "error";
  message?: string;
  text?: string;
}

interface UseAIFeedbackStreamReturn {
  feedback: string;
  isStreaming: boolean;
  error: string | null;
  startStream: (
    submissionId: string,
    answerId: string,
    answerText: string,
    questionText?: string,
    assessmentData?: {
      essayScores?: {
        overall?: number;
        dimensions?: {
          TA?: number;
          CC?: number;
          Vocab?: number;
          Grammar?: number;
          Overall?: number;
        };
      };
      ltErrors?: any[];
      llmErrors?: any[];
    },
  ) => Promise<void>;
  stopStream: () => void;
}

/**
 * Hook for consuming AI feedback SSE stream
 */
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
      assessmentData?: {
        essayScores?: {
          overall?: number;
          dimensions?: {
            TA?: number;
            CC?: number;
            Vocab?: number;
            Grammar?: number;
            Overall?: number;
          };
        };
        ltErrors?: any[];
        llmErrors?: any[];
      },
    ) => {
      // Stop any existing stream
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      // Reset state
      setFeedback("");
      setError(null);
      setIsStreaming(true);

      // Create new abort controller
      const abortController = new AbortController();
      abortControllerRef.current = abortController;

      try {
        const requestBody: any = {
          submissionId,
          answerId,
          answerText,
        };

        if (questionText) {
          requestBody.questionText = questionText;
        }

        if (assessmentData) {
          requestBody.assessmentData = assessmentData;
        }

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

            // Decode chunk and add to buffer
            buffer += decoder.decode(value, { stream: true });

            // Process complete SSE messages (lines ending with \n\n)
            const lines = buffer.split("\n\n");
            buffer = lines.pop() || ""; // Keep incomplete line in buffer

            for (const line of lines) {
              // Skip empty lines
              if (!line.trim()) continue;

              // Handle SSE data lines
              if (line.startsWith("data: ")) {
                try {
                  const jsonStr = line.slice(6).trim();
                  if (!jsonStr) continue;

                  const data = JSON.parse(jsonStr) as StreamEvent;

                  if (data.type === "start") {
                    setFeedback("");
                    console.log("[Stream] Start event received");
                  } else if (data.type === "chunk" && data.text) {
                    setFeedback((prev) => {
                      const newFeedback = prev + data.text;
                      console.log("[Stream] Chunk received, length:", newFeedback.length);
                      return newFeedback;
                    });
                  } else if (data.type === "done") {
                    console.log("[Stream] Done event received");
                    setIsStreaming(false);
                    break;
                  } else if (data.type === "error") {
                    throw new Error(data.message || "Stream error");
                  }
                } catch (parseError) {
                  console.error("Failed to parse SSE event:", line, parseError);
                  // Don't break on parse errors, continue processing
                }
              } else {
                // Log non-data lines for debugging
                console.log("[Stream] Non-data line:", line.substring(0, 50));
              }
            }
          }
        } finally {
          reader.releaseLock();
        }
      } catch (err) {
        if (err instanceof Error && err.name === "AbortError") {
          // Stream was aborted, this is expected
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
