import { fetchWithTimeout } from "../utils/fetch-with-timeout";

/**
 * Check if mocking should be enabled
 * - If MOCK_OPENAI env var is "true" (for Node.js/test environments)
 * - If API key is "MOCK" or starts with "test_" (for worker environments)
 */
function shouldUseMock(apiKey: string): boolean {
  // Check Node.js environment variable (for tests running in Node.js)
  // Use globalThis to safely access process in both Node.js and Workers environments
  const globalProcess = (globalThis as any).process;
  if (globalProcess?.env?.MOCK_OPENAI === "true") {
    return true;
  }
  // Check for test API key patterns (for worker environments)
  if (apiKey === "MOCK" || apiKey.startsWith("test_")) {
    return true;
  }
  return false;
}

/**
 * Call OpenAI API or return mock response if MOCK_OPENAI=true
 *
 * Token usage estimates:
 * - Grammar check (getLLMAssessment): ~2000-3000 input tokens, ~500-1000 output tokens (max 2500)
 * - Detailed feedback (getCombinedFeedback): ~3000-5000 input tokens, ~400-500 output tokens (max 500)
 * - Teacher feedback (getTeacherFeedback): ~2000-4000 input tokens, ~100-200 output tokens (initial/clues: max 150, explanation: max 800)
 *
 * Cost estimate: ~$0.01 per request (varies by model and token count)
 */
export async function callOpenAIAPI(
  apiKey: string,
  modelName: string,
  messages: Array<{ role: string; content: string }>,
  maxTokens: number
): Promise<string> {
  // Use mock if enabled (for tests)
  if (shouldUseMock(apiKey)) {
    const { mockCallOpenAIAPI } = await import("./openai.mock");
    return mockCallOpenAIAPI(apiKey, modelName, messages, maxTokens);
  }

  // Real OpenAI API call
  // OpenAI API timeout: 30 seconds (reasonable for LLM requests)
  const response = await fetchWithTimeout("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: modelName,
      messages: messages,
      max_tokens: maxTokens,
      temperature: 0.3,
    }),
    timeout: 30000, // 30 seconds
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`OpenAI API error: ${response.status} ${response.statusText} - ${errorText}`);
  }

  const data = (await response.json()) as {
    choices?: Array<{
      message?: {
        content?: string;
      };
    }>;
    usage?: {
      prompt_tokens: number;
      completion_tokens: number;
      total_tokens: number;
    };
  };

  if (!data.choices || !data.choices[0] || !data.choices[0].message) {
    throw new Error(`Invalid OpenAI API response: ${JSON.stringify(data)}`);
  }

  // Log token usage in development (helps track costs)
  const globalProcess = (globalThis as any).process;
  if (globalProcess?.env?.NODE_ENV === "development" && data.usage) {
    console.log(
      `[OpenAI API] Tokens used: ${data.usage.prompt_tokens} prompt + ${data.usage.completion_tokens} completion = ${data.usage.total_tokens} total`
    );
  }

  return data.choices[0].message.content || "";
}

/**
 * Stream OpenAI API responses using native streaming
 * Yields text chunks as they arrive from the API
 */
export async function* streamOpenAIAPI(
  apiKey: string,
  modelName: string,
  messages: Array<{ role: string; content: string }>,
  maxTokens: number
): AsyncGenerator<string, void, unknown> {
  // Use mock if enabled (for tests)
  if (shouldUseMock(apiKey)) {
    const { mockStreamOpenAIAPI } = await import("./openai.mock");
    yield* mockStreamOpenAIAPI(apiKey, modelName, messages, maxTokens);
    return;
  }

  // Real OpenAI streaming API call
  const response = await fetchWithTimeout("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: modelName,
      messages: messages,
      max_tokens: maxTokens,
      temperature: 0.3,
      stream: true, // Enable streaming
    }),
    timeout: 30000, // 30 seconds
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`OpenAI API error: ${response.status} ${response.statusText} - ${errorText}`);
  }

  if (!response.body) {
    throw new Error("OpenAI API response body is null");
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

      // Process complete SSE messages (events are separated by \n\n)
      const events = buffer.split("\n\n");
      buffer = events.pop() || ""; // Keep incomplete event in buffer

      for (const event of events) {
        if (!event.trim()) continue;

        // Process each line in the event
        const lines = event.split("\n");
        let dataLine = "";
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            dataLine = line.slice(6).trim();
            break;
          }
        }

        if (!dataLine) continue;

        if (dataLine === "[DONE]") {
          return;
        }

        try {
          const data = JSON.parse(dataLine) as {
            choices?: Array<{
              delta?: {
                content?: string;
              };
            }>;
          };

          if (data.choices && data.choices[0]?.delta?.content) {
            yield data.choices[0].delta.content;
          }
        } catch (parseError) {
          // Skip malformed JSON lines
          console.error("[OpenAI Stream] Failed to parse chunk:", dataLine);
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
