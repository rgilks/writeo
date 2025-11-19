import { fetchWithTimeout } from "../utils/fetch-with-timeout";

/**
 * Check if mocking should be enabled
 * - If MOCK_GROQ env var is "true" (for Node.js/test environments)
 * - If API key is "MOCK" or starts with "test_" (for worker environments)
 */
function shouldUseMock(apiKey: string): boolean {
  // Check Node.js environment variable (for tests running in Node.js)
  // Use globalThis to safely access process in both Node.js and Workers environments
  const globalProcess = (globalThis as any).process;
  if (globalProcess?.env?.MOCK_GROQ === "true") {
    return true;
  }
  // Check for test API key patterns (for worker environments)
  if (apiKey === "MOCK" || apiKey.startsWith("test_")) {
    return true;
  }
  return false;
}

/**
 * Call Groq API or return mock response if MOCK_GROQ=true
 *
 * Token usage estimates:
 * - Grammar check (getLLMAssessment): ~2000-3000 input tokens, ~500-1000 output tokens (max 2500)
 * - Detailed feedback (getCombinedFeedback): ~3000-5000 input tokens, ~400-500 output tokens (max 500)
 * - Teacher feedback (getTeacherFeedback): ~2000-4000 input tokens, ~100-200 output tokens (initial/clues: max 150, explanation: max 800)
 *
 * Cost estimate: ~$0.01 per request (varies by model and token count)
 */
export async function callGroqAPI(
  apiKey: string,
  modelName: string,
  messages: Array<{ role: string; content: string }>,
  maxTokens: number
): Promise<string> {
  // Use mock if enabled (for tests)
  if (shouldUseMock(apiKey)) {
    const { mockCallGroqAPI } = await import("./groq.mock");
    return mockCallGroqAPI(apiKey, modelName, messages, maxTokens);
  }

  // Real Groq API call
  // Groq API timeout: 30 seconds (reasonable for LLM requests)
  const response = await fetchWithTimeout("https://api.groq.com/openai/v1/chat/completions", {
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
    throw new Error(`Groq API error: ${response.status} ${response.statusText} - ${errorText}`);
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
    throw new Error(`Invalid Groq API response: ${JSON.stringify(data)}`);
  }

  // Log token usage in development (helps track costs)
  const globalProcess = (globalThis as any).process;
  if (globalProcess?.env?.NODE_ENV === "development" && data.usage) {
    console.log(
      `[Groq API] Tokens used: ${data.usage.prompt_tokens} prompt + ${data.usage.completion_tokens} completion = ${data.usage.total_tokens} total`
    );
  }

  return data.choices[0].message.content || "";
}
