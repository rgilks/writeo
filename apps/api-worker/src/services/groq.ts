import { fetchWithTimeout } from "../utils/fetch-with-timeout";

function shouldUseMock(apiKey: string): boolean {
  const globalProcess = (globalThis as any).process;
  // USE_MOCK_LLM enables deterministic mock responses (saves API costs)
  if (globalProcess?.env?.USE_MOCK_LLM === "true") return true;
  // API key-based mocking for local development
  return apiKey === "MOCK" || apiKey.startsWith("test_");
}

export async function callGroqAPI(
  apiKey: string,
  modelName: string,
  messages: Array<{ role: string; content: string }>,
  maxTokens: number
): Promise<string> {
  if (shouldUseMock(apiKey)) {
    const { mockCallLLMAPI } = await import("./llm.mock");
    return mockCallLLMAPI(apiKey, modelName, messages, maxTokens);
  }

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
    timeout: 30000,
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

  return data.choices[0].message.content || "";
}
