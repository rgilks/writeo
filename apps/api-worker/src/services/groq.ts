import { fetchWithTimeout } from "../utils/fetch-with-timeout";

const GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions";
const REQUEST_TIMEOUT_MS = 30000;
const DEFAULT_TEMPERATURE = 0.3;

interface GroqAPIResponse {
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
}

function shouldUseMock(apiKey: string): boolean {
  const globalProcess = (globalThis as any).process;
  // USE_MOCK_LLM enables deterministic mock responses (saves API costs)
  if (globalProcess?.env?.USE_MOCK_LLM === "true") return true;
  // API key-based mocking for local development
  return apiKey === "MOCK" || apiKey.startsWith("test_");
}

function parseAPIResponse(data: GroqAPIResponse): string {
  const content = data.choices?.[0]?.message?.content;
  if (!content) {
    throw new Error(`Invalid Groq API response: missing content in ${JSON.stringify(data)}`);
  }
  return content;
}

export async function callGroqAPI(
  apiKey: string,
  modelName: string,
  messages: Array<{ role: string; content: string }>,
  maxTokens: number,
): Promise<string> {
  if (shouldUseMock(apiKey)) {
    const { mockCallLLMAPI } = await import("./llm.mock");
    return mockCallLLMAPI(apiKey, modelName, messages, maxTokens);
  }

  const response = await fetchWithTimeout(GROQ_API_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: modelName,
      messages,
      max_tokens: maxTokens,
      temperature: DEFAULT_TEMPERATURE,
    }),
    timeout: REQUEST_TIMEOUT_MS,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Groq API error: ${response.status} ${response.statusText} - ${errorText}`);
  }

  const data = (await response.json()) as GroqAPIResponse;
  return parseAPIResponse(data);
}
