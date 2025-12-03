import { fetchWithTimeout } from "../utils/fetch-with-timeout";

const OPENAI_API_URL = "https://api.openai.com/v1/chat/completions";
const REQUEST_TIMEOUT_MS = 30000;
const DEFAULT_TEMPERATURE = 0.3;
const SSE_DATA_PREFIX = "data: ";
const SSE_DONE_TOKEN = "[DONE]";

interface OpenAIAPIResponse {
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

interface OpenAIStreamDelta {
  choices?: Array<{ delta?: { content?: string } }>;
}

function shouldUseMock(apiKey: string, useMockServices?: boolean): boolean {
  // USE_MOCK_SERVICES explicitly disables mocking when false
  if (useMockServices === false) return false;
  // USE_MOCK_SERVICES enables deterministic mock responses (saves API costs)
  if (useMockServices === true) return true;
  // Fallback: check global process.env (for Node.js test environments)
  const globalProcess = (globalThis as any).process;
  if (globalProcess?.env?.USE_MOCK_SERVICES === "true") return true;
  if (globalProcess?.env?.USE_MOCK_SERVICES === "false") return false;
  // API key-based mocking for local development (only if USE_MOCK_SERVICES is not set)
  return apiKey === "MOCK" || apiKey.startsWith("test_");
}

function parseAPIResponse(data: OpenAIAPIResponse): string {
  const content = data.choices?.[0]?.message?.content;
  if (!content) {
    throw new Error(`Invalid OpenAI API response: missing content in ${JSON.stringify(data)}`);
  }
  return content;
}

async function* parseStreamResponse(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<string, void, unknown> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";

      for (const event of events) {
        if (!event.trim()) continue;

        const dataLine = extractDataLine(event);
        if (!dataLine || dataLine === SSE_DONE_TOKEN) continue;

        const content = parseStreamDelta(dataLine);
        if (content) {
          yield content;
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

function extractDataLine(event: string): string {
  const lines = event.split("\n");
  for (const line of lines) {
    if (line.startsWith(SSE_DATA_PREFIX)) {
      return line.slice(SSE_DATA_PREFIX.length).trim();
    }
  }
  return "";
}

function parseStreamDelta(dataLine: string): string | null {
  try {
    const data = JSON.parse(dataLine) as OpenAIStreamDelta;
    return data.choices?.[0]?.delta?.content || null;
  } catch {
    // Skip malformed JSON
    return null;
  }
}

export async function callOpenAIAPI(
  apiKey: string,
  modelName: string,
  messages: Array<{ role: string; content: string }>,
  maxTokens: number,
  useMockServices?: boolean,
): Promise<string> {
  if (shouldUseMock(apiKey, useMockServices)) {
    const { mockCallLLMAPI } = await import("./llm.mock");
    return mockCallLLMAPI(apiKey, modelName, messages, maxTokens);
  }

  const response = await fetchWithTimeout(OPENAI_API_URL, {
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
    throw new Error(`OpenAI API error: ${response.status} ${response.statusText} - ${errorText}`);
  }

  const data = (await response.json()) as OpenAIAPIResponse;
  return parseAPIResponse(data);
}

export async function* streamOpenAIAPI(
  apiKey: string,
  modelName: string,
  messages: Array<{ role: string; content: string }>,
  maxTokens: number,
  useMockServices?: boolean,
): AsyncGenerator<string, void, unknown> {
  if (shouldUseMock(apiKey, useMockServices)) {
    const { mockStreamLLMAPI } = await import("./llm.mock");
    yield* mockStreamLLMAPI(apiKey, modelName, messages, maxTokens);
    return;
  }

  const response = await fetchWithTimeout(OPENAI_API_URL, {
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
      stream: true,
    }),
    timeout: REQUEST_TIMEOUT_MS,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`OpenAI API error: ${response.status} ${response.statusText} - ${errorText}`);
  }

  if (!response.body) {
    throw new Error("OpenAI API response body is null");
  }

  yield* parseStreamResponse(response.body);
}
