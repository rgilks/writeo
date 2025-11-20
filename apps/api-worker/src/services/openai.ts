import { fetchWithTimeout } from "../utils/fetch-with-timeout";

function shouldUseMock(apiKey: string): boolean {
  const globalProcess = (globalThis as any).process;
  if (globalProcess?.env?.MOCK_OPENAI === "true") return true;
  return apiKey === "MOCK" || apiKey.startsWith("test_");
}

export async function callOpenAIAPI(
  apiKey: string,
  modelName: string,
  messages: Array<{ role: string; content: string }>,
  maxTokens: number
): Promise<string> {
  if (shouldUseMock(apiKey)) {
    const { mockCallLLMAPI } = await import("./llm.mock");
    return mockCallLLMAPI(apiKey, modelName, messages, maxTokens);
  }

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

  return data.choices[0].message.content || "";
}

export async function* streamOpenAIAPI(
  apiKey: string,
  modelName: string,
  messages: Array<{ role: string; content: string }>,
  maxTokens: number
): AsyncGenerator<string, void, unknown> {
  if (shouldUseMock(apiKey)) {
    const { mockStreamLLMAPI } = await import("./llm.mock");
    yield* mockStreamLLMAPI(apiKey, modelName, messages, maxTokens);
    return;
  }

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
      stream: true,
    }),
    timeout: 30000,
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

      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";

      for (const event of events) {
        if (!event.trim()) continue;
        const lines = event.split("\n");
        let dataLine = "";
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            dataLine = line.slice(6).trim();
            break;
          }
        }
        if (!dataLine || dataLine === "[DONE]") continue;

        try {
          const data = JSON.parse(dataLine) as {
            choices?: Array<{ delta?: { content?: string } }>;
          };
          if (data.choices?.[0]?.delta?.content) {
            yield data.choices[0].delta.content;
          }
        } catch {
          // Skip malformed JSON
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
