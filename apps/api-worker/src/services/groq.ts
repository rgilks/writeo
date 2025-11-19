import { fetchWithTimeout } from "../utils/fetch-with-timeout";

export async function callGroqAPI(
  apiKey: string,
  modelName: string,
  messages: Array<{ role: string; content: string }>,
  maxTokens: number
): Promise<string> {
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
  };

  if (!data.choices || !data.choices[0] || !data.choices[0].message) {
    throw new Error(`Invalid Groq API response: ${JSON.stringify(data)}`);
  }

  return data.choices[0].message.content || "";
}
