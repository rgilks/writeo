import { callGroqAPI } from "./groq";
import { callOpenAIAPI } from "./openai";

export type LLMProvider = "groq" | "openai";

export function getDefaultModel(provider: LLMProvider): string {
  return provider === "groq" ? "llama-3.3-70b-versatile" : "gpt-4o-mini";
}

export function getAPIKey(
  provider: LLMProvider,
  env: {
    GROQ_API_KEY?: string;
    OPENAI_API_KEY?: string;
  }
): string {
  return provider === "groq" ? env.GROQ_API_KEY || "" : env.OPENAI_API_KEY || "";
}

export async function callLLMAPI(
  provider: LLMProvider,
  apiKey: string,
  modelName: string,
  messages: Array<{ role: string; content: string }>,
  maxTokens: number
): Promise<string> {
  return provider === "groq"
    ? callGroqAPI(apiKey, modelName, messages, maxTokens)
    : callOpenAIAPI(apiKey, modelName, messages, maxTokens);
}

export function parseLLMProvider(provider?: string): LLMProvider {
  return provider?.toLowerCase().trim() === "groq" ? "groq" : "openai";
}

export async function* streamLLMAPI(
  provider: LLMProvider,
  apiKey: string,
  modelName: string,
  messages: Array<{ role: string; content: string }>,
  maxTokens: number
): AsyncGenerator<string, void, unknown> {
  if (provider === "openai") {
    const { streamOpenAIAPI } = await import("./openai");
    yield* streamOpenAIAPI(apiKey, modelName, messages, maxTokens);
    return;
  }
  // Groq streaming not yet implemented, fall back to non-streaming
  const { callGroqAPI } = await import("./groq");
  const response = await callGroqAPI(apiKey, modelName, messages, maxTokens);
  const words = response.match(/\S+|\s+/g) || [];
  for (const word of words) {
    yield word;
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}
