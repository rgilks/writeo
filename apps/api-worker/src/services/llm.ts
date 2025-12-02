import { callGroqAPI } from "./groq";
import { callOpenAIAPI } from "./openai";

export type LLMProvider = "groq" | "openai";

const DEFAULT_MODELS: Record<LLMProvider, string> = {
  groq: "llama-3.3-70b-versatile",
  openai: "gpt-4o-mini",
};

const GROQ_FALLBACK_STREAM_DELAY_MS = 10;

export function getDefaultModel(provider: LLMProvider): string {
  return DEFAULT_MODELS[provider];
}

export function getAPIKey(
  provider: LLMProvider,
  env: {
    GROQ_API_KEY?: string;
    OPENAI_API_KEY?: string;
  },
): string {
  if (provider === "groq") {
    return env.GROQ_API_KEY || "";
  }
  return env.OPENAI_API_KEY || "";
}

export async function callLLMAPI(
  provider: LLMProvider,
  apiKey: string,
  modelName: string,
  messages: Array<{ role: string; content: string }>,
  maxTokens: number,
  useMockServices?: boolean,
): Promise<string> {
  if (provider === "groq") {
    return callGroqAPI(apiKey, modelName, messages, maxTokens, useMockServices);
  }
  return callOpenAIAPI(apiKey, modelName, messages, maxTokens, useMockServices);
}

export function parseLLMProvider(provider?: string): LLMProvider {
  const normalized = provider?.toLowerCase().trim();
  return normalized === "groq" ? "groq" : "openai";
}

function simulateStreaming(text: string): AsyncGenerator<string, void, unknown> {
  return (async function* () {
    const words = text.match(/\S+|\s+/g) || [];
    for (const word of words) {
      yield word;
      await new Promise((resolve) => setTimeout(resolve, GROQ_FALLBACK_STREAM_DELAY_MS));
    }
  })();
}

export async function* streamLLMAPI(
  provider: LLMProvider,
  apiKey: string,
  modelName: string,
  messages: Array<{ role: string; content: string }>,
  maxTokens: number,
  useMockServices?: boolean,
): AsyncGenerator<string, void, unknown> {
  if (provider === "openai") {
    const { streamOpenAIAPI } = await import("./openai");
    yield* streamOpenAIAPI(apiKey, modelName, messages, maxTokens, useMockServices);
    return;
  }

  // Groq streaming not yet implemented, fall back to non-streaming with simulated streaming
  const { callGroqAPI } = await import("./groq");
  const response = await callGroqAPI(apiKey, modelName, messages, maxTokens, useMockServices);
  yield* simulateStreaming(response);
}
