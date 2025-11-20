import { callGroqAPI } from "./groq";
import { callOpenAIAPI } from "./openai";

export type LLMProvider = "groq" | "openai" | "anthropic" | "google";

/**
 * Get the default model name for a given provider
 */
export function getDefaultModel(provider: LLMProvider): string {
  switch (provider) {
    case "openai":
      return "gpt-4o-mini";
    case "groq":
      return "llama-3.3-70b-versatile";
    case "anthropic":
      return "claude-3-haiku-20240307";
    case "google":
      return "gemini-1.5-flash";
    default:
      return "gpt-4o-mini";
  }
}

/**
 * Get the API key for a given provider from environment
 */
export function getAPIKey(
  provider: LLMProvider,
  env: {
    GROQ_API_KEY?: string;
    OPENAI_API_KEY?: string;
    ANTHROPIC_API_KEY?: string;
    GOOGLE_API_KEY?: string;
  }
): string {
  switch (provider) {
    case "groq":
      return env.GROQ_API_KEY || "";
    case "openai":
      return env.OPENAI_API_KEY || "";
    case "anthropic":
      return env.ANTHROPIC_API_KEY || "";
    case "google":
      return env.GOOGLE_API_KEY || "";
    default:
      return env.OPENAI_API_KEY || "";
  }
}

/**
 * Call LLM API based on provider
 * This is a provider-agnostic wrapper that routes to the correct API
 */
export async function callLLMAPI(
  provider: LLMProvider,
  apiKey: string,
  modelName: string,
  messages: Array<{ role: string; content: string }>,
  maxTokens: number
): Promise<string> {
  switch (provider) {
    case "groq":
      return callGroqAPI(apiKey, modelName, messages, maxTokens);
    case "openai":
      return callOpenAIAPI(apiKey, modelName, messages, maxTokens);
    case "anthropic":
      throw new Error("Anthropic provider not yet implemented");
    case "google":
      throw new Error("Google provider not yet implemented");
    default:
      // Default to OpenAI if provider is unknown
      return callOpenAIAPI(apiKey, modelName, messages, maxTokens);
  }
}

/**
 * Parse LLM_PROVIDER environment variable
 */
export function parseLLMProvider(provider?: string): LLMProvider {
  if (!provider) {
    return "openai"; // Default to OpenAI
  }
  const normalized = provider.toLowerCase().trim();
  if (normalized === "groq") return "groq";
  if (normalized === "openai") return "openai";
  if (normalized === "anthropic") return "anthropic";
  if (normalized === "google") return "google";
  return "openai"; // Default to OpenAI for unknown values
}

/**
 * Stream LLM API responses based on provider
 * This is a provider-agnostic wrapper that routes to the correct streaming API
 */
export async function* streamLLMAPI(
  provider: LLMProvider,
  apiKey: string,
  modelName: string,
  messages: Array<{ role: string; content: string }>,
  maxTokens: number
): AsyncGenerator<string, void, unknown> {
  switch (provider) {
    case "openai": {
      const { streamOpenAIAPI } = await import("./openai");
      yield* streamOpenAIAPI(apiKey, modelName, messages, maxTokens);
      return;
    }
    case "groq":
      // Groq streaming not yet implemented, fall back to non-streaming
      const { callGroqAPI } = await import("./groq");
      const response = await callGroqAPI(apiKey, modelName, messages, maxTokens);
      // Simulate streaming by yielding word by word
      const words = response.match(/\S+|\s+/g) || [];
      for (const word of words) {
        yield word;
        await new Promise((resolve) => setTimeout(resolve, 10));
      }
      return;
    case "anthropic":
      throw new Error("Anthropic provider streaming not yet implemented");
    case "google":
      throw new Error("Google provider streaming not yet implemented");
    default:
      // Default to OpenAI if provider is unknown
      const { streamOpenAIAPI: defaultStreamOpenAIAPI } = await import("./openai");
      yield* defaultStreamOpenAIAPI(apiKey, modelName, messages, maxTokens);
      return;
  }
}
