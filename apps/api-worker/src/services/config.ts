/**
 * Configuration service - validates and provides type-safe access to environment variables
 *
 * Note: Cloudflare Workers runtime bindings (like `AI`) are not included here
 * as they are runtime-specific and should be accessed directly from env.
 */

import type { Env } from "../types/env";
import { parseLLMProvider, getDefaultModel, getAPIKey, type LLMProvider } from "./llm";

export interface AppConfig {
  api: {
    key: string;
    testKey?: string;
  };
  modal: {
    gradeUrl: string;
    ltUrl?: string;
  };
  llm: {
    provider: LLMProvider;
    model: string;
    apiKey: string;
  };
  storage: {
    r2Bucket: R2Bucket;
    kvNamespace: KVNamespace;
  };
  features: {
    languageTool: {
      enabled: boolean;
      language: string;
    };
  };
  allowedOrigins?: string;
}

/**
 * Validates and builds configuration from environment variables
 * Throws on missing required values (fail-fast)
 */
export function buildConfig(env: Env): AppConfig {
  if (!env.API_KEY) {
    throw new Error("API_KEY is required");
  }

  if (!env.MODAL_GRADE_URL) {
    throw new Error("MODAL_GRADE_URL is required");
  }

  const llmProvider = parseLLMProvider(env.LLM_PROVIDER);
  const llmApiKey = getAPIKey(llmProvider, {
    GROQ_API_KEY: env.GROQ_API_KEY,
    OPENAI_API_KEY: env.OPENAI_API_KEY,
  });

  if (!llmApiKey) {
    throw new Error(
      `API key not found for provider: ${llmProvider}. Please set ${llmProvider === "groq" ? "GROQ_API_KEY" : "OPENAI_API_KEY"}`
    );
  }

  return {
    api: {
      key: env.API_KEY,
      testKey: env.TEST_API_KEY,
    },
    modal: {
      gradeUrl: env.MODAL_GRADE_URL,
      ltUrl: env.MODAL_LT_URL,
    },
    llm: {
      provider: llmProvider,
      model: env.AI_MODEL || getDefaultModel(llmProvider),
      apiKey: llmApiKey,
    },
    storage: {
      r2Bucket: env.WRITEO_DATA,
      kvNamespace: env.WRITEO_RESULTS,
    },
    features: {
      languageTool: {
        enabled: !!env.MODAL_LT_URL,
        language: env.LT_LANGUAGE || "en-GB",
      },
    },
    allowedOrigins: env.ALLOWED_ORIGINS,
  };
}
