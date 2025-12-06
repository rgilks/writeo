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
    corpusUrl: string; // Default to deployed service URL
    feedbackUrl: string; // T-AES-FEEDBACK service URL
    gecUrl: string; // T-GEC-SEQ2SEQ service URL
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
    mockServices: boolean;
  };
  allowedOrigins?: string;
}

const DEFAULT_LT_LANGUAGE = "en-GB";

function requireEnv(key: string, value: string | undefined): string {
  if (!value) {
    throw new Error(`${key} is required`);
  }
  return value;
}

function buildLLMConfig(env: Env): { provider: LLMProvider; model: string; apiKey: string } {
  const provider = parseLLMProvider(env.LLM_PROVIDER);
  const apiKey = getAPIKey(provider, {
    GROQ_API_KEY: env.GROQ_API_KEY,
    OPENAI_API_KEY: env.OPENAI_API_KEY,
  });

  if (!apiKey) {
    const envKey = provider === "groq" ? "GROQ_API_KEY" : "OPENAI_API_KEY";
    throw new Error(`API key not found for provider: ${provider}. Please set ${envKey}`);
  }

  return {
    provider,
    model: env.AI_MODEL || getDefaultModel(provider),
    apiKey,
  };
}

/**
 * Validates and builds configuration from environment variables
 * Throws on missing required values (fail-fast)
 */
export function buildConfig(env: Env): AppConfig {
  return {
    api: {
      key: requireEnv("API_KEY", env.API_KEY),
      testKey: env.TEST_API_KEY,
    },
    modal: {
      gradeUrl: requireEnv("MODAL_GRADE_URL", env.MODAL_GRADE_URL),
      ltUrl: env.MODAL_LT_URL,
      corpusUrl: env.MODAL_CORPUS_URL || "https://rob-gilks--writeo-corpus-fastapi-app.modal.run",
      feedbackUrl:
        env.MODAL_FEEDBACK_URL || "https://rob-gilks--writeo-feedback-fastapi-app.modal.run",
      gecUrl: env.MODAL_GEC_URL || "https://rob-gilks--writeo-gec-service-gec-endpoint.modal.run",
    },
    llm: buildLLMConfig(env),
    storage: {
      r2Bucket: env.WRITEO_DATA,
      kvNamespace: env.WRITEO_RESULTS,
    },
    features: {
      languageTool: {
        enabled: !!env.MODAL_LT_URL,
        language: env.LT_LANGUAGE || DEFAULT_LT_LANGUAGE,
      },
      mockServices: env.USE_MOCK_SERVICES === "true",
    },
    allowedOrigins: env.ALLOWED_ORIGINS,
  };
}
