export type Env = {
  WRITEO_DATA: R2Bucket;
  WRITEO_RESULTS: KVNamespace;
  MODAL_GRADE_URL: string;
  MODAL_LT_URL?: string;
  LT_LANGUAGE?: string;
  API_KEY: string;
  TEST_API_KEY?: string; // Optional test API key with higher rate limits
  AI: Ai;
  GROQ_API_KEY?: string; // Optional, used when LLM_PROVIDER=groq
  OPENAI_API_KEY?: string; // Optional, used when LLM_PROVIDER=openai
  LLM_PROVIDER?: string; // Provider: "groq" or "openai" (default: "openai")
  AI_MODEL?: string;
  ALLOWED_ORIGINS?: string;
};

export type ExecutionContext = {
  waitUntil(promise: Promise<any>): void;
  passThroughOnException(): void;
};
