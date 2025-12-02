/**
 * Cloudflare Workers environment bindings.
 * Injected at runtime from wrangler.toml and secrets.
 */
export type Env = {
  // Storage bindings
  WRITEO_DATA: R2Bucket;
  WRITEO_RESULTS: KVNamespace;

  // External service URLs
  MODAL_GRADE_URL: string;
  MODAL_LT_URL?: string;
  /** Default: "en-GB" */
  LT_LANGUAGE?: string;

  // API authentication
  API_KEY: string;
  /** Test API key with higher rate limits */
  TEST_API_KEY?: string;
  /** Enable mocked services */
  USE_MOCK_SERVICES?: string;

  // AI services
  AI: Ai;
  /** Required when LLM_PROVIDER=groq */
  GROQ_API_KEY?: string;
  /** Required when LLM_PROVIDER=openai */
  OPENAI_API_KEY?: string;
  /** "groq" or "openai" (default: "openai") */
  LLM_PROVIDER?: string;
  /** Defaults to provider-specific default */
  AI_MODEL?: string;

  // CORS configuration
  /** Comma-separated origins */
  ALLOWED_ORIGINS?: string;
};

/**
 * Execution context for Cloudflare Workers.
 * Minimal type definition; actual type from @cloudflare/workers-types
 * is passed as the third parameter to the fetch handler.
 */
export type ExecutionContext = {
  waitUntil(promise: Promise<any>): void;
  passThroughOnException(): void;
};
