// Text length limits
export const MAX_ESSAY_LENGTH = 15000;
export const MAX_QUESTION_LENGTH = 500;
export const MAX_ANSWER_TEXT_LENGTH = 50000;

// LLM token limits
/**
 * Reduced from 6000: pipe-delimited format is more compact.
 * With truncated text (12k chars) and concise prompt, 4000 tokens is sufficient.
 */
export const MAX_TOKENS_GRAMMAR_CHECK = 4000;
export const MAX_TOKENS_DETAILED_FEEDBACK = 500;
export const MAX_TOKENS_TEACHER_FEEDBACK_INITIAL = 150;
export const MAX_TOKENS_TEACHER_FEEDBACK_EXPLANATION = 800;

// Request limits
export const MAX_REQUEST_BODY_SIZE = 1024 * 1024; // 1MB

// Public paths (no authentication or rate limiting)
export const PUBLIC_PATHS = ["/health", "/docs", "/openapi.json"] as const;

export const KEY_OWNER = {
  ADMIN: "admin",
  TEST_RUNNER: "test-runner",
  UNKNOWN: "unknown",
} as const;

/**
 * Checks if a path is a public path that doesn't require authentication.
 */
export function isPublicPath(path: string): boolean {
  return PUBLIC_PATHS.includes(path as (typeof PUBLIC_PATHS)[number]);
}
