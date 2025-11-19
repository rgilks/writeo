/**
 * Shared constants for text processing and API limits
 */

// Text truncation limits (to control costs and context size)
export const MAX_ESSAY_LENGTH = 15000; // ~2500 words
export const MAX_QUESTION_LENGTH = 500; // characters

// Token limits for Groq API calls
export const MAX_TOKENS_GRAMMAR_CHECK = 2500;
export const MAX_TOKENS_DETAILED_FEEDBACK = 500;
export const MAX_TOKENS_TEACHER_FEEDBACK_INITIAL = 150;
export const MAX_TOKENS_TEACHER_FEEDBACK_EXPLANATION = 800;

// Request body size limits
export const MAX_REQUEST_BODY_SIZE = 1024 * 1024; // 1MB
export const MAX_ANSWER_TEXT_LENGTH = 50000; // characters
