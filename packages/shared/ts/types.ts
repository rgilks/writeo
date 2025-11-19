// Question creation request
export interface CreateQuestionRequest {
  text: string;
}

// Answer creation request
export interface CreateAnswerRequest {
  "question-id": string;
  text: string;
}

// Submission creation request
// Supports two formats:
// 1. Reference format: answers reference existing answer IDs
// 2. Inline format: answers include question and answer text directly
export interface SubmissionPart {
  part: number;
  answers: Array<{
    id: string;
    "question-number": number;
    // Inline format (optional - if provided, question and answer will be auto-created)
    "question-id"?: string; // UUID of the question (will be created if doesn't exist)
    "question-text"?: string; // Question text (required for inline format)
    text?: string; // Answer text (required for inline format)
  }>;
}

export interface CreateSubmissionRequest {
  submission: SubmissionPart[];
  template: { name: string; version: number };
  // Note: parentSubmissionId and draftNumber are accepted but ignored for API compatibility
  // Draft tracking is handled in Server Actions, not the API
  // Opt-in server storage: if false or omitted, results are returned but not stored on server
  storeResults?: boolean; // Default: false (no server storage)
}

// Modal request (consumer → Modal)
// Note: Uses snake_case for Modal API compatibility
export interface ModalAnswer {
  id: string;
  question_id: string;
  question_text: string;
  answer_text: string;
}

export interface ModalRequest {
  submission_id: string;
  template: { name: string; version: number };
  parts: Array<{
    part: number;
    answers: ModalAnswer[];
  }>;
}

// LanguageTool error structure
export interface LanguageToolError {
  start: number; // Character offset (0-based)
  end: number; // Character offset (exclusive)
  length: number; // end - start (for convenience)
  sentenceIndex?: number; // Optional: sentence number (0-based)
  category: string; // Error category (e.g., "GRAMMAR", "TYPOGRAPHY", "STYLE")
  rule_id: string; // LanguageTool rule identifier
  message: string; // Human-readable error message
  suggestions?: string[]; // Array of suggested corrections (top 3-5)
  source: "LT" | "LLM"; // "LT" for LanguageTool, "LLM" for LLM assessment
  severity: "warning" | "error"; // Error severity level
  // Precision filtering fields
  confidenceScore?: number; // Confidence score (0-1), higher = more confident
  highConfidence?: boolean; // Whether this error meets the high-confidence threshold (>0.8)
  mediumConfidence?: boolean; // Whether this error meets the medium-confidence threshold (0.6-0.8)
  // Explicit, actionable feedback fields
  errorType?: string; // Human-readable error type (e.g., "Subject-verb agreement", "Article use")
  explanation?: string; // Plain-language explanation of the error
  example?: string; // Example correction pattern (e.g., "you goes → you go")
}

// Assessment results
export interface AssessorResult {
  id: string;
  name: string;
  type: "grader" | "conf" | "ard" | "feedback";
  overall?: number;
  label?: string;
  dimensions?: {
    TA?: number;
    CC?: number;
    Vocab?: number;
    Grammar?: number;
    Overall?: number;
  };
  errors?: LanguageToolError[]; // LanguageTool errors (for type: "feedback")
  meta?: Record<string, unknown>; // Assessor metadata
  [key: string]: unknown;
}

// Answer result with assessor results
export interface AnswerResult {
  id: string;
  "assessor-results": AssessorResult[];
}

export interface AssessmentPart {
  part: number;
  status: "success" | "error";
  answers: AnswerResult[];
}

export interface DraftMetadata {
  draftNumber: number;
  timestamp: string; // ISO 8601 timestamp
  wordCount: number;
  errorCount: number;
  overallScore?: number;
  previousDraftNumber?: number;
}

export interface AssessmentResults {
  status: "success" | "error" | "pending" | "bypassed";
  results?: {
    parts: AssessmentPart[];
  };
  template: { name: string; version: number };
  error_message?: string;
  meta?: Record<string, unknown>; // Metadata (e.g., answer texts, wordCount, errorCount, overallScore, timestamp, draft tracking info)
}

// CEFR mapping
export function mapScoreToCEFR(overall: number): string {
  if (overall >= 8.5) return "C2";
  if (overall >= 7.0) return "C1";
  if (overall >= 5.5) return "B2";
  if (overall >= 4.0) return "B1";
  return "A2";
}

// UUID validation
const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function isValidUUID(uuid: string): boolean {
  return UUID_REGEX.test(uuid);
}
