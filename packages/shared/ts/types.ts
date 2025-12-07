// Question creation request
export interface CreateQuestionRequest {
  text: string;
}

// Answer creation request
export interface CreateAnswerRequest {
  questionId: string;
  text: string;
}

// Submission creation request
// Supports two formats:
// 1. Reference format: answers reference existing question IDs
// 2. Inline format: answers include question and answer text directly
export interface SubmissionPart {
  part: number; // Always a number (no string coercion)
  answers: Array<{
    id: string;
    // Inline format (optional - if provided, question and answer will be auto-created)
    questionId?: string; // UUID of the question (will be created if questionText is provided, otherwise question must exist)
    questionText?: string | null; // Question text: if provided (non-null), create/update question; if null, free writing (no question); if omitted, question must exist
    text: string; // Answer text (required - answers must always be sent inline)
  }>;
}

export interface CreateSubmissionRequest {
  submission: SubmissionPart[];
  assessors?: string[]; // Optional list of assessor IDs to run (defaults to all enabled)

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
  assessors: string[]; // List of assessor IDs to run
  parts: Array<{
    part: number;
    answers: ModalAnswer[];
  }>;
}

// LanguageTool API response types
export interface LanguageToolReplacement {
  value: string;
}

export interface LanguageToolRuleCategory {
  id?: string;
  name?: string;
}

export interface LanguageToolRule {
  id?: string;
  description?: string;
  category?: LanguageToolRuleCategory;
  type?: string;
}

export interface LanguageToolMatchContext {
  text?: string;
  offset?: number;
  length?: number;
}

export interface LanguageToolMatch {
  offset?: number;
  length?: number;
  message?: string;
  shortMessage?: string;
  rule?: LanguageToolRule;
  replacements?: LanguageToolReplacement[];
  issueType?: "error" | "warning";
  context?: LanguageToolMatchContext;
}

export interface LanguageToolResponse {
  matches?: LanguageToolMatch[];
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

export interface EssayScoreDimensions {
  TA?: number;
  CC?: number;
  Vocab?: number;
  Grammar?: number;
  Overall?: number;
}

export interface EssayScores {
  overall?: number;
  dimensions?: EssayScoreDimensions;
  label?: string;
}

export interface RelevanceCheck {
  addressesQuestion: boolean;
  score: number;
  threshold?: number;
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
  assessorResults: AssessorResult[];
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
  requestedAssessors: string[]; // What client asked for (or defaults if not specified)
  activeAssessors: string[]; // What actually ran
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

// Utility types for AssessorResult lookups
export type AssessorResultId =
  | "AES-ESSAY"
  | "AES-CORPUS" // Corpus-trained RoBERTa model (dev mode)
  | "AES-FEEDBACK" // Multi-task feedback model (dev mode)
  | "GEC-LT"
  | "GEC-LLM"
  | "GEC-SEQ2SEQ" // Seq2Seq GEC model (best grammar correction)
  | "TEACHER-FEEDBACK"
  | "RELEVANCE-CHECK";

/**
 * Type guard to check if an AssessorResult has a specific ID
 */
export function isAssessorResultWithId(
  result: AssessorResult,
  id: AssessorResultId,
): result is AssessorResult & { id: typeof id } {
  return result.id === id;
}

/**
 * Find an AssessorResult by ID from an array
 */
export function findAssessorResultById(
  results: AssessorResult[],
  id: AssessorResultId,
): AssessorResult | undefined {
  return results.find((r) => r.id === id);
}

/**
 * Type-safe helper to get essay assessor result from an array of assessor results.
 *
 * @param results - Array of assessor results to search
 * @returns Essay assessor result with guaranteed overall score and dimensions, or undefined if not found
 *
 * @example
 * ```typescript
 * const essayAssessor = getEssayAssessorResult(assessorResults);
 * const overall = essayAssessor?.overall ?? 0;
 * ```
 */
export function getEssayAssessorResult(results: AssessorResult[]):
  | (AssessorResult & {
      id: "AES-ESSAY";
      overall: number;
      dimensions: NonNullable<AssessorResult["dimensions"]>;
    })
  | undefined {
  const result = findAssessorResultById(results, "AES-ESSAY");
  if (result && result.overall !== undefined && result.dimensions) {
    return result as AssessorResult & {
      id: "AES-ESSAY";
      overall: number;
      dimensions: NonNullable<AssessorResult["dimensions"]>;
    };
  }
  return undefined;
}

/**
 * Type-safe helper to get corpus assessor result from an array of assessor results.
 *
 * @param results - Array of assessor results to search
 * @returns Corpus assessor result with guaranteed overall score and label, or undefined if not found
 */
export function getCorpusAssessorResult(results: AssessorResult[]):
  | (AssessorResult & {
      id: "AES-CORPUS";
      overall: number;
      label: string;
    })
  | undefined {
  const result = findAssessorResultById(results, "AES-CORPUS");
  if (result && "overall" in result && "label" in result) {
    return result as AssessorResult & { id: "AES-CORPUS"; overall: number; label: string };
  }
  return undefined;
}

/**
 * Get AES-FEEDBACK assessor result (multi-task feedback model)
 */
export function getFeedbackAssessorResult(results: AssessorResult[]):
  | (AssessorResult & {
      id: "AES-FEEDBACK";
      overall: number;
      cefr: string;
    })
  | undefined {
  const result = findAssessorResultById(results, "AES-FEEDBACK");
  if (result && "overall" in result && "cefr" in result) {
    return result as AssessorResult & { id: "AES-FEEDBACK"; overall: number; cefr: string };
  }
  return undefined;
}

/**
 * Type-safe helper to get LanguageTool assessor result from an array of assessor results.
 *
 * @param results - Array of assessor results to search
 * @returns LanguageTool assessor result with guaranteed errors array, or undefined if not found
 */
export function getLanguageToolAssessorResult(
  results: AssessorResult[],
): (AssessorResult & { id: "GEC-LT"; errors: LanguageToolError[] }) | undefined {
  const result = findAssessorResultById(results, "GEC-LT");
  if (result && Array.isArray(result.errors)) {
    return result as AssessorResult & { id: "GEC-LT"; errors: LanguageToolError[] };
  }
  return undefined;
}

/**
 * Type-safe helper to get LLM assessor result from an array of assessor results.
 *
 * @param results - Array of assessor results to search
 * @returns LLM assessor result with guaranteed errors array, or undefined if not found
 */
export function getLLMAssessorResult(
  results: AssessorResult[],
): (AssessorResult & { id: "GEC-LLM"; errors: LanguageToolError[] }) | undefined {
  const result = findAssessorResultById(results, "GEC-LLM");
  if (result && Array.isArray(result.errors)) {
    return result as AssessorResult & { id: "GEC-LLM"; errors: LanguageToolError[] };
  }
  return undefined;
}

/**
 * Type-safe helper to get teacher feedback assessor result from an array of assessor results.
 *
 * @param results - Array of assessor results to search
 * @returns Teacher feedback assessor result with guaranteed meta fields, or undefined if not found
 */
export function getTeacherFeedbackAssessorResult(results: AssessorResult[]):
  | (AssessorResult & {
      id: "TEACHER-FEEDBACK";
      meta: {
        message: string;
        focusArea?: string;
        cluesMessage?: string;
        explanationMessage?: string;
      };
    })
  | undefined {
  const result = findAssessorResultById(results, "TEACHER-FEEDBACK");
  if (result && result.meta && typeof result.meta === "object" && "message" in result.meta) {
    return result as AssessorResult & {
      id: "TEACHER-FEEDBACK";
      meta: {
        message: string;
        focusArea?: string;
        cluesMessage?: string;
        explanationMessage?: string;
      };
    };
  }
  return undefined;
}

/**
 * Type-safe helper to get relevance check assessor result from an array of assessor results.
 *
 * @param results - Array of assessor results to search
 * @returns Relevance check assessor result with guaranteed meta fields, or undefined if not found
 */
export function getRelevanceCheckAssessorResult(results: AssessorResult[]):
  | (AssessorResult & {
      id: "RELEVANCE-CHECK";
      meta: {
        addressesQuestion: boolean;
        similarityScore: number;
        threshold: number;
      };
    })
  | undefined {
  const result = findAssessorResultById(results, "RELEVANCE-CHECK");
  if (
    result &&
    result.meta &&
    typeof result.meta === "object" &&
    "addressesQuestion" in result.meta &&
    "similarityScore" in result.meta
  ) {
    return result as AssessorResult & {
      id: "RELEVANCE-CHECK";
      meta: {
        addressesQuestion: boolean;
        similarityScore: number;
        threshold: number;
      };
    };
  }
  return undefined;
}

/**
 * GEC Seq2Seq edit structure from the GEC-SEQ2SEQ assessor
 */
export interface GECSeq2seqEdit {
  start: number;
  end: number;
  original: string;
  correction: string;
  operation: "insert" | "replace" | "delete";
  category?: string; // grammar, fluency, mechanics, vocabulary
}

/**
 * Type-safe helper to get GEC Seq2Seq assessor result from an array of assessor results.
 *
 * @param results - Array of assessor results to search
 * @returns GEC Seq2Seq assessor result with edits array, or undefined if not found
 */
export function getGECSeq2seqAssessorResult(results: AssessorResult[]):
  | (AssessorResult & {
      id: "GEC-SEQ2SEQ";
      meta: {
        edits: GECSeq2seqEdit[];
        correctedText: string;
      };
    })
  | undefined {
  const result = results.find((r) => r.id === "GEC-SEQ2SEQ");
  if (result && result.meta && Array.isArray((result.meta as Record<string, unknown>).edits)) {
    return result as AssessorResult & {
      id: "GEC-SEQ2SEQ";
      meta: { edits: GECSeq2seqEdit[]; correctedText: string };
    };
  }
  return undefined;
}

/**
 * Convert GEC Seq2Seq edits to LanguageToolError format for display in the heatmap.
 * This allows the precise diff-based edits from the Seq2Seq model to appear as suggestions.
 *
 * @param edits - Array of GEC Seq2Seq edits
 * @returns Array of LanguageToolError objects compatible with the heatmap
 */
export function convertGECEditsToErrors(edits: GECSeq2seqEdit[]): LanguageToolError[] {
  return edits.map((edit) => {
    // Determine error type and message based on operation
    let errorCategory = "GRAMMAR";
    let message = "";
    let errorType = "Grammar correction";

    if (edit.operation === "insert") {
      errorCategory = "GRAMMAR";
      message = `Add "${edit.correction}"`;
      errorType = "Missing word/phrase";
    } else if (edit.operation === "delete") {
      errorCategory = "STYLE";
      message = `Remove "${edit.original}"`;
      errorType = "Unnecessary word/phrase";
    } else {
      // replace
      errorCategory = "GRAMMAR";
      message = `"${edit.original}" → "${edit.correction}"`;
      errorType = "Word/phrase correction";
    }

    return {
      start: edit.start,
      end: edit.end,
      length: edit.end - edit.start,
      category: errorCategory,
      rule_id: `GEC-SEQ2SEQ-${edit.operation.toUpperCase()}`,
      message,
      suggestions: edit.correction ? [edit.correction] : [],
      source: "LLM" as const, // Use LLM source to indicate AI-based
      severity: "warning" as const,
      confidenceScore: 0.85, // Seq2Seq model has high precision
      highConfidence: true,
      mediumConfidence: true,
      errorType,
      explanation: message, // Use the clear message as explanation
    };
  });
}
