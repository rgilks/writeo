"use server";

/**
 * Server Actions for Writeo API
 * These run on the server and include the API key in requests
 */

import { getApiBase, getApiKey } from "./api-config";

// Generate UUID - use crypto.randomUUID if available, otherwise fallback
const generateUUID = (): string =>
  typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
        const r = (Math.random() * 16) | 0;
        return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
      });

// Extract error message from response with user-friendly formatting
const getErrorMessage = async (response: Response): Promise<string> => {
  try {
    const errorData = await response.json().catch(() => null);
    const errorText =
      errorData?.error || errorData?.message || (await response.text().catch(() => null));

    if (errorText) {
      // Make error messages more user-friendly
      if (errorText.includes("network") || errorText.includes("fetch")) {
        return "Unable to connect to the server. Please check your internet connection and try again.";
      }
      if (errorText.includes("timeout")) {
        return "The request took too long. Please try again.";
      }
      if (response.status === 429) {
        return "Too many requests. Please wait a moment and try again.";
      }
      if (response.status >= 500) {
        return "Server error. Please try again in a moment.";
      }
      if (response.status === 404) {
        return "Resource not found. Please check and try again.";
      }
      if (response.status === 401 || response.status === 403) {
        return "Authentication error. Please refresh the page and try again.";
      }

      return errorText;
    }

    // Fallback to status-based messages
    if (response.status >= 500) {
      return "Server error. Please try again in a moment.";
    }
    if (response.status === 429) {
      return "Too many requests. Please wait a moment and try again.";
    }
    if (response.status === 404) {
      return "Resource not found. Please check and try again.";
    }

    return `HTTP ${response.status}`;
  } catch {
    return `HTTP ${response.status}`;
  }
};

// Make error serializable for Next.js Server Actions
const makeSerializableError = (error: unknown): Error => {
  if (error instanceof Error) {
    return new Error(error.message || "An unexpected error occurred");
  }
  if (typeof error === "string") {
    return new Error(error);
  }
  try {
    const errorStr = JSON.stringify(error);
    return new Error(
      errorStr && errorStr !== "{}"
        ? `Error: ${errorStr.substring(0, 200)}`
        : "An unexpected error occurred"
    );
  } catch {
    return new Error("An unexpected error occurred");
  }
};

// Retry helper with exponential backoff
async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries: number = 3,
  baseDelay: number = 1000
): Promise<T> {
  let lastError: unknown;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      // Don't retry on client errors (4xx) - these are not transient
      if (error instanceof Error && error.message.includes("HTTP 4")) {
        throw error;
      }

      // Don't retry on last attempt
      if (attempt === maxRetries) {
        break;
      }

      // Exponential backoff: 1s, 2s, 4s
      const delay = baseDelay * Math.pow(2, attempt);
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }

  throw lastError;
}

// Create API request helper with retry logic
const apiRequest = async (endpoint: string, method: string, body: any): Promise<Response> => {
  const apiBase = getApiBase();
  const apiKey = getApiKey();

  if (apiKey === "MISSING_API_KEY" || apiBase === "MISSING_API_BASE_URL") {
    throw new Error("Server configuration error: API credentials not set");
  }

  return retryWithBackoff(async () => {
    const response = await fetch(`${apiBase}${endpoint}`, {
      method,
      headers: {
        "Content-Type": "application/json",
        Authorization: `Token ${apiKey}`,
      },
      body: JSON.stringify(body),
    });

    // Retry on server errors (5xx) and network errors
    if (!response.ok && response.status >= 500) {
      throw new Error(`Server error: HTTP ${response.status}. Please try again.`);
    }

    return response;
  });
};

async function createSubmission(
  questionText: string,
  answerText: string,
  storeResults: boolean = false // Default: false (no server storage)
): Promise<{ submissionId: string; results: any }> {
  const submissionId = generateUUID();
  const questionId = generateUUID();
  const answerId = generateUUID();

  // Use inline format - send question and answer text directly with submission
  // The API will auto-create the question and answer entities
  const body: any = {
    submission: [
      {
        part: 1,
        answers: [
          {
            id: answerId,
            "question-number": 1,
            "question-id": questionId,
            "question-text": questionText,
            text: answerText,
          },
        ],
      },
    ],
    template: { name: "generic", version: 1 },
    storeResults: storeResults, // Opt-in server storage
  };

  // Note: API ignores draft tracking fields
  // Draft tracking is handled separately in Server Actions

  // API now processes synchronously and returns results in response
  const response = await apiRequest(`/text/submissions/${submissionId}`, "PUT", body);

  if (!response.ok) {
    throw new Error(`Failed to create submission: ${await getErrorMessage(response)}`);
  }

  // Parse results from response body
  const results = await response.json();

  return { submissionId, results };
}

// Draft tracking helper - stores draft relationships in API metadata
async function linkDraftToParent(
  submissionId: string,
  parentSubmissionId: string,
  results: any
): Promise<void> {
  // Store draft relationship in R2 (via API metadata)
  // This is done by updating the results metadata
  // For now, we'll handle this in the frontend by storing relationships locally
  // In a full implementation, this could use a separate KV namespace or R2 path
}

// Calculate draft number and history from parent submission
async function getDraftInfo(
  parentSubmissionId: string,
  storeResults: boolean = false,
  parentResults?: any
): Promise<{
  draftNumber: number;
  parentSubmissionId: string;
  draftHistory: Array<{
    draftNumber: number;
    timestamp: string;
    wordCount: number;
    errorCount: number;
    overallScore?: number;
  }>;
}> {
  try {
    let parentResultsData = parentResults;

    // If parent results not provided, try to fetch them
    if (!parentResultsData) {
      // When storeResults is false, the parent submission might not be on the server
      // In that case, we'll fall back to assuming this is draft 2
      if (storeResults) {
        parentResultsData = await getSubmissionResults(parentSubmissionId);
      } else {
        // For local mode, if parent results aren't provided, assume this is draft 2
        // The caller should provide parentResults from localStorage when available
        throw new Error("Parent results not available in local mode");
      }
    }

    // Get draft info from parent's metadata
    const parentDraftNumber = (parentResultsData.meta?.draftNumber as number) || 1;
    const rootSubmissionId =
      (parentResultsData.meta?.parentSubmissionId as string) || parentSubmissionId;
    const parentHistory = (parentResultsData.meta?.draftHistory as any[]) || [];

    // Build draft history including parent
    const draftHistory = [
      ...parentHistory,
      {
        draftNumber: parentDraftNumber,
        timestamp: (parentResultsData.meta?.timestamp as string) || new Date().toISOString(),
        wordCount: (parentResultsData.meta?.wordCount as number) || 0,
        errorCount: (parentResultsData.meta?.errorCount as number) || 0,
        overallScore: parentResultsData.meta?.overallScore as number | undefined,
      },
    ];

    return {
      draftNumber: parentDraftNumber + 1,
      parentSubmissionId: rootSubmissionId,
      draftHistory,
    };
  } catch {
    // Parent not found or not ready - assume this is draft 2
    return {
      draftNumber: 2,
      parentSubmissionId,
      draftHistory: [],
    };
  }
}

/**
 * Submit an essay for scoring
 * Uses inline format to send question and answer text directly with submission
 * Wrapped to ensure all errors are properly serialized for Next.js
 *
 * @param questionText - The question/prompt text
 * @param answerText - The essay/answer text
 * @param parentSubmissionId - Optional parent submission ID for draft tracking (handled in Server Actions)
 */
export async function submitEssay(
  questionText: string,
  answerText: string,
  parentSubmissionId?: string,
  storeResults: boolean = false, // Default: false (no server storage)
  parentResults?: any // Optional parent results from localStorage (for local mode)
): Promise<{ submissionId: string; results: any }> {
  try {
    if (!questionText?.trim()) throw new Error("Question text is required");
    if (!answerText?.trim()) throw new Error("Answer text is required");

    // Validate word count (minimum 250 words, maximum 500 words for cost control)
    const wordCount = answerText
      .trim()
      .split(/\s+/)
      .filter((w) => w.length > 0).length;
    const MIN_WORDS = 250;
    const MAX_WORDS = 500;
    if (wordCount < MIN_WORDS) {
      throw new Error(
        `Essay is too short. Please write at least ${MIN_WORDS} words (currently ${wordCount} words).`
      );
    }
    if (wordCount > MAX_WORDS) {
      throw new Error(
        `Essay is too long. Please keep it under ${MAX_WORDS} words (currently ${wordCount} words).`
      );
    }

    // Create submission with inline format - API will auto-create question and answer
    // This reduces from 3 API calls to 1
    const { submissionId, results } = await createSubmission(
      questionText,
      answerText,
      storeResults
    );

    // Handle draft tracking if parent submission provided
    if (parentSubmissionId && results) {
      try {
        const draftInfo = await getDraftInfo(parentSubmissionId, storeResults, parentResults);
        results.meta = {
          ...results.meta,
          draftNumber: draftInfo.draftNumber,
          parentSubmissionId: draftInfo.parentSubmissionId,
          draftHistory: draftInfo.draftHistory,
        };
      } catch (error) {
        console.warn("[submitEssay] Failed to get draft info:", error);
        // Continue without draft tracking
      }
    }

    return { submissionId, results };
  } catch (error) {
    console.error("[submitEssay] Error:", error instanceof Error ? error.message : String(error));
    throw makeSerializableError(error);
  }
}

/**
 * Get submission results with draft tracking info
 * Enhances API results with draft tracking metadata
 */
export async function getSubmissionResultsWithDraftTracking(
  submissionId: string,
  parentSubmissionId?: string,
  storeResults: boolean = false,
  parentResults?: any
): Promise<any> {
  const results = await getSubmissionResults(submissionId);

  // If this is a draft, enhance results with draft tracking info
  if (parentSubmissionId && results.status === "success") {
    try {
      const draftInfo = await getDraftInfo(parentSubmissionId, storeResults, parentResults);

      // Add draft tracking to metadata
      if (!results.meta) {
        results.meta = {};
      }
      results.meta.draftNumber = draftInfo.draftNumber;
      results.meta.parentSubmissionId = draftInfo.parentSubmissionId;
      results.meta.draftHistory = draftInfo.draftHistory;

      // Add current draft to history
      if (results.meta.wordCount !== undefined && results.meta.errorCount !== undefined) {
        const currentDraft = {
          draftNumber: draftInfo.draftNumber,
          timestamp: results.meta.timestamp || new Date().toISOString(),
          wordCount: results.meta.wordCount as number,
          errorCount: results.meta.errorCount as number,
          overallScore: results.meta.overallScore as number | undefined,
        };
        results.meta.draftHistory = [...draftInfo.draftHistory, currentDraft];
      }
    } catch (error) {
      // If parent lookup fails, just return results without draft info
      console.warn("Failed to get draft info:", error);
    }
  } else if (results.status === "success" && results.meta) {
    // First draft - initialize draft tracking
    results.meta.draftNumber = 1;
    results.meta.draftHistory = [
      {
        draftNumber: 1,
        timestamp: results.meta.timestamp || new Date().toISOString(),
        wordCount: results.meta.wordCount || 0,
        errorCount: results.meta.errorCount || 0,
        overallScore: results.meta.overallScore,
      },
    ];
  }

  return results;
}

export async function getSubmissionResults(submissionId: string) {
  const apiBase = getApiBase();
  const apiKey = getApiKey();

  if (apiKey === "MISSING_API_KEY" || apiBase === "MISSING_API_BASE_URL") {
    throw new Error("Server configuration error: API credentials not set");
  }

  const response = await retryWithBackoff(async () => {
    const res = await fetch(`${apiBase}/text/submissions/${submissionId}`, {
      headers: { Authorization: `Token ${apiKey}` },
    });

    // Don't retry on 404 - submission not found on server (may be in localStorage)
    // Return the response so we can handle it below
    if (res.status === 404) {
      return res;
    }

    if (!res.ok && res.status >= 500) {
      throw new Error(`Server error: HTTP ${res.status}. Please try again.`);
    }

    return res;
  });

  if (!response.ok) {
    // For 404, throw a specific error that the frontend can handle gracefully
    if (response.status === 404) {
      const error = new Error("Submission not found on server");
      (error as any).status = 404;
      throw error;
    }
    const errorText = await response.text();
    let errorMessage = `Failed to fetch results: HTTP ${response.status}`;
    try {
      const errorJson = JSON.parse(errorText);
      errorMessage = errorJson.error || errorJson.message || errorMessage;
    } catch {
      errorMessage = errorText || errorMessage;
    }
    throw new Error(errorMessage);
  }

  const data = await response.json();

  // The GET endpoint returns AssessmentResults directly: {status, results: {parts: [...]}, template, meta}
  // The PUT endpoint returns {status: "success", results: AssessmentResults}
  // Check if this is a wrapped PUT response (has nested results.status) and extract the inner AssessmentResults
  if (
    data.status === "success" &&
    data.results &&
    typeof data.results === "object" &&
    "status" in data.results &&
    "results" in data.results
  ) {
    // This is a wrapped PUT response, extract the inner AssessmentResults
    return data.results;
  }

  // Otherwise return as-is (should be AssessmentResults from GET endpoint)
  return data;
}

export async function pollSubmissionResults(
  submissionId: string,
  maxAttempts: number = 20,
  initialIntervalMs: number = 1000,
  parentSubmissionId?: string
): Promise<any> {
  for (let attempts = 0; attempts < maxAttempts; attempts++) {
    try {
      const data = await getSubmissionResults(submissionId);
      if (data.status !== "pending") {
        // Enhance with draft tracking if this is a draft
        if (parentSubmissionId) {
          return await getSubmissionResultsWithDraftTracking(submissionId, parentSubmissionId);
        }
        return data;
      }
    } catch (error) {
      if (attempts >= maxAttempts - 1) {
        throw makeSerializableError(error);
      }
    }

    const intervalMs = Math.min(initialIntervalMs * Math.pow(2, attempts), 10000);
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }

  throw new Error("Request timed out. Please try again.");
}

/**
 * Get Teacher feedback (clues or explanation) for an answer
 * This is a frontend-only feature, so it's implemented as a Server Action
 * rather than part of the API
 */
export async function getTeacherFeedback(
  submissionId: string,
  answerId: string,
  mode: "clues" | "explanation",
  answerText: string,
  questionText?: string,
  assessmentData?: {
    essayScores?: {
      overall?: number;
      dimensions?: {
        TA?: number;
        CC?: number;
        Vocab?: number;
        Grammar?: number;
        Overall?: number;
      };
    };
    ltErrors?: any[];
    llmErrors?: any[];
    relevanceCheck?: {
      addressesQuestion: boolean;
      score: number;
      threshold: number;
    };
  }
): Promise<{ message: string; focusArea?: string }> {
  try {
    if (!submissionId || !answerId || !mode || !answerText) {
      throw new Error("Missing required fields: submissionId, answerId, mode, answerText");
    }

    if (mode !== "clues" && mode !== "explanation") {
      throw new Error("Mode must be 'clues' or 'explanation'");
    }

    const apiBase = getApiBase();
    const apiKey = getApiKey();

    if (apiKey === "MISSING_API_KEY" || apiBase === "MISSING_API_BASE_URL") {
      throw new Error("Server configuration error: API credentials not set");
    }

    const requestBody: any = {
      answerId,
      mode,
      answerText,
    };

    // Add optional fields if provided (allows endpoints to work without storage)
    if (questionText) {
      requestBody.questionText = questionText;
    }
    if (assessmentData) {
      requestBody.assessmentData = assessmentData;
    }

    const response = await retryWithBackoff(async () => {
      const res = await fetch(`${apiBase}/text/submissions/${submissionId}/teacher-feedback`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Token ${apiKey}`,
        },
        body: JSON.stringify(requestBody),
      });

      if (!res.ok && res.status >= 500) {
        throw new Error(`Server error: HTTP ${res.status}. Please try again.`);
      }

      return res;
    });

    if (!response.ok) {
      throw new Error(`Failed to get feedback: ${await getErrorMessage(response)}`);
    }

    const data = await response.json();
    const feedbackResult = {
      message: data.message || "",
      focusArea: data.focusArea,
    };

    // Store teacher feedback locally in localStorage
    if (typeof window !== "undefined") {
      try {
        const storedResults = localStorage.getItem(`results_${submissionId}`);
        if (storedResults) {
          const results = JSON.parse(storedResults) as any;

          // Find or create the teacher feedback assessor in the results structure
          if (results.results?.parts?.[0]?.answers?.[0]) {
            const firstAnswer = results.results.parts[0].answers[0];
            if (!firstAnswer["assessor-results"]) {
              firstAnswer["assessor-results"] = [];
            }

            // Find existing teacher feedback assessor
            let teacherAssessor = firstAnswer["assessor-results"].find(
              (a: any) => a.id === "T-TEACHER-FEEDBACK"
            );

            if (!teacherAssessor) {
              // Create new teacher feedback assessor
              teacherAssessor = {
                id: "T-TEACHER-FEEDBACK",
                name: "Teacher's Feedback",
                type: "feedback",
                meta: {},
              };
              firstAnswer["assessor-results"].push(teacherAssessor);
            }

            // Update meta with feedback data
            const existingMeta = (teacherAssessor.meta || {}) as Record<string, any>;
            teacherAssessor.meta = {
              ...existingMeta,
              message: existingMeta.message || feedbackResult.message,
              focusArea: feedbackResult.focusArea || existingMeta.focusArea,
              // Store mode-specific messages
              ...(mode === "clues" && { cluesMessage: feedbackResult.message }),
              ...(mode === "explanation" && { explanationMessage: feedbackResult.message }),
            };

            // Save updated results back to localStorage
            localStorage.setItem(`results_${submissionId}`, JSON.stringify(results));
          }
        }
      } catch (storageError) {
        // Don't fail the request if localStorage update fails
        console.warn("[getTeacherFeedback] Failed to store feedback locally:", storageError);
      }
    }

    return feedbackResult;
  } catch (error) {
    console.error("[getTeacherFeedback] Error:", error);
    throw makeSerializableError(error);
  }
}

/**
 * Stream AI feedback generation using Server-Sent Events
 * Returns a Response object with SSE stream that can be consumed by the frontend
 * This is a frontend-only feature, so it's implemented as a Server Action
 */
export async function streamAIFeedback(
  submissionId: string,
  answerId: string,
  answerText: string,
  questionText?: string,
  assessmentData?: {
    essayScores?: {
      overall?: number;
      dimensions?: {
        TA?: number;
        CC?: number;
        Vocab?: number;
        Grammar?: number;
        Overall?: number;
      };
    };
    ltErrors?: any[];
  }
): Promise<Response> {
  try {
    if (!submissionId || !answerId || !answerText) {
      throw new Error("Missing required fields: submissionId, answerId, answerText");
    }

    const apiBase = getApiBase();
    const apiKey = getApiKey();

    if (apiKey === "MISSING_API_KEY" || apiBase === "MISSING_API_BASE_URL") {
      throw new Error("Server configuration error: API credentials not set");
    }

    const requestBody: any = {
      answerId,
      answerText,
    };

    // Add optional fields if provided (allows endpoints to work without storage)
    if (questionText) {
      requestBody.questionText = questionText;
    }
    if (assessmentData) {
      requestBody.assessmentData = assessmentData;
    }

    // Call the API worker's streaming endpoint
    const response = await fetch(`${apiBase}/text/submissions/${submissionId}/ai-feedback/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Token ${apiKey}`,
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      throw new Error(`Failed to stream feedback: ${await getErrorMessage(response)}`);
    }

    // Return the SSE stream response directly
    // Server Actions can return Response objects, which Next.js will handle correctly
    return response;
  } catch (error) {
    console.error("[streamAIFeedback] Error:", error);
    // Return an error response as SSE
    const errorMessage = error instanceof Error ? error.message : String(error);
    return new Response(`data: ${JSON.stringify({ type: "error", message: errorMessage })}\n\n`, {
      status: 500,
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  }
}
