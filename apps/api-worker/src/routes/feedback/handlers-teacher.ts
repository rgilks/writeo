/**
 * Teacher feedback request handler
 */

import { errorResponse } from "../../utils/errors";
import { validateText, validateRequestBodySize } from "../../utils/validation";
import {
  isValidUUID,
  findAssessorResultById,
  type AssessmentResults,
  type AssessorResult,
} from "@writeo/shared";
import { MAX_ANSWER_TEXT_LENGTH, MAX_REQUEST_BODY_SIZE } from "../../utils/constants";
import { parseLLMProvider, getDefaultModel, getAPIKey, type LLMProvider } from "../../services/llm";
import { getTeacherFeedback, type TeacherFeedback } from "../../services/feedback";
import { StorageService } from "../../services/storage";
import {
  loadFeedbackDataFromStorage,
  getCachedTeacherFeedback,
  type FeedbackData,
} from "./storage";
import { TeacherFeedbackRequestSchema, formatZodError } from "./validation";
import type { Context } from "hono";
import type { Env } from "../../types/env";

type FeedbackMode = "clues" | "explanation";

interface TeacherFeedbackResponse {
  cached: boolean;
  feedback: {
    message: string;
    focusArea?: string;
    clues?: string;
  };
}

interface TeacherFeedbackMeta {
  message?: string;
  focusArea?: string;
  cluesMessage?: string;
  explanationMessage?: string;
  engine?: string;
  model?: string;
}

/**
 * Sets up LLM configuration from environment
 */
function setupLLMConfig(env: Env): { provider: LLMProvider; apiKey: string; model: string } | null {
  const provider = parseLLMProvider(env.LLM_PROVIDER);
  const apiKey = getAPIKey(provider, {
    GROQ_API_KEY: env.GROQ_API_KEY,
    OPENAI_API_KEY: env.OPENAI_API_KEY,
  });

  if (!apiKey) {
    return null;
  }

  const model = env.AI_MODEL || getDefaultModel(provider);
  return { provider, apiKey, model };
}

/**
 * Gets teacher assessor result from assessment results
 */
function getTeacherAssessor(results: AssessmentResults | null): AssessorResult | undefined {
  if (!results?.results?.parts?.[0]?.answers?.[0]?.["assessor-results"]) {
    return undefined;
  }
  return findAssessorResultById(
    results.results.parts[0].answers[0]["assessor-results"],
    "T-TEACHER-FEEDBACK",
  );
}

/**
 * Builds response from cached teacher feedback
 */
function buildCachedResponse(
  cachedMessage: string,
  assessor: AssessorResult | undefined,
  mode: FeedbackMode,
): TeacherFeedbackResponse {
  const meta = (assessor?.meta || {}) as TeacherFeedbackMeta;
  return {
    cached: true,
    feedback: {
      message: cachedMessage,
      focusArea: meta.focusArea,
      ...(mode === "clues" && { clues: cachedMessage }),
    },
  };
}

/**
 * Builds response from generated teacher feedback
 */
function buildFeedbackResponse(
  teacherFeedback: TeacherFeedback,
  mode: FeedbackMode,
): TeacherFeedbackResponse {
  return {
    cached: false,
    feedback: {
      message: teacherFeedback.message,
      focusArea: teacherFeedback.focusArea,
      ...(mode === "clues" && { clues: teacherFeedback.message }),
    },
  };
}

/**
 * Validates and extracts question text from request or storage
 */
function validateQuestionText(
  feedbackData: FeedbackData | null,
  providedQuestionText?: string,
): string | null {
  const questionText = feedbackData?.questionText || providedQuestionText || "";
  if (!questionText || questionText.trim().length === 0) {
    return null;
  }
  return questionText;
}

/**
 * Handles teacher feedback requests by validating input, checking cache,
 * fetching missing data, and generating or returning cached feedback.
 *
 * @param c - Hono context with environment bindings
 * @returns Teacher feedback response or error response
 */
export async function handleTeacherFeedbackRequest(
  c: Context<{ Bindings: Env }>,
): Promise<TeacherFeedbackResponse | Response> {
  const submissionId = c.req.param("submission_id");
  if (!isValidUUID(submissionId)) {
    return errorResponse(400, "Invalid submission_id format", c);
  }

  const sizeValidation = await validateRequestBodySize(c.req.raw, MAX_REQUEST_BODY_SIZE);
  if (!sizeValidation.valid) {
    return errorResponse(413, sizeValidation.error || "Request body too large (max 1MB)", c);
  }

  const parsedBody = TeacherFeedbackRequestSchema.safeParse(await c.req.json());
  if (!parsedBody.success) {
    return errorResponse(400, `Invalid request body: ${formatZodError(parsedBody.error)}`, c);
  }
  const body = parsedBody.data;

  const textValidation = validateText(body.answerText, MAX_ANSWER_TEXT_LENGTH);
  if (!textValidation.valid) {
    return errorResponse(
      400,
      `Invalid answerText: ${textValidation.error || "Invalid content"}`,
      c,
    );
  }

  const storage = new StorageService(c.env.WRITEO_DATA, c.env.WRITEO_RESULTS);
  const results = await storage.getResults(submissionId);
  const cachedMessage = getCachedTeacherFeedback(results, body.mode);

  // Return cached feedback if available
  if (cachedMessage) {
    const teacherAssessor = getTeacherAssessor(results);
    return buildCachedResponse(cachedMessage, teacherAssessor, body.mode);
  }

  // Load feedback data from storage
  const feedbackData = await loadFeedbackDataFromStorage(
    storage,
    submissionId,
    body.answerId,
    body.questionText,
    body.assessmentData,
  );

  // Validate question text
  const questionText = validateQuestionText(feedbackData, body.questionText);
  if (!questionText) {
    return errorResponse(
      400,
      "questionText is required when submission is not stored. Please provide questionText in the request body.",
      c,
    );
  }

  // Setup LLM configuration
  const llmConfig = setupLLMConfig(c.env);
  if (!llmConfig) {
    const provider = parseLLMProvider(c.env.LLM_PROVIDER);
    return errorResponse(
      500,
      `API key not found for provider: ${provider}. Please set ${provider === "groq" ? "GROQ_API_KEY" : "OPENAI_API_KEY"}`,
      c,
    );
  }

  // Generate teacher feedback
  const teacherFeedback = await getTeacherFeedback(
    llmConfig.provider,
    llmConfig.apiKey,
    questionText,
    body.answerText,
    llmConfig.model,
    body.mode,
    feedbackData?.essayScores || body.assessmentData?.essayScores,
    feedbackData?.ltErrors || body.assessmentData?.ltErrors,
    feedbackData?.llmErrors || body.assessmentData?.llmErrors,
    feedbackData?.relevanceCheck || body.assessmentData?.relevanceCheck,
  );

  // Save to storage if results exist
  if (results) {
    await saveTeacherFeedbackToStorage(
      storage,
      submissionId,
      results,
      teacherFeedback,
      body.mode,
      llmConfig.provider,
      llmConfig.model,
    );
  }

  return buildFeedbackResponse(teacherFeedback, body.mode);
}

const STORAGE_TTL_SECONDS = 60 * 60 * 24 * 90; // 90 days

/**
 * Saves teacher feedback to storage in the assessment results
 */
async function saveTeacherFeedbackToStorage(
  storage: StorageService,
  submissionId: string,
  results: AssessmentResults,
  teacherFeedback: TeacherFeedback,
  mode: FeedbackMode,
  llmProvider: LLMProvider,
  aiModel: string,
): Promise<void> {
  const firstPart = results.results?.parts?.[0];
  if (!firstPart?.answers?.[0]) {
    return;
  }

  const firstAnswer = firstPart.answers[0];
  if (!firstAnswer["assessor-results"]) {
    firstAnswer["assessor-results"] = [];
  }

  let teacherAssessor = findAssessorResultById(
    firstAnswer["assessor-results"],
    "T-TEACHER-FEEDBACK",
  );

  if (!teacherAssessor) {
    teacherAssessor = {
      id: "T-TEACHER-FEEDBACK",
      name: "Teacher's Feedback",
      type: "feedback",
      meta: {},
    };
    firstAnswer["assessor-results"].push(teacherAssessor);
  }

  const existingMeta = (teacherAssessor.meta || {}) as TeacherFeedbackMeta;
  teacherAssessor.meta = {
    ...existingMeta,
    message: existingMeta.message || teacherFeedback.message,
    focusArea: teacherFeedback.focusArea || existingMeta.focusArea,
    ...(mode === "clues" && { cluesMessage: teacherFeedback.message }),
    ...(mode === "explanation" && { explanationMessage: teacherFeedback.message }),
    engine: llmProvider === "groq" ? "Groq" : "OpenAI",
    model: aiModel,
  };

  await storage.putResults(submissionId, results, STORAGE_TTL_SECONDS);
}
