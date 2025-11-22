/**
 * Teacher feedback request handler
 */

import { errorResponse } from "../../utils/errors";
import { validateText, validateRequestBodySize } from "../../utils/validation";
import { isValidUUID } from "@writeo/shared";
import { MAX_ANSWER_TEXT_LENGTH, MAX_REQUEST_BODY_SIZE } from "../../utils/constants";
import { parseLLMProvider, getDefaultModel, getAPIKey } from "../../services/llm";
import { getTeacherFeedback } from "../../services/feedback";
import { StorageService } from "../../services/storage";
import { loadFeedbackDataFromStorage, getCachedTeacherFeedback } from "./storage";
import type { Context } from "hono";
import type { Env } from "../../types/env";

export async function handleTeacherFeedbackRequest(c: Context<{ Bindings: Env }>) {
  const submissionId = c.req.param("submission_id");
  if (!isValidUUID(submissionId)) {
    return errorResponse(400, "Invalid submission_id format", c);
  }

  const sizeValidation = await validateRequestBodySize(c.req.raw, MAX_REQUEST_BODY_SIZE);
  if (!sizeValidation.valid) {
    return errorResponse(413, sizeValidation.error || "Request body too large (max 1MB)", c);
  }

  const body = (await c.req.json()) as {
    answerId: string;
    mode: "clues" | "explanation";
    answerText: string;
    questionText?: string;
    assessmentData?: any;
  };

  if (!body.answerId || !body.mode || !body.answerText) {
    return errorResponse(400, "Missing required fields: answerId, mode, answerText", c);
  }

  if (!isValidUUID(body.answerId)) {
    return errorResponse(400, "Invalid answerId format", c);
  }

  if (body.mode !== "clues" && body.mode !== "explanation") {
    return errorResponse(400, "Mode must be 'clues' or 'explanation'", c);
  }

  const textValidation = validateText(body.answerText, MAX_ANSWER_TEXT_LENGTH);
  if (!textValidation.valid) {
    return errorResponse(
      400,
      `Invalid answerText: ${textValidation.error || "Invalid content"}`,
      c
    );
  }

  const storage = new StorageService(c.env.WRITEO_DATA, c.env.WRITEO_RESULTS);
  const results = await storage.getResults(submissionId);
  const cachedMessage = getCachedTeacherFeedback(results, body.mode);

  if (cachedMessage) {
    const firstPart = results?.results?.parts?.[0];
    const teacherAssessor = firstPart?.answers?.[0]?.["assessor-results"]?.find(
      (a: any) => a.id === "T-TEACHER-FEEDBACK"
    );
    const existingMeta = (teacherAssessor?.meta || {}) as Record<string, any>;
    return {
      cached: true,
      feedback: {
        message: cachedMessage,
        focusArea: existingMeta.focusArea as string | undefined,
      },
    };
  }

  const feedbackData = await loadFeedbackDataFromStorage(
    storage,
    submissionId,
    body.answerId,
    body.questionText,
    body.assessmentData
  );

  if (!feedbackData?.questionText && !body.questionText) {
    return errorResponse(
      400,
      "questionText is required when submission is not stored. Please provide questionText in the request body.",
      c
    );
  }

  const questionText = feedbackData?.questionText || body.questionText || "";
  if (!questionText || questionText.trim().length === 0) {
    return errorResponse(
      400,
      "questionText is required. Please provide questionText in the request body.",
      c
    );
  }

  const llmProvider = parseLLMProvider(c.env.LLM_PROVIDER);
  const defaultModel = getDefaultModel(llmProvider);
  const aiModel = c.env.AI_MODEL || defaultModel;
  const apiKey = getAPIKey(llmProvider, {
    GROQ_API_KEY: c.env.GROQ_API_KEY,
    OPENAI_API_KEY: c.env.OPENAI_API_KEY,
  });

  if (!apiKey) {
    return errorResponse(
      500,
      `API key not found for provider: ${llmProvider}. Please set ${llmProvider === "groq" ? "GROQ_API_KEY" : "OPENAI_API_KEY"}`,
      c
    );
  }

  const teacherFeedback = await getTeacherFeedback(
    llmProvider,
    apiKey,
    questionText,
    body.answerText,
    aiModel,
    body.mode,
    feedbackData?.essayScores || body.assessmentData?.essayScores,
    feedbackData?.ltErrors || body.assessmentData?.ltErrors,
    feedbackData?.llmErrors || body.assessmentData?.llmErrors,
    feedbackData?.relevanceCheck || body.assessmentData?.relevanceCheck
  );

  if (results) {
    await saveTeacherFeedbackToStorage(
      storage,
      submissionId,
      results,
      teacherFeedback,
      body.mode,
      llmProvider,
      aiModel
    );
  }

  return {
    cached: false,
    feedback: {
      message: teacherFeedback.message,
      focusArea: teacherFeedback.focusArea,
    },
  };
}

async function saveTeacherFeedbackToStorage(
  storage: StorageService,
  submissionId: string,
  results: any,
  teacherFeedback: any,
  mode: string,
  llmProvider: string,
  aiModel: string
): Promise<void> {
  const firstPart = results.results?.parts?.[0];
  if (firstPart) {
    let teacherAssessor = firstPart.answers?.[0]?.["assessor-results"]?.find(
      (a: any) => a.id === "T-TEACHER-FEEDBACK"
    );

    if (!teacherAssessor) {
      teacherAssessor = {
        id: "T-TEACHER-FEEDBACK",
        name: "Teacher's Feedback",
        type: "feedback",
        meta: {},
      };
      const firstAnswer = firstPart.answers[0];
      if (!firstAnswer["assessor-results"]) {
        firstAnswer["assessor-results"] = [];
      }
      firstAnswer["assessor-results"].push(teacherAssessor);
    }

    const existingMeta = (teacherAssessor?.meta || {}) as Record<string, any>;
    teacherAssessor.meta = {
      ...existingMeta,
      message: existingMeta.message || teacherFeedback.message,
      focusArea: teacherFeedback.focusArea || existingMeta.focusArea,
      ...(mode === "clues" && { cluesMessage: teacherFeedback.message }),
      ...(mode === "explanation" && { explanationMessage: teacherFeedback.message }),
      engine: llmProvider === "groq" ? "Groq" : "OpenAI",
      model: aiModel,
    };

    await storage.putResults(submissionId, results, 60 * 60 * 24 * 90);
  }
}
