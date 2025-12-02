import type { Context } from "hono";
import type { Env } from "../types/env";
import type { CreateSubmissionRequest, AssessmentResults } from "@writeo/shared";
import { validateRequestBodySize } from "../utils/validation";
import { errorResponse, ERROR_CODES } from "../utils/errors";
import { safeLogError, safeLogInfo, sanitizeError } from "../utils/logging";
import { getCombinedFeedbackWithRetry } from "./feedback";
import { mergeAssessmentResults } from "./merge-results";
import type { AIFeedback, TeacherFeedback } from "./feedback";
import { validateSubmissionBody, type ValidationResult } from "./submission/validator";
import { storeSubmissionEntities } from "./submission/storage";
import { buildModalRequest } from "./submission/data-loader";
import { prepareServiceRequests, executeServiceRequests } from "./submission/services";
import { iterateAnswers } from "./submission/utils";
import { processEssayResult } from "./submission/results-essay";
import { processLanguageToolResults } from "./submission/results-languagetool";
import { processLLMResults } from "./submission/results-llm";
import { extractEssayScores } from "./submission/results-scores";
import { processRelevanceResults } from "./submission/results-relevance";
import { buildMetadata, buildResponseHeaders } from "./submission/metadata";
import type { LLMProvider } from "./llm";
import type { ModalRequest, LanguageToolError, RelevanceCheck } from "@writeo/shared";
import {
  MAX_REQUEST_BODY_SIZE,
  RESULTS_TTL_SECONDS,
  FEEDBACK_RETRY_OPTIONS,
} from "../utils/constants";
import { getServices } from "../utils/context";
import { createModalService } from "./modal/factory";

interface FeedbackMaps {
  llmFeedbackByAnswerId: Map<string, AIFeedback>;
  teacherFeedbackByAnswerId: Map<string, TeacherFeedback>;
}

/**
 * Generates combined AI feedback for all answers in parallel.
 * Handles failures gracefully, logging errors but continuing with successful results.
 */
async function generateCombinedFeedback(
  parts: ModalRequest["parts"],
  essayScoresByAnswerId: Map<string, any>,
  ltErrorsByAnswerId: Map<string, any>,
  llmErrorsByAnswerId: Map<string, any>,
  relevanceByAnswerId: Map<string, any>,
  serviceRequests: { llmProvider: LLMProvider; apiKey: string; aiModel: string },
  c: Context<{ Bindings: Env; Variables: { requestId?: string } }>,
): Promise<FeedbackMaps> {
  const llmFeedbackByAnswerId = new Map<string, AIFeedback>();
  const teacherFeedbackByAnswerId = new Map<string, TeacherFeedback>();

  const combinedFeedbackPromises = Array.from(iterateAnswers(parts)).map(async (answer) => {
    const feedback = await getCombinedFeedbackWithRetry(
      {
        llmProvider: serviceRequests.llmProvider,
        apiKey: serviceRequests.apiKey,
        questionText: answer.question_text,
        answerText: answer.answer_text,
        modelName: serviceRequests.aiModel,
        essayScores: essayScoresByAnswerId.get(answer.id),
        languageToolErrors: ltErrorsByAnswerId.get(answer.id),
        llmErrors: llmErrorsByAnswerId.get(answer.id),
        relevanceCheck: relevanceByAnswerId.get(answer.id),
      },
      FEEDBACK_RETRY_OPTIONS,
    );
    return { answerId: answer.id, feedback };
  });

  const results = await Promise.allSettled(combinedFeedbackPromises);
  for (const result of results) {
    if (result.status === "rejected") {
      const errorMsg =
        result.reason instanceof Error ? result.reason.message : String(result.reason);
      safeLogError("Combined feedback generation failed", { error: errorMsg }, c);
    } else {
      const { answerId, feedback } = result.value;
      llmFeedbackByAnswerId.set(answerId, feedback.detailed);
      teacherFeedbackByAnswerId.set(answerId, feedback.teacher);
    }
  }

  return { llmFeedbackByAnswerId, teacherFeedbackByAnswerId };
}

/**
 * Applies computed metadata (word count, error count, scores, timestamp) to assessment results.
 */
function applyMetadata(
  mergedAssessment: AssessmentResults,
  metadata: ReturnType<typeof buildMetadata>,
): void {
  if (!mergedAssessment.meta) {
    mergedAssessment.meta = {};
  }
  mergedAssessment.meta.wordCount = metadata.wordCount;
  mergedAssessment.meta.errorCount = metadata.errorCount;
  if (metadata.overallScore !== undefined) {
    mergedAssessment.meta.overallScore = metadata.overallScore;
  }
  mergedAssessment.meta.timestamp = metadata.timestamp;
}

/**
 * Loads submission data and prepares service requests.
 */
async function loadSubmissionData(
  body: CreateSubmissionRequest,
  validation: ValidationResult,
  storeResults: boolean,
  submissionId: string,
  storage: ReturnType<typeof getServices>["storage"],
  config: ReturnType<typeof getServices>["config"],
  c: Context<{ Bindings: Env; Variables: { requestId?: string } }>,
  timings: Record<string, number>,
): Promise<
  | { modalRequest: ModalRequest; serviceRequests: ReturnType<typeof prepareServiceRequests> }
  | Response
> {
  const autoCreateStartTime = performance.now();
  if (storeResults) {
    const storageResult = await storeSubmissionEntities(storage, validation, submissionId, body, c);
    if (storageResult) {
      return storageResult;
    }
  }
  timings["1c_auto_create_entities"] = performance.now() - autoCreateStartTime;

  const loadDataStartTime = performance.now();
  const modalRequestResult = await buildModalRequest(body, storeResults, storage, submissionId, c);
  if (modalRequestResult instanceof Response) {
    return modalRequestResult;
  }
  const modalRequest = modalRequestResult as ModalRequest;
  timings["4_load_data_from_r2"] = performance.now() - loadDataStartTime;

  const modalService = createModalService(config);
  const serviceRequests = prepareServiceRequests(
    modalRequest.parts,
    config,
    c.env.AI,
    modalService,
  );

  return { modalRequest, serviceRequests };
}

/**
 * Processes all service results into structured data.
 */
async function processServiceResults(
  essayResult: PromiseSettledResult<Response>,
  ltResults: PromiseSettledResult<Response[]>,
  llmResults: PromiseSettledResult<LanguageToolError[][]>,
  relevanceResults: PromiseSettledResult<(RelevanceCheck | null)[]>,
  serviceRequests: ReturnType<typeof prepareServiceRequests>,
  modalRequest: ModalRequest,
  submissionId: string,
  config: ReturnType<typeof getServices>["config"],
  timings: Record<string, number>,
) {
  const processEssayStartTime = performance.now();
  const essayAssessment = await processEssayResult(essayResult, submissionId);
  timings["6_process_essay"] = performance.now() - processEssayStartTime;

  const processLTStartTime = performance.now();
  const { ltErrorsByAnswerId, answerTextsByAnswerId } = await processLanguageToolResults(
    ltResults,
    serviceRequests.ltRequests,
    modalRequest.parts,
    config.features.languageTool.enabled,
  );
  timings["7_process_languagetool"] = performance.now() - processLTStartTime;

  const processLLMStartTime = performance.now();
  const llmErrorsByAnswerId = processLLMResults(
    llmResults,
    serviceRequests.llmAssessmentRequests,
    serviceRequests.llmProvider,
    serviceRequests.aiModel,
  );
  timings["7b_process_ai_assessment"] = performance.now() - processLLMStartTime;

  const essayScoresByAnswerId = extractEssayScores(essayAssessment, modalRequest.parts);

  const processRelevanceStartTime = performance.now();
  const relevanceByAnswerId = processRelevanceResults(
    relevanceResults,
    serviceRequests.relevanceRequests,
  );
  timings["9_process_relevance"] = performance.now() - processRelevanceStartTime;

  return {
    essayAssessment,
    ltErrorsByAnswerId,
    llmErrorsByAnswerId,
    answerTextsByAnswerId,
    essayScoresByAnswerId,
    relevanceByAnswerId,
  };
}

/**
 * Processes a submission request, orchestrating all assessment services.
 *
 * This is the main entry point for essay submission processing. It:
 * 1. Validates the submission request
 * 2. Stores entities (if storeResults is true)
 * 3. Calls assessment services in parallel (Essay Scoring, LanguageTool, Relevance Check)
 * 4. Generates AI feedback with context from all services
 * 5. Merges all results and returns them synchronously
 *
 * @param c - Hono context with environment bindings
 * @returns Response with assessment results or error
 *
 * @example
 * ```typescript
 * app.post("/v1/text/submissions", createSubmissionHandler);
 * ```
 */
export async function processSubmission(
  c: Context<{ Bindings: Env; Variables: { requestId?: string } }>,
  submissionId: string,
  body?: CreateSubmissionRequest,
) {
  const requestStartTime = performance.now();
  const timings: Record<string, number> = {};

  try {
    // Phase 1: Validate and parse request
    let submissionBody: CreateSubmissionRequest;
    let validation: ValidationResult;

    if (body) {
      // Body provided as parameter (from POST)
      const sizeValidation = await validateRequestBodySize(c.req.raw, MAX_REQUEST_BODY_SIZE);
      if (!sizeValidation.valid) {
        return errorResponse(413, sizeValidation.error || "Request body too large (max 1MB)", c);
      }
      const validateStartTime = performance.now();
      const validationResult = validateSubmissionBody(body, c);
      if (validationResult instanceof Response) {
        return validationResult;
      }
      validation = validationResult;
      submissionBody = body;
      timings["1b_validate_submission"] = performance.now() - validateStartTime;
    } else {
      // This should never happen - body is always provided from POST handler
      return errorResponse(
        400,
        "Request body is required",
        c,
        ERROR_CODES.INVALID_SUBMISSION_FORMAT,
      );
    }
    const storeResults = submissionBody.storeResults === true;

    // Phase 2: Initialize services and load data
    const { config, storage } = getServices(c);
    const loadResult = await loadSubmissionData(
      submissionBody,
      validation,
      storeResults,
      submissionId,
      storage,
      config,
      c,
      timings,
    );
    if (loadResult instanceof Response) {
      return loadResult;
    }
    const { modalRequest, serviceRequests } = loadResult;

    // Phase 3: Execute assessment services in parallel
    const { essayResult, ltResults, llmResults, relevanceResults } = await executeServiceRequests(
      modalRequest,
      serviceRequests,
      config,
      timings,
    );

    // Phase 4: Process service results
    const processedResults = await processServiceResults(
      essayResult,
      ltResults,
      llmResults,
      relevanceResults,
      serviceRequests,
      modalRequest,
      submissionId,
      config,
      timings,
    );

    // Phase 5: Generate AI feedback
    const aiFeedbackStartTime = performance.now();
    const { llmFeedbackByAnswerId, teacherFeedbackByAnswerId } = await generateCombinedFeedback(
      modalRequest.parts,
      processedResults.essayScoresByAnswerId,
      processedResults.ltErrorsByAnswerId,
      processedResults.llmErrorsByAnswerId,
      processedResults.relevanceByAnswerId,
      serviceRequests,
      c,
    );
    timings["8_ai_feedback"] = performance.now() - aiFeedbackStartTime;

    // Phase 6: Merge results and apply metadata
    const mergeStartTime = performance.now();
    // LanguageTool is enabled if the map has entries (meaning LT was run)
    const languageToolEnabled = processedResults.ltErrorsByAnswerId.size > 0;
    // LLM assessment is enabled if the map has entries (meaning LLM assessment was run)
    const llmAssessmentEnabled = processedResults.llmErrorsByAnswerId.size > 0;
    const mergedAssessment = mergeAssessmentResults(
      processedResults.essayAssessment,
      processedResults.ltErrorsByAnswerId,
      processedResults.llmErrorsByAnswerId,
      processedResults.answerTextsByAnswerId,
      modalRequest,
      config.features.languageTool.language,
      serviceRequests.aiModel,
      serviceRequests.llmProvider,
      llmFeedbackByAnswerId,
      processedResults.relevanceByAnswerId,
      teacherFeedbackByAnswerId,
      languageToolEnabled,
      llmAssessmentEnabled,
    );
    timings["10_merge_results"] = performance.now() - mergeStartTime;

    const metadataStartTime = performance.now();
    const metadata = buildMetadata(
      processedResults.answerTextsByAnswerId,
      processedResults.ltErrorsByAnswerId,
      processedResults.llmErrorsByAnswerId,
      processedResults.essayAssessment,
    );
    applyMetadata(mergedAssessment, metadata);
    timings["10a_metadata"] = performance.now() - metadataStartTime;

    // Phase 7: Store results if requested
    if (storeResults) {
      const storeResultsStartTime = performance.now();
      await storage.putResults(submissionId, mergedAssessment, RESULTS_TTL_SECONDS);
      timings["11_store_results"] = performance.now() - storeResultsStartTime;
    }

    timings["0_total"] = performance.now() - requestStartTime;
    const requestId = c.get("requestId") as string | undefined;
    const headersObj = buildResponseHeaders(timings, requestId);

    // Add request ID to response body for easier debugging
    const responseBody = {
      ...mergedAssessment,
      ...(requestId && { requestId }),
    };

    // Always return 201 Created for new submissions
    const statusCode = 201;

    // Add Location header for created resource
    {
      const url = new URL(c.req.url);
      headersObj["Location"] = `${url.origin}/v1/text/submissions/${submissionId}`;
    }

    // Log performance metrics with request ID
    safeLogInfo(
      "Request completed",
      {
        submissionId,
        endpoint: c.req.path,
        method: c.req.method,
        statusCode,
        timings,
        totalMs: timings["0_total"]?.toFixed(2),
      },
      c,
    );

    return c.json(responseBody, statusCode, headersObj);
  } catch (error) {
    const sanitized = sanitizeError(error);
    safeLogError(
      "Error creating submission",
      {
        submissionId,
        error: sanitized.message,
        name: sanitized.name,
      },
      c,
    );
    return errorResponse(500, "Internal server error", c, ERROR_CODES.INTERNAL_SERVER_ERROR);
  }
}
