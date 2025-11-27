import type { Context } from "hono";
import type { Env } from "../types/env";
import type { CreateSubmissionRequest, AssessmentResults } from "@writeo/shared";
import { StorageService } from "./storage";
import { validateRequestBodySize } from "../utils/validation";
import { errorResponse } from "../utils/errors";
import { safeLogError, sanitizeError } from "../utils/logging";
import { getCombinedFeedbackWithRetry } from "./feedback";
import { mergeAssessmentResults } from "./merge-results";
import type { AIFeedback, TeacherFeedback } from "./feedback";
import { validateSubmissionBody } from "./submission/validator";
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
import { buildConfig } from "./config";
import type { LLMProvider } from "./llm";
import type { ModalRequest } from "@writeo/shared";
import { uuidStringSchema, formatZodMessage } from "../utils/zod";
import { MAX_REQUEST_BODY_SIZE } from "../utils/constants";

/** Results storage TTL: 90 days */
const RESULTS_TTL_SECONDS = 60 * 60 * 24 * 90;
/** Retry configuration for combined feedback generation */
const FEEDBACK_RETRY_OPTIONS = { maxAttempts: 3, baseDelayMs: 500 };

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
      safeLogError("Combined feedback generation failed", { error: errorMsg });
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
 * app.put("/text/submissions/:submission_id", processSubmission);
 * ```
 */
export async function processSubmission(c: Context<{ Bindings: Env }>) {
  const submissionIdResult = uuidStringSchema("submission_id").safeParse(
    c.req.param("submission_id"),
  );
  if (!submissionIdResult.success) {
    return errorResponse(
      400,
      formatZodMessage(submissionIdResult.error, "Invalid submission_id format"),
      c,
    );
  }
  const submissionId = submissionIdResult.data;

  const requestStartTime = performance.now();
  const timings: Record<string, number> = {};

  try {
    const sizeValidation = await validateRequestBodySize(c.req.raw, MAX_REQUEST_BODY_SIZE);
    if (!sizeValidation.valid) {
      return errorResponse(413, sizeValidation.error || "Request body too large (max 1MB)", c);
    }

    const parseStartTime = performance.now();
    const body = await c.req.json<CreateSubmissionRequest>();
    timings["1_parse_request"] = performance.now() - parseStartTime;

    const validateStartTime = performance.now();
    const validation = validateSubmissionBody(body, c);
    if (validation instanceof Response) {
      return validation;
    }
    timings["1b_validate_submission"] = performance.now() - validateStartTime;

    const storeResults = body.storeResults === true;
    const autoCreateStartTime = performance.now();
    const config = buildConfig(c.env);
    const storage = new StorageService(config.storage.r2Bucket, config.storage.kvNamespace);

    if (storeResults) {
      const storageResult = await storeSubmissionEntities(
        storage,
        validation,
        submissionId,
        body,
        c,
      );
      if (storageResult) {
        return storageResult;
      }
    }
    timings["1c_auto_create_entities"] = performance.now() - autoCreateStartTime;

    const loadDataStartTime = performance.now();
    const modalRequestResult = await buildModalRequest(body, storeResults, storage, c);
    if (modalRequestResult instanceof Response) {
      return modalRequestResult;
    }
    const modalRequest = modalRequestResult as ModalRequest;
    timings["4_load_data_from_r2"] = performance.now() - loadDataStartTime;

    const serviceRequests = prepareServiceRequests(modalRequest.parts, config, c.env.AI);
    const { essayResult, ltResults, llmResults, relevanceResults } = await executeServiceRequests(
      modalRequest,
      serviceRequests,
      config,
      timings,
    );

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

    const aiFeedbackStartTime = performance.now();
    const { llmFeedbackByAnswerId, teacherFeedbackByAnswerId } = await generateCombinedFeedback(
      modalRequest.parts,
      essayScoresByAnswerId,
      ltErrorsByAnswerId,
      llmErrorsByAnswerId,
      relevanceByAnswerId,
      serviceRequests,
    );
    timings["8_ai_feedback"] = performance.now() - aiFeedbackStartTime;

    const mergeStartTime = performance.now();
    const mergedAssessment = mergeAssessmentResults(
      essayAssessment,
      ltErrorsByAnswerId,
      llmErrorsByAnswerId,
      answerTextsByAnswerId,
      modalRequest,
      config.features.languageTool.language,
      serviceRequests.aiModel,
      serviceRequests.llmProvider,
      llmFeedbackByAnswerId,
      relevanceByAnswerId,
      teacherFeedbackByAnswerId,
    );
    timings["10_merge_results"] = performance.now() - mergeStartTime;

    const metadataStartTime = performance.now();
    const metadata = buildMetadata(
      answerTextsByAnswerId,
      ltErrorsByAnswerId,
      llmErrorsByAnswerId,
      essayAssessment,
    );
    applyMetadata(mergedAssessment, metadata);
    timings["10a_metadata"] = performance.now() - metadataStartTime;

    if (storeResults) {
      const storeResultsStartTime = performance.now();
      await storage.putResults(submissionId, mergedAssessment, RESULTS_TTL_SECONDS);
      timings["11_store_results"] = performance.now() - storeResultsStartTime;
    }

    timings["0_total"] = performance.now() - requestStartTime;

    const headersObj = buildResponseHeaders(timings);

    return c.json(mergedAssessment, 200, headersObj);
  } catch (error) {
    const sanitized = sanitizeError(error);
    safeLogError("Error creating submission", {
      submissionId,
      error: sanitized.message,
      name: sanitized.name,
    });
    return errorResponse(500, "Internal server error", c);
  }
}
