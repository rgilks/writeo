import type { Context } from "hono";
import type { Env, ExecutionContext } from "../types/env";
import type {
  CreateSubmissionRequest,
  CreateQuestionRequest,
  CreateAnswerRequest,
  ModalRequest,
  AssessmentResults,
  LanguageToolError,
  AssessorResult,
} from "@writeo/shared";
import { isValidUUID } from "@writeo/shared";
import { StorageService } from "./storage";
import { validateText, validateRequestBodySize } from "../utils/validation";
import { errorResponse } from "../utils/errors";
import { fetchWithTimeout } from "../utils/fetch-with-timeout";
import { safeLogError, safeLogWarn, sanitizeError } from "../utils/logging";
import { transformLanguageToolResponse } from "../utils/text-processing";
import { getLLMAssessment } from "./ai-assessment";
import { getCombinedFeedback, getCombinedFeedbackWithRetry } from "./feedback";
import { checkAnswerRelevance, type RelevanceCheck } from "./relevance";
import { mergeAssessmentResults } from "./merge-results";
import type { AIFeedback, TeacherFeedback } from "./feedback";

export async function processSubmission(c: Context<{ Bindings: Env }>) {
  const submissionId = c.req.param("submission_id");
  if (!isValidUUID(submissionId)) {
    return errorResponse(400, "Invalid submission_id format", c);
  }

  const requestStartTime = performance.now();
  const timings: Record<string, number> = {};

  try {
    const sizeValidation = await validateRequestBodySize(c.req.raw, 1024 * 1024);
    if (!sizeValidation.valid) {
      return errorResponse(413, sizeValidation.error || "Request body too large (max 1MB)", c);
    }

    const parseStartTime = performance.now();
    const body = await c.req.json<CreateSubmissionRequest>();
    timings["1_parse_request"] = performance.now() - parseStartTime;

    if (!body.submission || !Array.isArray(body.submission)) {
      return errorResponse(400, "Missing or invalid 'submission' array", c);
    }

    if (!body.template || !body.template.name || typeof body.template.version !== "number") {
      return errorResponse(400, "Missing or invalid 'template' object", c);
    }

    // Always process synchronously - wait for results and return them
    const asyncMode = false;

    const validateStartTime = performance.now();
    const answerIds: string[] = [];
    const questionsToCreate: Array<{ id: string; text: string }> = [];
    const answersToCreate: Array<{
      id: string;
      questionId: string;
      answerText: string;
    }> = [];

    for (const part of body.submission) {
      if (!part.part || !Array.isArray(part.answers)) {
        return errorResponse(400, "Invalid submission part structure");
      }
      for (const answer of part.answers) {
        if (!answer.id || !isValidUUID(answer.id)) {
          return errorResponse(400, `Invalid answer id: ${answer.id}`);
        }
        answerIds.push(answer.id);

        // Answers must always be sent inline (no reference format)
        const answerText = answer["text"];
        if (!answerText) {
          return errorResponse(
            400,
            `Answer text is required. Answers must be sent inline with the submission.`,
            c
          );
        }

        const questionText = answer["question-text"];
        if (!answer["question-id"]) {
          return errorResponse(400, `question-id is required for each answer`, c);
        }

        const questionId = answer["question-id"];
        if (!isValidUUID(questionId)) {
          return errorResponse(400, `Invalid question-id format: ${questionId}`, c);
        }

        const answerTextValidation = validateText(answerText, 50000);
        if (!answerTextValidation.valid) {
          return errorResponse(
            400,
            `Invalid answer text: ${answerTextValidation.error || "Invalid content"}`,
            c
          );
        }

        // Question can be sent inline or referenced by ID
        // If question-text is provided, we'll create/update the question
        // If not, the question must already exist
        if (questionText) {
          const questionTextValidation = validateText(questionText, 10000);
          if (!questionTextValidation.valid) {
            return errorResponse(
              400,
              `Invalid question-text: ${questionTextValidation.error || "Invalid content"}`,
              c
            );
          }
          questionsToCreate.push({ id: questionId, text: questionText });
        }

        answersToCreate.push({
          id: answer.id,
          questionId: questionId,
          answerText: answerText,
        });
      }
    }
    timings["1b_validate_submission"] = performance.now() - validateStartTime;

    // Check if user opted in to server storage (default: false - no storage)
    const storeResults = body.storeResults === true;

    const autoCreateStartTime = performance.now();
    const storage = new StorageService(c.env.WRITEO_DATA, c.env.WRITEO_RESULTS);

    // Only store questions/answers if user opted in to server storage
    if (storeResults) {
      const questionPromises = questionsToCreate.map(async (question) => {
        const existing = await storage.getQuestion(question.id);
        if (!existing) {
          await storage.putQuestion(question.id, { text: question.text });
        } else {
          if (existing.text !== question.text) {
            throw new Error(`Question ${question.id} already exists with different content`);
          }
        }
      });

      const questionResults = await Promise.allSettled(questionPromises);
      for (const result of questionResults) {
        if (result.status === "rejected") {
          return errorResponse(
            409,
            result.reason instanceof Error ? result.reason.message : String(result.reason)
          );
        }
      }

      const answersWithoutQuestionText = answersToCreate.filter((answer) => {
        return !questionsToCreate.some((q) => q.id === answer.questionId);
      });

      for (const answer of answersWithoutQuestionText) {
        const existingQuestion = await storage.getQuestion(answer.questionId);
        if (!existingQuestion) {
          return errorResponse(
            400,
            `question-text is required when question ${answer.questionId} does not exist`
          );
        }
      }

      // Create a set of question IDs that were just created for quick lookup
      const createdQuestionIds = new Set(questionsToCreate.map((q) => q.id));

      const answerPromises = answersToCreate.map(async (answer) => {
        const existing = await storage.getAnswer(answer.id);
        if (!existing) {
          // If question was just created inline, we know it exists
          // Otherwise, check storage for referenced questions
          if (!createdQuestionIds.has(answer.questionId)) {
            const questionExists = await storage.getQuestion(answer.questionId);
            if (!questionExists) {
              throw new Error(`Referenced question does not exist: ${answer.questionId}`);
            }
          }
          await storage.putAnswer(answer.id, {
            "question-id": answer.questionId,
            text: answer.answerText,
          });
        } else {
          if (
            existing["question-id"] !== answer.questionId ||
            existing.text !== answer.answerText
          ) {
            throw new Error(`Answer ${answer.id} already exists with different content`);
          }
        }
      });

      const answerResults = await Promise.allSettled(answerPromises);
      for (const result of answerResults) {
        if (result.status === "rejected") {
          return errorResponse(
            409,
            result.reason instanceof Error ? result.reason.message : String(result.reason)
          );
        }
      }

      // Check/store submission if user opted in
      const checkExistingStartTime = performance.now();
      const existing = await storage.getSubmission(submissionId);
      if (existing) {
        if (JSON.stringify(existing) === JSON.stringify(body)) {
          return new Response(null, { status: 204 });
        } else {
          return errorResponse(409, "Submission already exists with different content", c);
        }
      }
      timings["2_check_existing"] = performance.now() - checkExistingStartTime;

      const storeSubmissionStartTime = performance.now();
      await storage.putSubmission(submissionId, body);
      timings["3_store_submission"] = performance.now() - storeSubmissionStartTime;
    }
    timings["1c_auto_create_entities"] = performance.now() - autoCreateStartTime;

    const loadDataStartTime = performance.now();
    const modalParts: ModalRequest["parts"] = [];

    // Build Modal request directly from inline data (answers are always sent inline)
    for (const part of body.submission) {
      const modalAnswers: ModalRequest["parts"][0]["answers"] = [];
      for (const answerRef of part.answers) {
        // Get question text - either from inline data or from storage (if stored)
        let questionText: string | undefined = answerRef["question-text"];
        if (!questionText && storeResults) {
          // Try to load from storage if not inline
          const question = await storage.getQuestion(answerRef["question-id"] || "");
          questionText = question?.text;
        }
        if (!questionText) {
          return errorResponse(
            400,
            `question-text is required for answer ${answerRef.id} (either inline or must exist in storage)`
          );
        }

        // Answer text is always sent inline
        const answerText = answerRef.text;
        if (!answerText) {
          return errorResponse(400, `Answer text is required for answer ${answerRef.id}`);
        }

        modalAnswers.push({
          id: answerRef.id,
          question_id: answerRef["question-id"] || "",
          question_text: questionText,
          answer_text: answerText,
        });
      }
      modalParts.push({ part: part.part, answers: modalAnswers });
    }
    timings["4_load_data_from_r2"] = performance.now() - loadDataStartTime;

    const modalRequest: ModalRequest = {
      submission_id: submissionId,
      template: body.template,
      parts: modalParts,
    };

    if (!c.env.MODAL_GRADE_URL) {
      return errorResponse(500, "MODAL_GRADE_URL is not configured", c);
    }

    const language = c.env.LT_LANGUAGE ?? "en-GB";

    const ltRequests: Array<{ answerId: string; request: Promise<Response> }> = [];
    if (c.env.MODAL_LT_URL) {
      for (const part of modalParts) {
        for (const answer of part.answers) {
          ltRequests.push({
            answerId: answer.id,
            request: fetchWithTimeout(`${c.env.MODAL_LT_URL}/check`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                language,
                text: answer.answer_text,
                answer_id: answer.id,
              }),
              timeout: 30000, // 30 seconds for LanguageTool
            }),
          });
        }
      }
    }

    const relevanceRequests: Array<{
      answerId: string;
      questionText: string;
      answerText: string;
      request: Promise<RelevanceCheck | null>;
    }> = [];
    for (const part of modalParts) {
      for (const answer of part.answers) {
        relevanceRequests.push({
          answerId: answer.id,
          questionText: answer.question_text,
          answerText: answer.answer_text,
          request: checkAnswerRelevance(c.env.AI, answer.question_text, answer.answer_text, 0.5),
        });
      }
    }

    const aiModel = c.env.AI_MODEL || "llama-3.3-70b-versatile";
    const llmAssessmentRequests: Array<{
      answerId: string;
      questionText: string;
      answerText: string;
      request: Promise<LanguageToolError[]>;
    }> = [];
    for (const part of modalParts) {
      for (const answer of part.answers) {
        llmAssessmentRequests.push({
          answerId: answer.id,
          questionText: answer.question_text,
          answerText: answer.answer_text,
          request: getLLMAssessment(
            c.env.GROQ_API_KEY,
            answer.question_text,
            answer.answer_text,
            aiModel
          ),
        });
      }
    }

    const essayStartTime = performance.now();
    const [essayResult, ltResults, llmResults, relevanceResults] = await Promise.allSettled([
      (async () => {
        const start = performance.now();
        const result = await fetchWithTimeout(`${c.env.MODAL_GRADE_URL}/grade`, {
          timeout: 60000, // 60 seconds for essay scoring (longer operation)
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(modalRequest),
        });
        timings["5a_essay_fetch"] = performance.now() - start;
        return result;
      })(),
      (async () => {
        const start = performance.now();
        const result =
          ltRequests.length > 0 ? await Promise.all(ltRequests.map((r) => r.request)) : [];
        timings["5b_languagetool_fetch"] = performance.now() - start;
        return result;
      })(),
      (async () => {
        const start = performance.now();
        const result =
          llmAssessmentRequests.length > 0
            ? await Promise.all(llmAssessmentRequests.map((r) => r.request))
            : [];
        timings["5d_ai_assessment_fetch"] = performance.now() - start;
        return result;
      })(),
      (async () => {
        const start = performance.now();
        const result =
          relevanceRequests.length > 0
            ? await Promise.all(relevanceRequests.map((r) => r.request))
            : [];
        timings["5c_relevance_fetch"] = performance.now() - start;
        return result;
      })(),
    ]);

    timings["5_parallel_services_total"] = performance.now() - essayStartTime;

    const processEssayStartTime = performance.now();
    let essayAssessment: AssessmentResults | null = null;
    if (essayResult.status === "fulfilled") {
      const response = essayResult.value;
      if (response.ok) {
        try {
          essayAssessment = await response.json<AssessmentResults>();
        } catch (parseError) {
          const errorMsg = parseError instanceof Error ? parseError.message : String(parseError);
          safeLogError("Failed to parse essay assessment response", {
            error: errorMsg,
            status: response.status,
            statusText: response.statusText,
          });
          // Continue without essay assessment - other assessors can still work
        }
      } else {
        // Essay grading service returned an error
        const errorText = await response.text().catch(() => response.statusText);
        safeLogError("Essay grading service failed", {
          status: response.status,
          statusText: response.statusText,
          error: errorText.substring(0, 500), // Limit error text length
        });
        // Don't throw - continue without essay assessment
        // The submission can still be processed with other assessors (LanguageTool, LLM, etc.)
      }
    } else {
      // Essay grading request was rejected (timeout, network error, etc.)
      const errorMsg =
        essayResult.reason instanceof Error
          ? essayResult.reason.message
          : String(essayResult.reason);
      safeLogError("Essay grading request failed", {
        error: errorMsg,
        submissionId,
      });
      // Don't throw - continue without essay assessment
    }
    timings["6_process_essay"] = performance.now() - processEssayStartTime;

    const processLTStartTime = performance.now();
    const ltErrorsByAnswerId = new Map<string, LanguageToolError[]>();
    const answerTextsByAnswerId = new Map<string, string>();

    if (c.env.MODAL_LT_URL) {
      if (ltResults.status !== "fulfilled" || !Array.isArray(ltResults.value)) {
        const errorMsg =
          ltResults.status === "rejected"
            ? ltResults.reason instanceof Error
              ? ltResults.reason.message
              : String(ltResults.reason)
            : "Invalid response";
        safeLogError("LanguageTool service failed", { error: errorMsg });
        // Don't throw - continue without LanguageTool errors
        // throw new Error(`LanguageTool service failed: ${errorMsg}`);
      } else {
        const ltResponses = await Promise.allSettled(
          ltResults.value.map(async (res) => {
            if (!res.ok) {
              const errorText = await res.text().catch(() => res.statusText);
              throw new Error(
                `LanguageTool request failed: ${res.status} ${res.statusText} - ${errorText}`
              );
            }
            return await res.json();
          })
        );

        // Process successful responses, log failures but continue
        for (let i = 0; i < ltResponses.length; i++) {
          const result = ltResponses[i];
          if (result.status === "fulfilled") {
            const answerId = ltRequests[i]?.answerId;
            if (answerId) {
              for (const part of modalParts) {
                const answer = part.answers.find((a) => a.id === answerId);
                if (answer) {
                  answerTextsByAnswerId.set(answerId, answer.answer_text);
                  break;
                }
              }
              let fullText = "";
              for (const part of modalParts) {
                const answer = part.answers.find((a) => a.id === answerId);
                if (answer) {
                  fullText = answer.answer_text;
                  break;
                }
              }
              const errors = transformLanguageToolResponse(result.value, fullText);
              ltErrorsByAnswerId.set(answerId, errors);
            }
          } else {
            safeLogError(`LanguageTool request ${i} failed`, { reason: result.reason });
          }
        }
      }
    } else {
      // No LanguageTool URL - populate answer texts for other services
      for (const part of modalParts) {
        for (const answer of part.answers) {
          answerTextsByAnswerId.set(answer.id, answer.answer_text);
        }
      }
    }
    timings["7_process_languagetool"] = performance.now() - processLTStartTime;

    const processLLMStartTime = performance.now();
    const llmErrorsByAnswerId = new Map<string, LanguageToolError[]>();
    if (llmResults.status === "fulfilled" && Array.isArray(llmResults.value)) {
      const llmResponses = llmResults.value;
      for (let i = 0; i < llmAssessmentRequests.length; i++) {
        const { answerId } = llmAssessmentRequests[i];
        const llmErrors = llmResponses[i] || [];
        llmErrorsByAnswerId.set(answerId, llmErrors);
      }
    }
    timings["7b_process_ai_assessment"] = performance.now() - processLLMStartTime;

    const essayScoresByAnswerId = new Map<
      string,
      {
        overall?: number;
        dimensions?: {
          TA?: number;
          CC?: number;
          Vocab?: number;
          Grammar?: number;
          Overall?: number;
        };
        label?: string;
      }
    >();

    if (essayAssessment?.results?.parts) {
      for (const essayPart of essayAssessment.results.parts) {
        const matchingPart = modalParts.find((p) => p.part === essayPart.part);
        if (matchingPart) {
          // Get assessor-results from first answer
          const essayAssessor = essayPart.answers?.[0]?.["assessor-results"]?.find(
            (a: AssessorResult) => a.id === "T-AES-ESSAY"
          );
          if (essayAssessor) {
            for (const answer of matchingPart.answers) {
              essayScoresByAnswerId.set(answer.id, {
                overall: essayAssessor.overall,
                dimensions: essayAssessor.dimensions,
                label: essayAssessor.label,
              });
            }
          }
        }
      }
    }

    // Process relevance results early so they're available for teacher feedback
    const processRelevanceStartTime = performance.now();
    const relevanceByAnswerId = new Map<string, RelevanceCheck>();
    if (relevanceResults.status === "fulfilled" && Array.isArray(relevanceResults.value)) {
      for (let i = 0; i < relevanceRequests.length; i++) {
        const { answerId } = relevanceRequests[i];
        const relevance = relevanceResults.value[i];
        if (relevance) {
          relevanceByAnswerId.set(answerId, relevance);
        }
      }
    }
    timings["9_process_relevance"] = performance.now() - processRelevanceStartTime;

    let llmFeedbackByAnswerId = new Map<string, AIFeedback>();
    let teacherFeedbackByAnswerId = new Map<string, TeacherFeedback>();

    if (!asyncMode) {
      const aiFeedbackStartTime = performance.now();
      const combinedFeedbackPromises: Array<Promise<{ answerId: string; feedback: any }>> = [];

      for (const part of modalParts) {
        for (const answer of part.answers) {
          const essayScores = essayScoresByAnswerId.get(answer.id);
          const ltErrors = ltErrorsByAnswerId.get(answer.id);
          const llmErrors = llmErrorsByAnswerId.get(answer.id);
          const relevanceCheck = relevanceByAnswerId.get(answer.id);

          combinedFeedbackPromises.push(
            (async () => {
              const feedback = await getCombinedFeedbackWithRetry(
                {
                  groqApiKey: c.env.GROQ_API_KEY,
                  questionText: answer.question_text,
                  answerText: answer.answer_text,
                  modelName: aiModel,
                  essayScores,
                  languageToolErrors: ltErrors,
                  llmErrors,
                  relevanceCheck,
                },
                { maxAttempts: 3, baseDelayMs: 500 }
              );
              return { answerId: answer.id, feedback };
            })()
          );
        }
      }

      const combinedFeedbackResults = await Promise.allSettled(combinedFeedbackPromises);
      for (const result of combinedFeedbackResults) {
        if (result.status === "rejected") {
          // Log error but continue - feedback is helpful but not critical
          const errorMsg =
            result.reason instanceof Error ? result.reason.message : String(result.reason);
          safeLogError("Combined feedback generation failed", { error: errorMsg });
          // Don't throw - continue without feedback for this answer
          // The answer will still be processed with other assessors
        } else {
          const { answerId, feedback } = result.value;
          llmFeedbackByAnswerId.set(answerId, feedback.detailed);
          teacherFeedbackByAnswerId.set(answerId, feedback.teacher);
        }
      }

      timings["8_ai_feedback"] = performance.now() - aiFeedbackStartTime;
    }

    // Validate relevance results (already processed above, but check for errors)
    if (relevanceResults.status !== "fulfilled" || !Array.isArray(relevanceResults.value)) {
      // Log warning but don't fail - relevance is helpful but not critical
      safeLogWarn("Relevance check failed", {
        reason:
          relevanceResults.status === "rejected" ? relevanceResults.reason : "Invalid response",
      });
    }

    const mergeStartTime = performance.now();
    const mergedAssessment = mergeAssessmentResults(
      essayAssessment,
      ltErrorsByAnswerId,
      llmErrorsByAnswerId,
      answerTextsByAnswerId,
      modalRequest,
      language,
      aiModel,
      llmFeedbackByAnswerId,
      relevanceByAnswerId,
      teacherFeedbackByAnswerId
    );
    timings["10_merge_results"] = performance.now() - mergeStartTime;

    const metadataStartTime = performance.now();
    let totalWordCount = 0;
    let totalErrorCount = 0;
    let overallScore: number | undefined;

    for (const text of answerTextsByAnswerId.values()) {
      totalWordCount += text
        .trim()
        .split(/\s+/)
        .filter((w) => w.length > 0).length;
    }

    for (const errors of ltErrorsByAnswerId.values()) {
      totalErrorCount += errors.length;
    }
    for (const errors of llmErrorsByAnswerId.values()) {
      totalErrorCount += errors.length;
    }

    // Get overall score from first answer
    const firstPart = essayAssessment?.results?.parts?.[0];
    if (firstPart?.answers && firstPart.answers.length > 0) {
      const essayAssessor = firstPart.answers[0]?.["assessor-results"]?.find(
        (a: any) => a.id === "T-AES-ESSAY"
      );
      overallScore = essayAssessor?.overall;
    }

    if (!mergedAssessment.meta) {
      mergedAssessment.meta = {};
    }
    mergedAssessment.meta.wordCount = totalWordCount;
    mergedAssessment.meta.errorCount = totalErrorCount;
    if (overallScore !== undefined) {
      mergedAssessment.meta.overallScore = overallScore;
    }
    mergedAssessment.meta.timestamp = new Date().toISOString();
    timings["10a_metadata"] = performance.now() - metadataStartTime;

    // Only store results if user opted in to server storage
    if (storeResults) {
      const storeResultsStartTime = performance.now();
      await storage.putResults(submissionId, mergedAssessment, 60 * 60 * 24 * 90);
      timings["11_store_results"] = performance.now() - storeResultsStartTime;
    }

    timings["0_total"] = performance.now() - requestStartTime;

    const responseHeaders = new Headers();
    responseHeaders.set("Content-Type", "application/json");
    responseHeaders.set("X-Timing-Data", JSON.stringify(timings));
    responseHeaders.set("X-Timing-Total", timings["0_total"].toFixed(2));

    const sortedTimings = Object.entries(timings)
      .filter(([key]) => key !== "0_total")
      .sort(([, a], [, b]) => (b as number) - (a as number))
      .slice(0, 5);

    const slowestOps = sortedTimings
      .map(([key, value]) => `${key}:${(value as number).toFixed(2)}`)
      .join("; ");
    responseHeaders.set("X-Timing-Slowest", slowestOps);

    // Convert Headers to plain object for c.json()
    const headersObj: Record<string, string> = {};
    responseHeaders.forEach((value, key) => {
      headersObj[key] = value;
    });

    // Return results directly in response body
    return c.json(mergedAssessment, 200, headersObj);
  } catch (error) {
    const sanitized = sanitizeError(error);
    safeLogError("Error creating submission", {
      submissionId,
      error: sanitized.message,
      name: sanitized.name,
    });
    // Return generic error message (don't expose internal details)
    return errorResponse(500, "Internal server error", c);
  }
}
