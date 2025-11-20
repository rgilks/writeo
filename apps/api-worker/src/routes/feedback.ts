import { Hono } from "hono";
import type { Env } from "../types/env";
import { isValidUUID } from "@writeo/shared";
import { errorResponse } from "../utils/errors";
import { safeLogError, sanitizeError } from "../utils/logging";
import { validateText, validateRequestBodySize } from "../utils/validation";
import { StorageService } from "../services/storage";
import { getTeacherFeedback } from "../services/feedback";
import {
  callLLMAPI,
  streamLLMAPI,
  parseLLMProvider,
  getDefaultModel,
  getAPIKey,
  type LLMProvider,
} from "../services/llm";
import { truncateEssayText, truncateQuestionText } from "../utils/text-processing";
import { MAX_ANSWER_TEXT_LENGTH, MAX_REQUEST_BODY_SIZE } from "../utils/constants";
import type {
  CreateQuestionRequest,
  CreateAnswerRequest,
  AssessmentResults,
  LanguageToolError,
} from "@writeo/shared";

export const feedbackRouter = new Hono<{ Bindings: Env }>();

feedbackRouter.post("/text/submissions/:submission_id/ai-feedback/stream", async (c) => {
  const submissionId = c.req.param("submission_id");
  const apiKey = c.req.header("Authorization")?.replace(/^Token\s+/i, "");
  const testApiKey = (c.env as any).TEST_API_KEY;

  // Accept either API_KEY or TEST_API_KEY
  const isValidKey = apiKey && (apiKey === c.env.API_KEY || (testApiKey && apiKey === testApiKey));
  if (!isValidKey) {
    return errorResponse(401, "Unauthorized");
  }

  try {
    const body = (await c.req.json()) as {
      answerId: string;
      answerText: string;
      questionText?: string;
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
        ltErrors?: LanguageToolError[];
      };
    };
    const { answerId, answerText, questionText: providedQuestionText, assessmentData } = body;

    if (!answerId || !answerText) {
      return errorResponse(400, "Missing required fields: answerId, answerText");
    }

    let questionText = providedQuestionText || "";
    let essayScores = assessmentData?.essayScores;
    let ltErrors = assessmentData?.ltErrors;

    // Check if we have questionText (required to proceed)
    // Assessment data is optional - if not provided, we'll try storage or proceed without it
    const hasQuestionText = !!providedQuestionText;

    // If questionText not provided in request, try to get from storage
    if (!hasQuestionText) {
      const storage = new StorageService(c.env.WRITEO_DATA, c.env.WRITEO_RESULTS);
      const results = await storage.getResults(submissionId);

      if (results) {
        // Extract question text if not provided
        if (!questionText) {
          const questionTexts = results.meta?.questionTexts as Record<string, string> | undefined;
          if (questionTexts && questionTexts[answerId]) {
            questionText = questionTexts[answerId];
          } else {
            const answer = await storage.getAnswer(answerId);
            if (answer) {
              const question = await storage.getQuestion(answer["question-id"]);
              if (question) {
                questionText = question.text;
              }
            }
          }
        }

        // Extract assessment data if not provided
        if (!essayScores || !ltErrors) {
          for (const part of results.results?.parts || []) {
            for (const answer of part.answers || []) {
              for (const assessor of answer["assessor-results"] || []) {
                if (!essayScores && assessor.id === "T-AES-ESSAY") {
                  essayScores = { overall: assessor.overall, dimensions: assessor.dimensions };
                }
                if (!ltErrors && assessor.id === "T-GEC-LT") {
                  ltErrors = assessor.errors as LanguageToolError[] | undefined;
                }
              }
            }
          }
        }
      } else if (!questionText) {
        // If no storage and no question text provided, we can't proceed
        return errorResponse(
          400,
          "questionText is required when submission is not stored. Please provide questionText in the request body."
        );
      }
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
        `API key not found for provider: ${llmProvider}. Please set ${llmProvider === "groq" ? "GROQ_API_KEY" : "OPENAI_API_KEY"}`
      );
    }

    let essayContext = "";
    if (essayScores) {
      essayContext = `\n\nAssessment Results:
- Overall Score: ${essayScores.overall ?? essayScores.dimensions?.Overall ?? "N/A"} / 9.0
- Task Achievement (TA): ${essayScores.dimensions?.TA ?? "N/A"} / 9.0
- Coherence & Cohesion (CC): ${essayScores.dimensions?.CC ?? "N/A"} / 9.0
- Vocabulary: ${essayScores.dimensions?.Vocab ?? "N/A"} / 9.0
- Grammar: ${essayScores.dimensions?.Grammar ?? "N/A"} / 9.0`;
    }

    let grammarContext = "";
    if (ltErrors && ltErrors.length > 0) {
      const errorSummary = ltErrors
        .slice(0, 10)
        .map(
          (err, idx) =>
            `${idx + 1}. ${err.message} (${err.category})${err.suggestions ? ` - Suggestions: ${err.suggestions.slice(0, 2).join(", ")}` : ""}`
        )
        .join("\n");
      grammarContext = `\n\nGrammar & Language Issues Found (${ltErrors.length} total):\n${errorSummary}${ltErrors.length > 10 ? `\n... and ${ltErrors.length - 10} more issues` : ""}`;
    }

    // Truncate very long essays to keep costs under control
    const truncatedAnswerText = truncateEssayText(answerText);
    const truncatedQuestionText = truncateQuestionText(questionText);

    const prompt = `You are an expert English language tutor specializing in academic argumentative writing. Analyze the following essay answer and provide detailed, contextual feedback.

Question: ${truncatedQuestionText}

Answer: ${truncatedAnswerText}${essayContext}${grammarContext}

Provide comprehensive feedback focusing on:

1. Task Achievement:
- Does the answer address ALL parts of the question?
- Is there a clear position/opinion stated (for opinion questions)?
- Are ideas fully developed with specific examples and explanations?
- For discussion questions: Are both sides addressed before giving an opinion?

2. Structure & Organization:
- Is there a clear introduction that presents the topic and position?
- Are body paragraphs well-organized with topic sentences?
- Does each paragraph focus on one main idea?
- Is there a conclusion that summarizes the main points?

3. Coherence & Cohesion:
- Are ideas connected smoothly using linking words (however, furthermore, therefore, etc.)?
- Do paragraphs flow logically from one to the next?
- Is the writing cohesive and easy to follow?

4. Vocabulary:
- Is there a good range of vocabulary (avoiding repetition)?
- Are words used accurately and appropriately?
- Are there attempts at less common vocabulary?

5. Grammar:
- Is grammar accurate?
- Is there a variety of sentence structures?

Provide feedback covering:
- Relevance: Does the answer fully address the question? (explain which parts were addressed and which might be missing)
- Strengths: 2-3 specific things the student did well (be specific, e.g., "Clear position stated in introduction" or "Good use of linking words")
- Improvements: 2-3 specific, actionable areas to improve (prioritize the most important issues, e.g., "Develop your second paragraph with a concrete example" or "Use more varied vocabulary instead of repeating 'important'")
- Overall: A brief summary comment (1-2 sentences) that focuses on improvement and suggests next steps

Respond in a clear, professional manner. Don't mention technical terms like "CEFR" or "band scores" - focus on actionable feedback to improve their academic writing skills.`;

    const stream = new ReadableStream({
      async start(controller) {
        const encoder = new TextEncoder();
        try {
          controller.enqueue(
            encoder.encode(
              `data: ${JSON.stringify({ type: "start", message: "Starting AI feedback generation..." })}\n\n`
            )
          );

          // Use streaming API for real-time feedback
          let hasContent = false;
          try {
            for await (const chunk of streamLLMAPI(
              llmProvider,
              apiKey,
              aiModel,
              [
                {
                  role: "system",
                  content:
                    "You are a professional writing tutor specializing in academic argumentative writing. Provide clear, direct feedback focused on actionable improvements. Never mention technical terms like , CEFR, or band scores.",
                },
                { role: "user", content: prompt },
              ],
              1000
            )) {
              hasContent = true;
              controller.enqueue(
                encoder.encode(`data: ${JSON.stringify({ type: "chunk", text: chunk })}\n\n`)
              );
            }
          } catch (streamError) {
            // If streaming fails, fall back to non-streaming API
            console.error("[Stream] Streaming failed, falling back to non-streaming:", streamError);
            const responseText = await callLLMAPI(
              llmProvider,
              apiKey,
              aiModel,
              [
                {
                  role: "system",
                  content:
                    "You are a professional writing tutor specializing in academic argumentative writing. Provide clear, direct feedback focused on actionable improvements. Never mention technical terms like , CEFR, or band scores.",
                },
                { role: "user", content: prompt },
              ],
              1000
            );

            if (responseText && responseText.trim().length > 0) {
              hasContent = true;
              const words = responseText.match(/\S+|\s+/g) || [];
              if (words.length === 0) {
                controller.enqueue(
                  encoder.encode(
                    `data: ${JSON.stringify({ type: "chunk", text: responseText })}\n\n`
                  )
                );
              } else {
                for (const word of words) {
                  controller.enqueue(
                    encoder.encode(`data: ${JSON.stringify({ type: "chunk", text: word })}\n\n`)
                  );
                  await new Promise((resolve) => setTimeout(resolve, 20));
                }
              }
            }
          }

          if (!hasContent) {
            controller.enqueue(
              encoder.encode(
                `data: ${JSON.stringify({ type: "chunk", text: "Unable to generate feedback at this time. Please try again." })}\n\n`
              )
            );
          }

          controller.enqueue(
            encoder.encode(
              `data: ${JSON.stringify({ type: "done", message: "Feedback generation complete" })}\n\n`
            )
          );
        } catch (error) {
          controller.enqueue(
            encoder.encode(
              `data: ${JSON.stringify({ type: "error", message: error instanceof Error ? error.message : String(error) })}\n\n`
            )
          );
        } finally {
          controller.close();
        }
      },
    });

    return new Response(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  } catch (error) {
    const sanitized = sanitizeError(error);
    safeLogError("Error streaming AI feedback", sanitized);
    return errorResponse(500, "Internal server error", c);
  }
});

feedbackRouter.post("/text/submissions/:submission_id/teacher-feedback", async (c) => {
  console.log("[teacher-feedback] Endpoint called, method:", c.req.method, "path:", c.req.path);
  const submissionId = c.req.param("submission_id");
  console.log("[teacher-feedback] submissionId:", submissionId);
  if (!isValidUUID(submissionId)) {
    console.log("[teacher-feedback] Invalid UUID");
    return errorResponse(400, "Invalid submission_id format", c);
  }

  const apiKey = c.req.header("Authorization")?.replace(/^Token\s+/i, "");
  const testApiKey = (c.env as any).TEST_API_KEY;

  // Accept either API_KEY or TEST_API_KEY
  const isValidKey = apiKey && (apiKey === c.env.API_KEY || (testApiKey && apiKey === testApiKey));
  if (!isValidKey) {
    return errorResponse(401, "Unauthorized", c);
  }

  try {
    const sizeValidation = await validateRequestBodySize(c.req.raw, MAX_REQUEST_BODY_SIZE);
    if (!sizeValidation.valid) {
      return errorResponse(413, sizeValidation.error || "Request body too large (max 1MB)", c);
    }

    const body = (await c.req.json()) as {
      answerId: string;
      mode: "clues" | "explanation";
      answerText: string;
      questionText?: string;
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
        ltErrors?: LanguageToolError[];
        llmErrors?: LanguageToolError[];
        relevanceCheck?: {
          addressesQuestion: boolean;
          score: number;
          threshold: number;
        };
      };
    };
    const { answerId, mode, answerText, questionText: providedQuestionText, assessmentData } = body;

    // Debug logging
    console.log("[teacher-feedback] Request received:", {
      submissionId,
      answerId,
      mode,
      hasQuestionText: !!providedQuestionText,
      questionTextLength: providedQuestionText?.length || 0,
      hasAssessmentData: !!assessmentData,
      assessmentDataKeys: assessmentData ? Object.keys(assessmentData) : [],
    });

    if (!answerId || !mode || !answerText) {
      return errorResponse(400, "Missing required fields: answerId, mode, answerText", c);
    }

    if (!isValidUUID(answerId)) {
      return errorResponse(400, "Invalid answerId format", c);
    }

    if (mode !== "clues" && mode !== "explanation") {
      return errorResponse(400, "Mode must be 'clues' or 'explanation'", c);
    }

    const textValidation = validateText(answerText, MAX_ANSWER_TEXT_LENGTH);
    if (!textValidation.valid) {
      return errorResponse(
        400,
        `Invalid answerText: ${textValidation.error || "Invalid content"}`,
        c
      );
    }

    let questionText = providedQuestionText || "";
    let essayScores = assessmentData?.essayScores;
    let ltErrors = assessmentData?.ltErrors;
    let llmErrors = assessmentData?.llmErrors;
    let relevanceCheck = assessmentData?.relevanceCheck;
    let results: AssessmentResults | null = null;
    let teacherAssessor: any = undefined;

    // Check if we have required data, otherwise try storage
    // At minimum we need questionText to proceed
    const hasRequiredData = providedQuestionText;

    // If assessment data not provided in request, try to get from storage
    if (!hasRequiredData) {
      const storage = new StorageService(c.env.WRITEO_DATA, c.env.WRITEO_RESULTS);
      results = await storage.getResults(submissionId);

      if (results) {
        // Extract question text if not provided
        if (!questionText) {
          const questionTexts = results.meta?.questionTexts as Record<string, string> | undefined;
          if (questionTexts && questionTexts[answerId]) {
            questionText = questionTexts[answerId];
          } else {
            const answer = await storage.getAnswer(answerId);
            if (answer) {
              const question = await storage.getQuestion(answer["question-id"]);
              if (question) {
                questionText = question.text;
              }
            }
          }
        }

        // Extract assessment data if not provided
        if (!essayScores || !ltErrors || !llmErrors || !relevanceCheck) {
          for (const part of results.results?.parts || []) {
            for (const answer of part.answers || []) {
              for (const assessor of answer["assessor-results"] || []) {
                if (!essayScores && assessor.id === "T-AES-ESSAY") {
                  essayScores = { overall: assessor.overall, dimensions: assessor.dimensions };
                }
                if (!ltErrors && assessor.id === "T-GEC-LT") {
                  ltErrors = assessor.errors as LanguageToolError[] | undefined;
                }
                if (!llmErrors && assessor.id === "T-GEC-LLM") {
                  llmErrors = assessor.errors as LanguageToolError[] | undefined;
                }
                if (!relevanceCheck && assessor.id === "T-RELEVANCE-CHECK" && assessor.meta) {
                  const meta = assessor.meta as any;
                  relevanceCheck = {
                    addressesQuestion: meta.addressesQuestion ?? false,
                    score: meta.similarityScore ?? 0,
                    threshold: meta.threshold ?? 0.5,
                  };
                }
              }
            }
          }
        }

        // Get teacher assessor if results exist
        const firstPart = results.results?.parts?.[0];
        teacherAssessor = firstPart?.answers?.[0]?.["assessor-results"]?.find(
          (a: any) => a.id === "T-TEACHER-FEEDBACK"
        );
      } else if (!questionText) {
        // If no storage and no question text provided, we can't proceed
        return errorResponse(
          400,
          "questionText is required when submission is not stored. Please provide questionText in the request body.",
          c
        );
      }
    }

    // Ensure we have questionText before proceeding (double-check)
    if (!questionText || questionText.trim().length === 0) {
      return errorResponse(
        400,
        "questionText is required. Please provide questionText in the request body.",
        c
      );
    }

    // Check if we already have cached feedback for this mode (only if results exist)
    const existingMeta = (teacherAssessor?.meta || {}) as Record<string, any>;
    const cachedMessage =
      mode === "clues"
        ? existingMeta.cluesMessage
        : mode === "explanation"
          ? existingMeta.explanationMessage
          : undefined;

    let teacherFeedback: { message: string; focusArea?: string };

    if (cachedMessage && typeof cachedMessage === "string") {
      // Return cached feedback
      teacherFeedback = {
        message: cachedMessage,
        focusArea: existingMeta.focusArea as string | undefined,
      };
    } else {
      // Generate new feedback
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
          `API key not found for provider: ${llmProvider}. Please set ${llmProvider === "groq" ? "GROQ_API_KEY" : "OPENAI_API_KEY"}`
        );
      }

      teacherFeedback = await getTeacherFeedback(
        llmProvider,
        apiKey,
        questionText,
        answerText,
        aiModel,
        mode,
        essayScores,
        ltErrors,
        llmErrors,
        relevanceCheck
      );

      // Store the new feedback (only if results exist from storage)
      if (results) {
        const storage = new StorageService(c.env.WRITEO_DATA, c.env.WRITEO_RESULTS);
        const firstPart = results.results?.parts?.[0];
        if (firstPart) {
          if (!teacherAssessor) {
            teacherAssessor = {
              id: "T-TEACHER-FEEDBACK",
              name: "Teacher's Feedback",
              type: "feedback",
              meta: {},
            };

            // Add to first answer's assessor-results
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
    }

    return c.json(teacherFeedback);
  } catch (error) {
    const sanitized = sanitizeError(error);
    safeLogError("Error getting Teacher feedback", sanitized);
    return errorResponse(500, "Internal server error", c);
  }
});
