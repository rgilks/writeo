/**
 * Streaming request handler
 */

import { errorResponse } from "../../utils/errors";
import { parseLLMProvider, getDefaultModel, getAPIKey, type LLMProvider } from "../../services/llm";
import { getServices } from "../../utils/context";
import { StorageService } from "../../services/storage";
import type { Context } from "hono";
import type { Env } from "../../types/env";
import type { AssessmentResults, LanguageToolError } from "@writeo/shared";
import {
  StreamingFeedbackRequestSchema,
  type AssessmentDataInput,
  formatZodError,
} from "./validation";

type EssayScores = AssessmentDataInput["essayScores"];
type GrammarErrors = AssessmentDataInput["ltErrors"];

interface StreamingRequestData {
  questionText: string;
  answerText: string;
  essayScores?: EssayScores;
  ltErrors?: GrammarErrors;
  llmProvider: LLMProvider;
  apiKey: string;
  aiModel: string;
}

/**
 * Extracts essay scores and language tool errors from assessment results
 */
function extractAssessmentData(results: AssessmentResults): {
  essayScores?: EssayScores;
  ltErrors?: GrammarErrors;
} {
  let essayScores: EssayScores | undefined;
  let ltErrors: GrammarErrors;

  for (const part of results.results?.parts || []) {
    for (const answer of part.answers || []) {
      for (const assessor of answer["assessor-results"] || []) {
        if (!essayScores && assessor.id === "T-AES-ESSAY") {
          essayScores = { overall: assessor.overall, dimensions: assessor.dimensions };
        }
        if (!ltErrors && assessor.id === "T-GEC-LT") {
          ltErrors = assessor.errors as LanguageToolError[] | undefined;
        }
        // Early exit if both found
        if (essayScores && ltErrors) break;
      }
      if (essayScores && ltErrors) break;
    }
    if (essayScores && ltErrors) break;
  }

  return { essayScores, ltErrors };
}

/**
 * Fetches question text from storage, trying multiple sources
 */
async function fetchQuestionText(
  storage: StorageService,
  results: AssessmentResults | null,
  answerId: string,
): Promise<string | null> {
  // Try metadata first (fastest)
  if (results) {
    const questionTexts = results.meta?.questionTexts as Record<string, string> | undefined;
    if (questionTexts?.[answerId]) {
      return questionTexts[answerId];
    }
  }

  // Fall back to answer -> question lookup
  const answer = await storage.getAnswer(answerId);
  if (answer) {
    const question = await storage.getQuestion(answer["question-id"]);
    if (question) {
      return question.text;
    }
  }

  return null;
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
 * Handles streaming feedback requests by validating input, fetching missing data from storage,
 * and setting up LLM configuration for streaming AI feedback generation.
 *
 * @param c - Hono context with environment bindings
 * @returns Request data for streaming or error response
 */
export async function handleStreamingRequest(
  c: Context<{ Bindings: Env }>,
): Promise<StreamingRequestData | Response> {
  const submissionId = c.req.param("submission_id");
  const parsedBody = StreamingFeedbackRequestSchema.safeParse(await c.req.json());
  if (!parsedBody.success) {
    return errorResponse(400, `Invalid request body: ${formatZodError(parsedBody.error)}`, c);
  }
  const body = parsedBody.data;

  let questionText = body.questionText || "";
  let essayScores = body.assessmentData?.essayScores;
  let ltErrors = body.assessmentData?.ltErrors;

  // Fetch missing data from storage if needed
  if (!questionText || !essayScores || !ltErrors) {
    const { storage } = getServices(c);
    const results = await storage.getResults(submissionId);

    if (!questionText) {
      const fetchedQuestionText = results
        ? await fetchQuestionText(storage, results, body.answerId)
        : null;
      if (fetchedQuestionText) {
        questionText = fetchedQuestionText;
      } else if (!results) {
        return errorResponse(
          400,
          "questionText is required when submission is not stored. Please provide questionText in the request body.",
          c,
        );
      }
    }

    if (results && (!essayScores || !ltErrors)) {
      const assessmentData = extractAssessmentData(results);
      essayScores = essayScores || assessmentData.essayScores;
      ltErrors = ltErrors || assessmentData.ltErrors;
    }
  }

  const llmConfig = setupLLMConfig(c.env);
  if (!llmConfig) {
    const provider = parseLLMProvider(c.env.LLM_PROVIDER);
    return errorResponse(
      500,
      `API key not found for provider: ${provider}. Please set ${provider === "groq" ? "GROQ_API_KEY" : "OPENAI_API_KEY"}`,
      c,
    );
  }

  return {
    questionText,
    answerText: body.answerText,
    essayScores,
    ltErrors,
    llmProvider: llmConfig.provider,
    apiKey: llmConfig.apiKey,
    aiModel: llmConfig.model,
  };
}
