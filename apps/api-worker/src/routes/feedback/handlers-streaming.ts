/**
 * Streaming request handler
 */

import { errorResponse } from "../../utils/errors";
import { parseLLMProvider, getDefaultModel, getAPIKey } from "../../services/llm";
import { StorageService } from "../../services/storage";
import type { Context } from "hono";
import type { Env } from "../../types/env";

export async function handleStreamingRequest(c: Context<{ Bindings: Env }>) {
  const submissionId = c.req.param("submission_id");
  const body = (await c.req.json()) as {
    answerId: string;
    answerText: string;
    questionText?: string;
    assessmentData?: {
      essayScores?: any;
      ltErrors?: any[];
    };
  };

  if (!body.answerId || !body.answerText) {
    return errorResponse(400, "Missing required fields: answerId, answerText", c);
  }

  let questionText = body.questionText || "";
  let essayScores = body.assessmentData?.essayScores;
  let ltErrors = body.assessmentData?.ltErrors;

  if (!questionText) {
    const storage = new StorageService(c.env.WRITEO_DATA, c.env.WRITEO_RESULTS);
    const results = await storage.getResults(submissionId);
    if (results) {
      const questionTexts = results.meta?.questionTexts as Record<string, string> | undefined;
      if (questionTexts && questionTexts[body.answerId]) {
        questionText = questionTexts[body.answerId] ?? "";
      } else {
        const answer = await storage.getAnswer(body.answerId);
        if (answer) {
          const question = await storage.getQuestion(answer["question-id"]);
          if (question) {
            questionText = question.text;
          }
        }
      }

      if (!essayScores || !ltErrors) {
        for (const part of results.results?.parts || []) {
          for (const answer of part.answers || []) {
            for (const assessor of answer["assessor-results"] || []) {
              if (!essayScores && assessor.id === "T-AES-ESSAY") {
                essayScores = { overall: assessor.overall, dimensions: assessor.dimensions };
              }
              if (!ltErrors && assessor.id === "T-GEC-LT") {
                ltErrors = assessor.errors;
              }
            }
          }
        }
      }
    } else if (!questionText) {
      return errorResponse(
        400,
        "questionText is required when submission is not stored. Please provide questionText in the request body.",
        c
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
      `API key not found for provider: ${llmProvider}. Please set ${llmProvider === "groq" ? "GROQ_API_KEY" : "OPENAI_API_KEY"}`,
      c
    );
  }

  return {
    questionText,
    answerText: body.answerText,
    essayScores,
    ltErrors,
    llmProvider,
    apiKey,
    aiModel,
  };
}
