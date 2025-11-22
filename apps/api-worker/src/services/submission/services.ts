/**
 * Parallel service calls for submission processing
 */

import type { ModalRequest } from "@writeo/shared";
import { fetchWithTimeout } from "../../utils/fetch-with-timeout";
import { getLLMAssessment } from "../ai-assessment";
import { checkAnswerRelevance, type RelevanceCheck } from "../relevance";
import { parseLLMProvider, getDefaultModel, getAPIKey, type LLMProvider } from "../llm";
import type { Env } from "../../types/env";

export interface ServiceRequests {
  ltRequests: Array<{ answerId: string; request: Promise<Response> }>;
  relevanceRequests: Array<{
    answerId: string;
    questionText: string;
    answerText: string;
    request: Promise<RelevanceCheck | null>;
  }>;
  llmAssessmentRequests: Array<{
    answerId: string;
    questionText: string;
    answerText: string;
    request: Promise<any[]>;
  }>;
  llmProvider: LLMProvider;
  apiKey: string;
  aiModel: string;
}

export function prepareServiceRequests(
  modalParts: ModalRequest["parts"],
  env: Env
): ServiceRequests {
  const language = env.LT_LANGUAGE ?? "en-GB";
  const llmProvider = parseLLMProvider(env.LLM_PROVIDER);
  const defaultModel = getDefaultModel(llmProvider);
  const aiModel = env.AI_MODEL || defaultModel;
  const apiKey = getAPIKey(llmProvider, {
    GROQ_API_KEY: env.GROQ_API_KEY,
    OPENAI_API_KEY: env.OPENAI_API_KEY,
  });

  if (!apiKey) {
    throw new Error(
      `API key not found for provider: ${llmProvider}. Please set ${llmProvider === "groq" ? "GROQ_API_KEY" : "OPENAI_API_KEY"}`
    );
  }

  const ltRequests: Array<{ answerId: string; request: Promise<Response> }> = [];
  if (env.MODAL_LT_URL) {
    for (const part of modalParts) {
      for (const answer of part.answers) {
        ltRequests.push({
          answerId: answer.id,
          request: fetchWithTimeout(`${env.MODAL_LT_URL}/check`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Token ${env.API_KEY}`,
            },
            body: JSON.stringify({
              language,
              text: answer.answer_text,
              answer_id: answer.id,
            }),
            timeout: 30000,
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
        request: checkAnswerRelevance(env.AI, answer.question_text, answer.answer_text, 0.5),
      });
    }
  }

  const llmAssessmentRequests: Array<{
    answerId: string;
    questionText: string;
    answerText: string;
    request: Promise<any[]>;
  }> = [];
  for (const part of modalParts) {
    for (const answer of part.answers) {
      llmAssessmentRequests.push({
        answerId: answer.id,
        questionText: answer.question_text,
        answerText: answer.answer_text,
        request: getLLMAssessment(
          llmProvider,
          apiKey,
          answer.question_text,
          answer.answer_text,
          aiModel
        ),
      });
    }
  }

  return {
    ltRequests,
    relevanceRequests,
    llmAssessmentRequests,
    llmProvider,
    apiKey,
    aiModel,
  };
}

export async function executeServiceRequests(
  modalRequest: ModalRequest,
  serviceRequests: ServiceRequests,
  env: Env,
  timings: Record<string, number>
): Promise<{
  essayResult: PromiseSettledResult<Response>;
  ltResults: PromiseSettledResult<Response[]>;
  llmResults: PromiseSettledResult<any[]>;
  relevanceResults: PromiseSettledResult<(RelevanceCheck | null)[]>;
}> {
  const essayStartTime = performance.now();
  const { ltRequests, llmAssessmentRequests, relevanceRequests } = serviceRequests;

  const [essayResult, ltResults, llmResults, relevanceResults] = await Promise.allSettled([
    (async () => {
      const start = performance.now();
      const result = await fetchWithTimeout(`${env.MODAL_GRADE_URL}/grade`, {
        timeout: 60000,
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Token ${env.API_KEY}`,
        },
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

  return { essayResult, ltResults, llmResults, relevanceResults };
}
