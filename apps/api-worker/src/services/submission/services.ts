/**
 * Parallel service calls for submission processing
 */

import type { ModalRequest, LanguageToolError } from "@writeo/shared";
import { getLLMAssessment } from "../ai-assessment";
import { checkAnswerRelevance, type RelevanceCheck } from "../relevance";
import type { AppConfig } from "../config";
import type { LLMProvider } from "../llm";
import { iterateAnswers } from "./utils";
import { postJsonWithAuth } from "../../utils/http";

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
    request: Promise<LanguageToolError[]>;
  }>;
  llmProvider: LLMProvider;
  apiKey: string;
  aiModel: string;
}

export function prepareServiceRequests(
  modalParts: ModalRequest["parts"],
  config: AppConfig,
  ai: Ai,
): ServiceRequests {
  const language = config.features.languageTool.language;

  const ltRequests: Array<{ answerId: string; request: Promise<Response> }> = [];
  const relevanceRequests: Array<{
    answerId: string;
    questionText: string;
    answerText: string;
    request: Promise<RelevanceCheck | null>;
  }> = [];
  const llmAssessmentRequests: Array<{
    answerId: string;
    questionText: string;
    answerText: string;
    request: Promise<LanguageToolError[]>;
  }> = [];

  const ltEnabled = config.features.languageTool.enabled && config.modal.ltUrl;
  for (const answer of iterateAnswers(modalParts)) {
    if (ltEnabled) {
      ltRequests.push({
        answerId: answer.id,
        request: postJsonWithAuth(
          `${config.modal.ltUrl}/check`,
          config.api.key,
          {
            language,
            text: answer.answer_text,
            answer_id: answer.id,
          },
          30000,
        ),
      });
    }

    relevanceRequests.push({
      answerId: answer.id,
      questionText: answer.question_text,
      answerText: answer.answer_text,
      request: checkAnswerRelevance(ai, answer.question_text, answer.answer_text, 0.5),
    });

    llmAssessmentRequests.push({
      answerId: answer.id,
      questionText: answer.question_text,
      answerText: answer.answer_text,
      request: getLLMAssessment(
        config.llm.provider,
        config.llm.apiKey,
        answer.question_text,
        answer.answer_text,
        config.llm.model,
      ),
    });
  }

  return {
    ltRequests,
    relevanceRequests,
    llmAssessmentRequests,
    llmProvider: config.llm.provider,
    apiKey: config.llm.apiKey,
    aiModel: config.llm.model,
  };
}

export async function executeServiceRequests(
  modalRequest: ModalRequest,
  serviceRequests: ServiceRequests,
  config: AppConfig,
  timings: Record<string, number>,
): Promise<{
  essayResult: PromiseSettledResult<Response>;
  ltResults: PromiseSettledResult<Response[]>;
  llmResults: PromiseSettledResult<LanguageToolError[][]>;
  relevanceResults: PromiseSettledResult<(RelevanceCheck | null)[]>;
}> {
  const essayStartTime = performance.now();
  const { ltRequests, llmAssessmentRequests, relevanceRequests } = serviceRequests;

  const [essayResult, ltResults, llmResults, relevanceResults] = await Promise.allSettled([
    (async () => {
      const start = performance.now();
      const result = await postJsonWithAuth(
        `${config.modal.gradeUrl}/grade`,
        config.api.key,
        modalRequest,
        60000,
      );
      timings["5a_essay_fetch"] = performance.now() - start;
      return result;
    })(),
    (async () => {
      const start = performance.now();
      const resolved = await Promise.all(ltRequests.map((r) => r.request));
      timings["5b_languagetool_fetch"] = performance.now() - start;
      return resolved;
    })(),
    (async () => {
      const start = performance.now();
      const resolved = await Promise.all(llmAssessmentRequests.map((r) => r.request));
      timings["5d_ai_assessment_fetch"] = performance.now() - start;
      return resolved;
    })(),
    (async () => {
      const start = performance.now();
      const resolved = await Promise.all(relevanceRequests.map((r) => r.request));
      timings["5c_relevance_fetch"] = performance.now() - start;
      return resolved;
    })(),
  ]);

  timings["5_parallel_services_total"] = performance.now() - essayStartTime;

  return { essayResult, ltResults, llmResults, relevanceResults };
}
