import type { ModalRequest, LanguageToolError, RelevanceCheck } from "@writeo/shared";
import { getLLMAssessment } from "../ai-assessment";
import { checkAnswerRelevance } from "../relevance";
import type { AppConfig } from "../config";
import type { LLMProvider } from "../llm";
import { iterateAnswers } from "./utils";
import type { ModalService } from "../modal/types";
import {
  createServiceRequests,
  executeServiceRequestsGeneric,
  type ServiceRequest,
} from "./service-registry";

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
  genericRequests: ServiceRequest[];
  llmProvider: LLMProvider;
  apiKey: string;
  aiModel: string;
  modalService: ModalService;
}

export function prepareServiceRequests(
  modalParts: ModalRequest["parts"],
  config: AppConfig,
  ai: Ai,
  modalService: ModalService,
): ServiceRequests {
  const language = config.features.languageTool.language;

  let ltRequests: Array<{ answerId: string; request: Promise<Response> }>;
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

  // Enable LT if configured OR if using mock services (to test the flow)
  const ltEnabled =
    (config.features.languageTool.enabled && config.modal.ltUrl) || config.features.mockServices;
  const requests: Array<{ answerId: string; request: Promise<Response> }> = [];
  const genericRequests: ServiceRequest[] = [];

  for (const answer of iterateAnswers(modalParts)) {
    // Legacy Services
    if (ltEnabled) {
      requests.push({
        answerId: answer.id,
        request: modalService.checkGrammar(answer.answer_text, language, answer.id),
      });
    }

    relevanceRequests.push({
      answerId: answer.id,
      questionText: answer.question_text,
      answerText: answer.answer_text,
      request: checkAnswerRelevance(ai, answer.question_text, answer.answer_text, 0.5),
    });

    if (config.features.assessors.grammar.gecLlm) {
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
          config.features.mockServices,
        ),
      });
    }

    // Generic Registry Services (Corpus, Feedback, GEC, GECToR, etc.)
    const registryRequests = createServiceRequests(
      answer.id,
      answer.answer_text,
      modalService,
      config,
    );
    genericRequests.push(...registryRequests);
  }

  ltRequests = requests;

  return {
    ltRequests,
    relevanceRequests,
    llmAssessmentRequests,
    genericRequests,
    llmProvider: config.llm.provider,
    apiKey: config.llm.apiKey,
    aiModel: config.llm.model,
    modalService,
  };
}

export async function executeServiceRequests(
  modalRequest: ModalRequest,
  serviceRequests: ServiceRequests,
  _config: AppConfig,
  timings: Record<string, number>,
): Promise<{
  essayResult: PromiseSettledResult<Response>;
  ltResults: PromiseSettledResult<Response[]>;
  llmResults: PromiseSettledResult<LanguageToolError[][]>;
  relevanceResults: PromiseSettledResult<(RelevanceCheck | null)[]>;
  genericResults: Map<string, Map<string, unknown>>;
}> {
  const essayStartTime = performance.now();
  const { ltRequests, llmAssessmentRequests, relevanceRequests, genericRequests, modalService } =
    serviceRequests;

  // Start parallel execution
  const essayPromise = (async () => {
    const start = performance.now();
    const result = await modalService.gradeEssay(modalRequest);
    timings["5a_essay_fetch"] = performance.now() - start;
    return result;
  })();

  const ltPromise = (async () => {
    const start = performance.now();
    const resolved = await Promise.all(ltRequests.map((r) => r.request));
    timings["5b_languagetool_fetch"] = performance.now() - start;
    return resolved;
  })();

  const llmPromise = (async () => {
    const start = performance.now();
    const resolved = await Promise.all(llmAssessmentRequests.map((r) => r.request));
    timings["5d_ai_assessment_fetch"] = performance.now() - start;
    return resolved;
  })();

  const relevancePromise = (async () => {
    const start = performance.now();
    const resolved = await Promise.all(relevanceRequests.map((r) => r.request));
    timings["5c_relevance_fetch"] = performance.now() - start;
    return resolved;
  })();

  const genericPromise = executeServiceRequestsGeneric(genericRequests, timings);

  const [essayResult, ltResults, llmResults, relevanceResults, genericResults] =
    await Promise.allSettled([
      essayPromise,
      ltPromise,
      llmPromise,
      relevancePromise,
      genericPromise,
    ]);

  timings["5_parallel_services_total"] = performance.now() - essayStartTime;

  // Unpack genericResults from PromiseSettledResult if successful, else generic empty map
  const finalGenericResults =
    genericResults.status === "fulfilled"
      ? genericResults.value
      : new Map<string, Map<string, unknown>>();

  if (genericResults.status === "rejected") {
    console.error("Generic services failed:", genericResults.reason);
  }

  return {
    essayResult,
    ltResults,
    llmResults,
    relevanceResults,
    genericResults: finalGenericResults,
  };
}
