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
  requestedAssessors: string[] = [],
): ServiceRequests {
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

  const ltRequests: Array<{ answerId: string; request: Promise<Response> }> = [];
  const genericRequests: ServiceRequest[] = [];

  for (const answer of iterateAnswers(modalParts)) {
    // Legacy Services
    // Legacy Services
    // Note: LanguageTool is now handled via ASSESSOR_REGISTRY below

    // Check for Relevance Check (RELEVANCE-CHECK)
    if (requestedAssessors.includes("RELEVANCE-CHECK")) {
      relevanceRequests.push({
        answerId: answer.id,
        questionText: answer.question_text,
        answerText: answer.answer_text,
        request: checkAnswerRelevance(ai, answer.question_text, answer.answer_text, 0.5),
      });
    }

    // Check for LLM GEC (GEC-LLM)
    if (requestedAssessors.includes("GEC-LLM") && config.features.assessors.grammar.gecLlm) {
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
      requestedAssessors,
    );
    genericRequests.push(...registryRequests);
  }

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
  _modalRequest: ModalRequest,
  serviceRequests: ServiceRequests,
  _config: AppConfig,
  timings: Record<string, number>,
): Promise<{
  ltResults: PromiseSettledResult<Response[]>;
  llmResults: PromiseSettledResult<LanguageToolError[][]>;
  relevanceResults: PromiseSettledResult<(RelevanceCheck | null)[]>;
  genericResults: Map<string, Map<string, unknown>>;
}> {
  const essayStartTime = performance.now();
  const { ltRequests, llmAssessmentRequests, relevanceRequests, genericRequests } = serviceRequests;

  // Start parallel execution

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

  const [ltResults, llmResults, relevanceResults, genericResults] = await Promise.allSettled([
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
    ltResults,
    llmResults,
    relevanceResults,
    genericResults: finalGenericResults,
  };
}
