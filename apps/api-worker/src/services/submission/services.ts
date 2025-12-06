/**
 * Parallel service calls for submission processing
 */

import type { ModalRequest, LanguageToolError, RelevanceCheck } from "@writeo/shared";
import { getLLMAssessment } from "../ai-assessment";
import { checkAnswerRelevance } from "../relevance";
import type { AppConfig } from "../config";
import type { LLMProvider } from "../llm";
import { iterateAnswers } from "./utils";
import type { ModalService } from "../modal/types";

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
  corpusRequests: Array<{ answerId: string; request: Promise<Response> }>; // Dev mode corpus scoring
  feedbackRequests: Array<{ answerId: string; request: Promise<Response> }>; // Dev mode T-AES-FEEDBACK
  gecRequests: Array<{ answerId: string; request: Promise<Response> }>; // Dev mode T-GEC-SEQ2SEQ
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

  // Corpus scoring requests (dev mode / mock services)
  const corpusRequests: Array<{ answerId: string; request: Promise<Response> }> = [];
  // Feedback scoring requests (dev mode / mock services)
  const feedbackRequests: Array<{ answerId: string; request: Promise<Response> }> = [];
  // GEC requests (dev mode / mock services)
  const gecRequests: Array<{ answerId: string; request: Promise<Response> }> = [];

  for (const answer of iterateAnswers(modalParts)) {
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

    // T-GEC-LLM: Only run if explicitly enabled (expensive, $0.002/submission)
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
          config.features.mockServices, // Pass mock flag from config
        ),
      });
    }

    // T-AES-CORPUS: Add corpus scoring if enabled (default: ON, best scorer)
    if (config.features.assessors.scoring.corpus) {
      corpusRequests.push({
        answerId: answer.id,
        request: modalService.scoreCorpus(answer.answer_text),
      });
    }

    // T-AES-FEEDBACK: Add feedback scoring if enabled (default: OFF, experimental)
    if (config.features.assessors.scoring.feedback) {
      feedbackRequests.push({
        answerId: answer.id,
        request: modalService.scoreFeedback(answer.answer_text),
      });
    }

    // T-GEC-SEQ2SEQ: Add GEC correction if enabled (default: ON, best GEC)
    if (config.features.assessors.grammar.gecSeq2seq) {
      gecRequests.push({
        answerId: answer.id,
        request: modalService.correctGrammar(answer.answer_text),
      });
    }
  }

  ltRequests = requests;

  return {
    ltRequests,
    relevanceRequests,
    llmAssessmentRequests,
    corpusRequests,
    feedbackRequests,
    gecRequests,
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
  corpusResults: PromiseSettledResult<Response[]>;
  feedbackResults: PromiseSettledResult<Response[]>;
  gecResults: PromiseSettledResult<Response[]>;
}> {
  const essayStartTime = performance.now();
  const {
    ltRequests,
    llmAssessmentRequests,
    relevanceRequests,
    corpusRequests,
    feedbackRequests,
    gecRequests,
    modalService,
  } = serviceRequests;

  const [
    essayResult,
    ltResults,
    llmResults,
    relevanceResults,
    corpusResults,
    feedbackResults,
    gecResults,
  ] = await Promise.allSettled([
    (async () => {
      const start = performance.now();
      const result = await modalService.gradeEssay(modalRequest);
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
    (async () => {
      const start = performance.now();
      const resolved = await Promise.all(corpusRequests.map((r) => r.request));
      timings["5e_corpus_fetch"] = performance.now() - start;
      return resolved;
    })(),
    (async () => {
      const start = performance.now();
      const resolved = await Promise.all(feedbackRequests.map((r) => r.request));
      timings["5f_feedback_fetch"] = performance.now() - start;
      return resolved;
    })(),
    (async () => {
      const start = performance.now();
      const resolved = await Promise.all(gecRequests.map((r) => r.request));
      timings["5g_gec_fetch"] = performance.now() - start;
      return resolved;
    })(),
  ]);

  timings["5_parallel_services_total"] = performance.now() - essayStartTime;

  return {
    essayResult,
    ltResults,
    llmResults,
    relevanceResults,
    corpusResults,
    feedbackResults,
    gecResults,
  };
}
