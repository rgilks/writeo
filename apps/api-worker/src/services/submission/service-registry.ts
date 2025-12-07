/**
 * Assessor Registry - Single source of truth for Modal-based assessor services.
 *
 * To add a new Modal service:
 * 1. Add an entry to ASSESSOR_REGISTRY below
 * 2. Add config flag to assessors.json
 * 3. Add ModalClient method (or use existing patterns)
 *
 * The registry handles: config checking, request creation, response parsing,
 * and AssessorResult creation - all in one place.
 */

import type { AssessorResult } from "@writeo/shared";
import type { ModalService } from "../modal/types";

// ============================================================================
// Result Types - Typed response structures from Modal services
// ============================================================================

export interface GECResult {
  original: string;
  corrected: string;
  edits: Array<{
    start: number;
    end: number;
    original: string;
    correction: string;
    type: string;
  }>;
}

export interface CorpusResult {
  score: number;
  cefr_level: string;
}

export interface FeedbackResult {
  cefr_score: number;
  cefr_level: string;
  error_spans: Array<{ start: number; tokens: string[] }>;
  error_types: Record<string, number>;
}

// ============================================================================
// Assessor IDs - Canonical identifiers used throughout the system
// ============================================================================

export const ASSESSOR_IDS = {
  CORPUS: "AES-CORPUS",
  FEEDBACK: "AES-FEEDBACK",
  GEC: "GEC-SEQ2SEQ",
  GECTOR: "GEC-GECTOR",
  ESSAY: "AES-ESSAY",
  LT: "GEC-LT",
} as const;

// ============================================================================
// Assessor Definition - Complete definition for a Modal-based assessor
// ============================================================================

export interface AssessorDefinition<T = unknown> {
  /** Assessor ID (e.g., "GEC-GECTOR") - matches ASSESSOR_IDS */
  assessorId: string;

  /** Short ID for internal use (e.g., "gector") */
  id: string;

  /** Human-readable display name */
  displayName: string;

  /** Assessor type for frontend display */
  type: "grader" | "feedback";

  /** Config path to check if enabled */
  configPath: string;

  /** Timing key for performance tracking */
  timingKey: string;

  /** Model name for meta info */
  model: string;

  /** Creates the request for this service */
  createRequest: (
    text: string,
    modalService: ModalService,
    answerId: string,
    config: any,
  ) => Promise<Response>;

  /** Parses the JSON response into typed data */
  parseResponse: (json: unknown) => T;

  /** Creates the AssessorResult from parsed data - uses unknown for array compatibility */
  createAssessor: (data: unknown) => AssessorResult;
}

// ============================================================================
// Assessor Registry - Add new services here!
// ============================================================================

export const ASSESSOR_REGISTRY: AssessorDefinition[] = [
  // -------------------------------------------------------------------------
  // Scoring Services (type: "grader")
  // -------------------------------------------------------------------------
  {
    assessorId: ASSESSOR_IDS.CORPUS,
    id: "corpus",
    displayName: "Corpus-Trained RoBERTa",
    type: "grader",
    configPath: "features.assessors.scoring.corpus",
    timingKey: "5e_corpus_fetch",
    model: "roberta-base",
    createRequest: (text, modal) => modal.scoreCorpus(text),
    parseResponse: (json) => json as CorpusResult,
    createAssessor: (data) => {
      const d = data as CorpusResult;
      return {
        id: ASSESSOR_IDS.CORPUS,
        name: "Corpus-Trained RoBERTa",
        type: "grader",
        overall: d.score,
        label: d.cefr_level,
        meta: { model: "roberta-base", source: "Write & Improve corpus", devMode: true },
      };
    },
  },
  {
    assessorId: ASSESSOR_IDS.FEEDBACK,
    id: "feedback",
    displayName: "AES-FEEDBACK (Multi-Task)",
    type: "grader",
    configPath: "features.assessors.scoring.feedback",
    timingKey: "5f_feedback_fetch",
    model: "deberta-v3-base",
    createRequest: (text, modal) => modal.scoreFeedback(text),
    parseResponse: (json) => json as FeedbackResult,
    createAssessor: (data) => {
      const d = data as FeedbackResult;
      return {
        id: ASSESSOR_IDS.FEEDBACK,
        name: "AES-FEEDBACK (Multi-Task)",
        type: "grader",
        overall: d.cefr_score,
        label: d.cefr_level,
        cefr: d.cefr_level,
        meta: { model: "deberta-v3-base", errorTypes: d.error_types, devMode: true },
      };
    },
  },

  // -------------------------------------------------------------------------
  // GEC Services (type: "feedback")
  // -------------------------------------------------------------------------
  {
    assessorId: ASSESSOR_IDS.GEC,
    id: "gec",
    displayName: "GEC Seq2Seq",
    type: "feedback",
    configPath: "features.assessors.grammar.gecSeq2seq",
    timingKey: "5g_gec_fetch",
    model: "flan-t5-base-gec",
    createRequest: (text, modal) => modal.correctGrammar(text),
    parseResponse: (json) => json as GECResult,
    createAssessor: (data) => {
      const d = data as GECResult;
      return {
        id: ASSESSOR_IDS.GEC,
        name: "GEC Seq2Seq",
        type: "feedback",
        meta: {
          model: "flan-t5-base-gec",
          edits: d.edits,
          correctedText: d.corrected,
          devMode: true,
        },
      };
    },
  },
  {
    assessorId: ASSESSOR_IDS.GECTOR,
    id: "gector",
    displayName: "GECToR Fast",
    type: "feedback",
    configPath: "features.assessors.grammar.gecGector",
    timingKey: "5h_gector_fetch",
    model: "gector-roberta-base-5k",
    createRequest: (text, modal) => modal.correctGrammarGector(text),
    parseResponse: (json) => json as GECResult,
    createAssessor: (data) => {
      const d = data as GECResult;
      return {
        id: ASSESSOR_IDS.GECTOR,
        name: "GECToR Fast",
        type: "feedback",
        meta: {
          model: "gector-roberta-base-5k",
          edits: d.edits,
          correctedText: d.corrected,
          devMode: true,
        },
      };
    },
  },

  // -------------------------------------------------------------------------
  // Essay & Legacy Services
  // -------------------------------------------------------------------------
  {
    assessorId: ASSESSOR_IDS.ESSAY,
    id: "essay",
    displayName: "Standard Essay Scorer",
    type: "grader",
    configPath: "features.assessors.scoring.essay",
    timingKey: "5a_essay_fetch",
    model: "roberta-base", // approximating main model
    createRequest: async (text, modal, answerId) => {
      // Construct a single-answer ModalRequest for this specific text
      return modal.gradeEssay({
        submission_id: "temp-sub-id",
        parts: [
          {
            part: 1,
            answers: [
              {
                id: answerId,
                question_id: "q1",
                question_text: "",
                answer_text: text,
              },
            ],
          },
        ],
        assessors: [],
      } as any);
    },
    parseResponse: (json) => {
      // Extract the first answer's result from the batch response
      // json is AssessmentResults { results: { parts: [...] } }
      const res = json as any;
      const part = res.results?.parts?.[0];
      const answer = part?.answers?.[0];
      const essayResult = answer?.assessorResults?.find((ar: any) => ar.id === "AES-ESSAY");
      return essayResult || null;
    },
    createAssessor: (data) => {
      // data is already the AssessorResult extracted above
      return data as AssessorResult;
    },
  },
  {
    assessorId: ASSESSOR_IDS.LT,
    id: "lt",
    displayName: "LanguageTool",
    type: "feedback",
    configPath: "features.assessors.grammar.languageTool",
    timingKey: "5b_languagetool_fetch",
    model: "languagetool",
    createRequest: (text, modal, answerId, config) => {
      const language = config?.features?.languageTool?.language || "en-GB";
      return modal.checkGrammar(text, language, answerId);
    },
    parseResponse: (json) => json as any,
    createAssessor: (data) => {
      const matches = (data as any).matches || [];
      return {
        id: ASSESSOR_IDS.LT,
        name: "LanguageTool (OSS)",
        type: "feedback",
        errors: matches.map((m: any) => ({
          message: m.message,
          start: m.offset,
          end: m.offset + m.length,
          replacements: m.replacements?.map((r: any) => r.value) || [],
          severity: m.rule?.issueType === "misspelling" ? "error" : "warning",
        })),
        meta: {
          engine: "LT-OSS",
          errorCount: matches.length,
        },
      };
    },
  },
];

/**
 * Helper to get a config value by dot-notation path.
 */
export function getConfigValue(config: unknown, path: string): boolean {
  const parts = path.split(".");
  let current: unknown = config;
  for (const part of parts) {
    if (current === null || current === undefined || typeof current !== "object") {
      return false;
    }
    current = (current as Record<string, unknown>)[part];
  }
  return Boolean(current);
}

/**
 * Pending request tracking for a service.
 */
export interface ServiceRequest {
  serviceId: string;
  answerId: string;
  request: Promise<Response>;
}

/**
 * Creates requests for all enabled services for a given answer.
 */

export function createServiceRequests(
  answerId: string,
  text: string,
  modalService: ModalService,
  config: unknown,
  requestedAssessors: string[],
): ServiceRequest[] {
  return ASSESSOR_REGISTRY.filter(
    (service) =>
      getConfigValue(config, service.configPath) && requestedAssessors.includes(service.assessorId),
  ).map((service) => ({
    serviceId: service.id,
    answerId,
    request: service.createRequest(text, modalService, answerId, config),
  }));
}

/**
 * Executes all service requests with timing tracking.
 */
export async function executeServiceRequestsGeneric(
  requests: ServiceRequest[],
  timings: Record<string, number>,
): Promise<Map<string, Map<string, unknown>>> {
  const start = performance.now();

  // Group requests by service
  const byService = new Map<string, ServiceRequest[]>();
  for (const req of requests) {
    const existing = byService.get(req.serviceId) ?? [];
    existing.push(req);
    byService.set(req.serviceId, existing);
  }

  // Execute each service's requests in parallel
  const results = new Map<string, Map<string, unknown>>();

  await Promise.all(
    Array.from(byService.entries()).map(async ([serviceId, serviceRequests]) => {
      const service = ASSESSOR_REGISTRY.find((s) => s.id === serviceId);
      if (!service) return;

      const serviceStart = performance.now();
      const map = new Map<string, unknown>();

      const resolved = await Promise.all(
        serviceRequests.map(async (req) => {
          try {
            const response = await req.request;
            if (response.ok) {
              const json = await response.json();
              return { answerId: req.answerId, data: service.parseResponse(json) };
            }
          } catch (e) {
            console.warn(`[${service.displayName}] Request failed for ${req.answerId}:`, e);
          }
          return null;
        }),
      );

      for (const result of resolved) {
        if (result) {
          map.set(result.answerId, result.data);
        }
      }

      results.set(serviceId, map);
      timings[service.timingKey] = performance.now() - serviceStart;
    }),
  );

  timings["5_generic_services_total"] = performance.now() - start;
  return results;
}
