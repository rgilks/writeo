/**
 * Service Registry - Declarative configuration for Modal-based assessor services.
 *
 * This pattern allows adding new services by registering them in a single place,
 * rather than modifying 6+ files.
 */

import type { ModalService } from "../modal/types";

/**
 * Service result data types for each service category
 */
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

/**
 * Configuration for a Modal-based service.
 */
export interface ServiceDefinition<T = unknown> {
  /** Unique service identifier (e.g., "gec", "gector", "corpus") */
  id: string;

  /** Human-readable name for logging */
  name: string;

  /** Config path to check if enabled (e.g., "assessors.grammar.gecGector") */
  configPath: string;

  /** Timing key for performance tracking */
  timingKey: string;

  /** Creates the request for this service */
  createRequest: (text: string, modalService: ModalService) => Promise<Response>;

  /** Parses the JSON response into typed data */
  parseResponse: (json: unknown) => T;
}

/**
 * Registry of all Modal-based services.
 * Add new services here instead of modifying multiple files.
 */
export const SERVICE_REGISTRY: ServiceDefinition[] = [
  // Scoring Services
  {
    id: "corpus",
    name: "T-AES-CORPUS",
    configPath: "features.assessors.scoring.corpus",
    timingKey: "5e_corpus_fetch",
    createRequest: (text, modal) => modal.scoreCorpus(text),
    parseResponse: (json) => json as CorpusResult,
  },
  {
    id: "feedback",
    name: "T-AES-FEEDBACK",
    configPath: "features.assessors.scoring.feedback",
    timingKey: "5f_feedback_fetch",
    createRequest: (text, modal) => modal.scoreFeedback(text),
    parseResponse: (json) => json as FeedbackResult,
  },

  // GEC Services
  {
    id: "gec",
    name: "T-GEC-SEQ2SEQ",
    configPath: "features.assessors.grammar.gecSeq2seq",
    timingKey: "5g_gec_fetch",
    createRequest: (text, modal) => modal.correctGrammar(text),
    parseResponse: (json) => json as GECResult,
  },
  {
    id: "gector",
    name: "T-GEC-GECTOR",
    configPath: "features.assessors.grammar.gecGector",
    timingKey: "5h_gector_fetch",
    createRequest: (text, modal) => modal.correctGrammarGector(text),
    parseResponse: (json) => json as GECResult,
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
): ServiceRequest[] {
  return SERVICE_REGISTRY.filter((service) => getConfigValue(config, service.configPath)).map(
    (service) => ({
      serviceId: service.id,
      answerId,
      request: service.createRequest(text, modalService),
    }),
  );
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
      const service = SERVICE_REGISTRY.find((s) => s.id === serviceId);
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
            console.warn(`[${service.name}] Request failed for ${req.answerId}:`, e);
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
