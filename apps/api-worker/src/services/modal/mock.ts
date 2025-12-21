import type { LanguageToolResponse } from "@writeo/shared";
import type { ModalService } from "./types";

// Mock error scenarios for testing error handling
export const MOCK_MODAL_ERROR_SCENARIOS = {
  TIMEOUT: "MOCK_MODAL_TIMEOUT",
  SERVER_ERROR: "MOCK_MODAL_SERVER_ERROR",
  INVALID_RESPONSE: "MOCK_MODAL_INVALID_RESPONSE",
} as const;

// Global flags for error scenarios
let mockModalErrorScenario: string | null = null;

export function setMockModalErrorScenario(scenario: string | null): void {
  mockModalErrorScenario = scenario;
}

export function getMockModalErrorScenario(): string | null {
  return mockModalErrorScenario;
}

export class MockModalClient implements ModalService {
  async checkGrammar(text: string, _language: string, _answerId: string): Promise<Response> {
    // Validate input
    if (!text || typeof text !== "string") {
      return new Response(JSON.stringify({ error: "Invalid text input" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    // Handle error scenarios
    if (mockModalErrorScenario === MOCK_MODAL_ERROR_SCENARIOS.TIMEOUT) {
      await new Promise((resolve) => setTimeout(resolve, 35000));
      return new Response(JSON.stringify({ error: "Request timeout" }), {
        status: 504,
        headers: { "Content-Type": "application/json" },
      });
    }

    if (mockModalErrorScenario === MOCK_MODAL_ERROR_SCENARIOS.SERVER_ERROR) {
      return new Response(JSON.stringify({ error: "Internal server error" }), {
        status: 500,
        headers: { "Content-Type": "application/json" },
      });
    }

    // Return mock errors for common grammar patterns found in tests
    const matches: LanguageToolResponse["matches"] = [];

    // Enhanced error patterns with more comprehensive detection
    type ErrorPattern = {
      pattern: RegExp;
      offset: number | ((match: RegExpMatchArray) => number);
      length: number | ((match: RegExpMatchArray) => number);
      message: string;
      shortMessage: string;
      replacement: string;
      ruleId: string;
      category: { id: string; name: string };
      issueType: "error" | "warning";
    };

    const errorPatterns: ErrorPattern[] = [
      {
        pattern: /\bI goes\b/i,
        offset: (m: RegExpMatchArray) => m.index!,
        length: 7,
        message: "Use 'I go' instead of 'I goes'",
        shortMessage: "Subject-verb agreement",
        replacement: "I go",
        ruleId: "MOCK_SUBJECT_VERB_AGREEMENT",
        category: { id: "GRAMMAR", name: "Grammar" },
        issueType: "error",
      },
      {
        pattern: /\bwe goes\b/i,
        offset: (m: RegExpMatchArray) => m.index!,
        length: 8,
        message: "Use 'we go' instead of 'we goes'",
        shortMessage: "Subject-verb agreement",
        replacement: "we go",
        ruleId: "MOCK_SUBJECT_VERB_AGREEMENT",
        category: { id: "GRAMMAR", name: "Grammar" },
        issueType: "error",
      },
      {
        pattern: /\bwe was\b/i,
        offset: (m: RegExpMatchArray) => m.index!,
        length: 6,
        message: "Use 'we were' instead of 'we was'",
        shortMessage: "Subject-verb agreement",
        replacement: "we were",
        ruleId: "MOCK_SUBJECT_VERB_AGREEMENT",
        category: { id: "GRAMMAR", name: "Grammar" },
        issueType: "error",
      },
      {
        pattern: /\bwe plays\b/i,
        offset: (m: RegExpMatchArray) => m.index!,
        length: 8,
        message: "Use 'we play' instead of 'we plays'",
        shortMessage: "Subject-verb agreement",
        replacement: "we play",
        ruleId: "MOCK_SUBJECT_VERB_AGREEMENT",
        category: { id: "GRAMMAR", name: "Grammar" },
        issueType: "error",
      },
      {
        pattern: /\bhe are\b/i,
        offset: (m: RegExpMatchArray) => m.index!,
        length: 6,
        message: "Use 'he is' instead of 'he are'",
        shortMessage: "Subject-verb agreement",
        replacement: "he is",
        ruleId: "MOCK_SUBJECT_VERB_AGREEMENT",
        category: { id: "GRAMMAR", name: "Grammar" },
        issueType: "error",
      },
      {
        pattern: /\bthey is\b/i,
        offset: (m: RegExpMatchArray) => m.index!,
        length: 7,
        message: "Use 'they are' instead of 'they is'",
        shortMessage: "Subject-verb agreement",
        replacement: "they are",
        ruleId: "MOCK_SUBJECT_VERB_AGREEMENT",
        category: { id: "GRAMMAR", name: "Grammar" },
        issueType: "error",
      },
      {
        pattern: /\bI go\b(?=.*\b(yesterday|last week|last weekend|last month|last year)\b)/i,
        offset: (m: RegExpMatchArray) => m.index!,
        length: 4,
        message: "Use 'I went' for past tense when describing past events",
        shortMessage: "Verb tense",
        replacement: "I went",
        ruleId: "MOCK_VERB_TENSE",
        category: { id: "GRAMMAR", name: "Grammar" },
        issueType: "error",
      },
      {
        pattern: /\bwe play\b(?=.*\b(yesterday|last week|last weekend)\b)/i,
        offset: (m: RegExpMatchArray) => m.index!,
        length: 8,
        message: "Use 'we played' for past tense",
        shortMessage: "Verb tense",
        replacement: "we played",
        ruleId: "MOCK_VERB_TENSE",
        category: { id: "GRAMMAR", name: "Grammar" },
        issueType: "error",
      },
      {
        pattern: /\bI have\b(?=.*\b(yesterday|last week|last weekend)\b)/i,
        offset: (m: RegExpMatchArray) => m.index!,
        length: 6,
        message: "Use 'I had' for past tense",
        shortMessage: "Verb tense",
        replacement: "I had",
        ruleId: "MOCK_VERB_TENSE",
        category: { id: "GRAMMAR", name: "Grammar" },
        issueType: "error",
      },
    ];

    // Find all matches (including overlapping patterns)
    for (const errorPattern of errorPatterns) {
      let searchText = text;
      let lastIndex = 0;

      while (true) {
        const match = searchText.slice(lastIndex).match(errorPattern.pattern);
        if (!match) break;

        const matchOffset =
          typeof errorPattern.offset === "function"
            ? errorPattern.offset(match) + lastIndex
            : errorPattern.offset + lastIndex;
        const matchLength =
          typeof errorPattern.length === "function"
            ? errorPattern.length(match)
            : errorPattern.length;

        // Avoid duplicate matches at the same position
        const isDuplicate = matches.some(
          (m) => m.offset === matchOffset && m.length === matchLength,
        );
        if (!isDuplicate) {
          matches.push({
            offset: matchOffset,
            length: matchLength,
            message: errorPattern.message,
            shortMessage: errorPattern.shortMessage,
            rule: {
              id: errorPattern.ruleId,
              description: errorPattern.message,
              category: errorPattern.category,
              type: "grammar", // Required for confidence calculation
            },
            replacements: [{ value: errorPattern.replacement }],
            issueType: errorPattern.issueType as "error" | "warning",
            context: {
              text: text, // Full text for context extraction
              offset: matchOffset, // Absolute offset in full text
              length: matchLength, // Length of match
            },
          });
        }

        lastIndex = matchOffset + matchLength;
        if (lastIndex >= text.length) break;
      }
    }

    const mockResult: LanguageToolResponse = {
      matches,
    };

    // Debug logging in CI to diagnose test failures
    // Check both process.env (Node.js) and globalThis (Cloudflare Workers)
    const isCI =
      (typeof process !== "undefined" && process.env?.CI === "true") ||
      (typeof globalThis !== "undefined" && (globalThis as any).CI === "true");
    if (isCI && matches.length > 0) {
      console.log("[MockModalClient.checkGrammar] Returning mock response", {
        textLength: text.length,
        matchCount: matches.length,
        firstMatch: matches[0]
          ? {
              offset: matches[0].offset,
              length: matches[0].length,
              ruleId: matches[0].rule?.id,
              ruleType: matches[0].rule?.type,
              hasContext: !!matches[0].context,
              contextOffset: matches[0].context?.offset,
              contextLength: matches[0].context?.length,
            }
          : null,
      });
    }

    return new Response(JSON.stringify(mockResult), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }

  async scoreFeedback(text: string): Promise<Response> {
    // Mock AES-FEEDBACK scoring
    if (!text || typeof text !== "string") {
      return new Response(JSON.stringify({ error: "Invalid text input" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    const wordCount = text.split(/\s+/).filter(Boolean).length;

    // Simple heuristic for CEFR level
    let cefrScore = 3.5;
    if (wordCount > 250) cefrScore = 7.2;
    else if (wordCount > 200) cefrScore = 6.0;
    else if (wordCount > 150) cefrScore = 5.2;
    else if (wordCount > 100) cefrScore = 4.5;
    else if (wordCount > 50) cefrScore = 3.8;

    // Determine CEFR level
    let cefrLevel = "A2+";
    if (cefrScore >= 7.5) cefrLevel = "C1";
    else if (cefrScore >= 6.5) cefrLevel = "B2+";
    else if (cefrScore >= 5.5) cefrLevel = "B1+";
    else if (cefrScore >= 4.5) cefrLevel = "B1";
    else if (cefrScore >= 3.5) cefrLevel = "A2+";

    // Mock error detection
    const hasGrammarIssues = /\b(I goes|we was|they is|he are)\b/i.test(text);

    return new Response(
      JSON.stringify({
        cefr_score: Math.round(cefrScore * 100) / 100,
        cefr_level: cefrLevel,
        error_spans: [], // Conservative - empty for now
        error_types: {
          grammar: hasGrammarIssues ? 0.65 : 0.15,
          vocabulary: 0.2,
          mechanics: 0.1,
          fluency: 0.1,
          other: 0.05,
        },
      }),
      {
        status: 200,
        headers: { "Content-Type": "application/json" },
      },
    );
  }

  async scoreDeberta(text: string): Promise<Response> {
    if (!text || typeof text !== "string") {
      return new Response(JSON.stringify({ error: "Invalid text input" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }
    const wordCount = text.split(/\s+/).filter(Boolean).length;
    const overall = wordCount > 150 ? 5.5 : 3.5;

    return new Response(
      JSON.stringify({
        type: "grader",
        overall: overall,
        label: overall > 5.0 ? "B2" : "A2",
        dimensions: {
          TA: overall,
          CC: overall,
          Vocab: overall,
          Grammar: overall,
          Overall: overall,
        },
        metadata: {
          model: "mock-deberta",
          inference_time_ms: 10,
        },
      }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    );
  }

  async correctGrammar(text: string): Promise<Response> {
    if (!text || typeof text !== "string") {
      return new Response(JSON.stringify({ error: "Invalid text input" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    // Mock GEC response
    const original = text;
    let corrected = text;
    const edits = [];

    // Simple mock corrections using regex
    if (/\bI goes\b/i.test(text)) {
      corrected = corrected.replace(/\bI goes\b/gi, "I go");
      edits.push({
        start: text.toLowerCase().indexOf("i goes"),
        end: text.toLowerCase().indexOf("i goes") + 6,
        original: "I goes",
        correction: "I go",
        type: "grammar",
      });
    }

    if (/\bthree book\b/i.test(text)) {
      corrected = corrected.replace(/\bthree book\b/gi, "three books");
      edits.push({
        start: text.toLowerCase().indexOf("three book"),
        end: text.toLowerCase().indexOf("three book") + 10,
        original: "three book",
        correction: "three books",
        type: "grammar",
      });
    }

    return new Response(
      JSON.stringify({
        original,
        corrected,
        edits,
      }),
      {
        status: 200,
        headers: { "Content-Type": "application/json" },
      },
    );
  }

  async correctGrammarGector(text: string): Promise<Response> {
    // GECToR mock - same as correctGrammar for now
    return this.correctGrammar(text);
  }
}
