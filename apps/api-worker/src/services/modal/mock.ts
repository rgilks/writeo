import type { ModalRequest, LanguageToolResponse } from "@writeo/shared";
import type { ModalService } from "./types";
import type { EssayResult } from "../clients/essay-client";

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
  async gradeEssay(request: ModalRequest): Promise<Response> {
    // Validate input
    if (!request || !request.submission_id || !request.parts || request.parts.length === 0) {
      return new Response(JSON.stringify({ error: "Invalid request" }), {
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

    if (mockModalErrorScenario === MOCK_MODAL_ERROR_SCENARIOS.INVALID_RESPONSE) {
      return new Response("Invalid JSON {", {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }

    // Generate realistic scores based on text quality (simple heuristic)
    const allText = request.parts
      .flatMap((p) => p.answers.map((a) => a.answer_text))
      .join(" ")
      .toLowerCase();

    // Detect common errors to adjust scores
    const hasErrors =
      /\b(I goes|we was|we plays|he are|they is)\b/i.test(allText) ||
      /\b(I go|we go)\b.*\b(yesterday|last week|last weekend)\b/i.test(allText);

    const baseScore = hasErrors ? 3.5 : 4.0;
    const variance = 0.3;

    const mockResult: EssayResult = {
      submission_id: request.submission_id,
      parts: request.parts.map((part) => ({
        part: part.part,
        answers: part.answers.map((answer) => {
          // Vary scores slightly per answer for realism
          const taScore = Math.max(
            1.0,
            Math.min(5.0, baseScore + (Math.random() - 0.5) * variance),
          );
          const ccScore = Math.max(
            1.0,
            Math.min(5.0, baseScore + (Math.random() - 0.5) * variance),
          );
          const vocabScore = Math.max(
            1.0,
            Math.min(5.0, baseScore + (Math.random() - 0.5) * variance),
          );
          const grammarScore = hasErrors
            ? Math.max(1.0, Math.min(5.0, baseScore - 0.5 + (Math.random() - 0.5) * variance))
            : Math.max(1.0, Math.min(5.0, baseScore + (Math.random() - 0.5) * variance));
          const overallScore = (taScore + ccScore + vocabScore + grammarScore) / 4;

          // Determine CEFR level based on overall score
          let label: string;
          if (overallScore >= 4.5) label = "C1";
          else if (overallScore >= 4.0) label = "B2";
          else if (overallScore >= 3.5) label = "B1";
          else if (overallScore >= 3.0) label = "A2";
          else label = "A1";

          return {
            answer_id: answer.id,
            scores: {
              TA: Math.round(taScore * 10) / 10,
              CC: Math.round(ccScore * 10) / 10,
              Vocab: Math.round(vocabScore * 10) / 10,
              Grammar: Math.round(grammarScore * 10) / 10,
              Overall: Math.round(overallScore * 10) / 10,
            },
            label,
          };
        }),
      })),
    };

    return new Response(JSON.stringify(mockResult), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }

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
            },
            replacements: [{ value: errorPattern.replacement }],
            issueType: errorPattern.issueType as "error" | "warning",
          });
        }

        lastIndex = matchOffset + matchLength;
        if (lastIndex >= text.length) break;
      }
    }

    const mockResult: LanguageToolResponse = {
      matches,
    };

    return new Response(JSON.stringify(mockResult), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }

  async scoreCorpus(text: string): Promise<Response> {
    // Mock corpus CEFR scoring based on word count and complexity
    if (!text || typeof text !== "string") {
      return new Response(JSON.stringify({ error: "Invalid text input" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    const wordCount = text.split(/\s+/).filter(Boolean).length;

    // Simple heuristic for CEFR level
    let score = 3.0; // Base A2
    if (wordCount > 250)
      score = 7.5; // C1
    else if (wordCount > 200)
      score = 6.0; // B2
    else if (wordCount > 150)
      score = 5.0; // B1+
    else if (wordCount > 100)
      score = 4.5; // B1
    else if (wordCount > 50) score = 3.5; // A2+

    // Determine CEFR level
    let cefr = "A2";
    if (score >= 8.0) cefr = "C1+";
    else if (score >= 7.5) cefr = "C1";
    else if (score >= 6.5) cefr = "B2+";
    else if (score >= 6.0) cefr = "B2";
    else if (score >= 5.0) cefr = "B1+";
    else if (score >= 4.5) cefr = "B1";
    else if (score >= 3.5) cefr = "A2+";
    else if (score >= 3.0) cefr = "A2";

    return new Response(
      JSON.stringify({
        score: Math.round(score * 100) / 100,
        cefr_level: cefr,
        model: "corpus-roberta-mock",
      }),
      {
        status: 200,
        headers: { "Content-Type": "application/json" },
      },
    );
  }
}
