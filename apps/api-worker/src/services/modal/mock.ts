import type { ModalRequest, LanguageToolResponse } from "@writeo/shared";
import type { ModalService } from "./types";
import type { EssayResult } from "../clients/essay-client";

export class MockModalClient implements ModalService {
  async gradeEssay(request: ModalRequest): Promise<Response> {
    const mockResult: EssayResult = {
      submission_id: request.submission_id,
      parts: request.parts.map((part) => ({
        part: part.part,
        answers: part.answers.map((answer) => ({
          answer_id: answer.id,
          scores: {
            TA: 4.0,
            CC: 4.0,
            Vocab: 4.0,
            Grammar: 4.0,
            Overall: 4.0,
          },
          label: "B1",
        })),
      })),
    };

    return new Response(JSON.stringify(mockResult), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }

  async checkGrammar(text: string, _language: string, _answerId: string): Promise<Response> {
    // Return mock errors for common grammar patterns found in tests
    const matches: LanguageToolResponse["matches"] = [];

    // Detect common grammar errors for testing
    type ErrorPattern = {
      pattern: RegExp;
      offset: number | ((match: RegExpMatchArray) => number);
      length: number | ((match: RegExpMatchArray) => number);
      message: string;
      replacement: string;
    };
    const errorPatterns: ErrorPattern[] = [
      {
        pattern: /\bI goes\b/i,
        offset: (m: RegExpMatchArray) => m.index!,
        length: 7,
        message: "Use 'I go' instead of 'I goes'",
        replacement: "I go",
      },
      {
        pattern: /\bwe goes\b/i,
        offset: (m: RegExpMatchArray) => m.index!,
        length: 8,
        message: "Use 'we go' instead of 'we goes'",
        replacement: "we go",
      },
      {
        pattern: /\bwe was\b/i,
        offset: (m: RegExpMatchArray) => m.index!,
        length: 6,
        message: "Use 'we were' instead of 'we was'",
        replacement: "we were",
      },
      {
        pattern: /\bwe plays\b/i,
        offset: (m: RegExpMatchArray) => m.index!,
        length: 8,
        message: "Use 'we play' instead of 'we plays'",
        replacement: "we play",
      },
      {
        pattern: /\bI go\b.*yesterday/i,
        offset: (m: RegExpMatchArray) => m.index!,
        length: 4,
        message: "Use 'I went' for past tense",
        replacement: "I went",
      },
      {
        pattern: /\bgrammar error\b/i,
        offset: (m: RegExpMatchArray) => m.index!,
        length: 13,
        message: "Mock grammar error detected",
        replacement: "correction",
      },
    ];

    for (const errorPattern of errorPatterns) {
      const match = text.match(errorPattern.pattern);
      if (match) {
        const matchOffset =
          typeof errorPattern.offset === "function"
            ? errorPattern.offset(match)
            : errorPattern.offset;
        const matchLength =
          typeof errorPattern.length === "function"
            ? errorPattern.length(match)
            : errorPattern.length;
        matches.push({
          offset: matchOffset,
          length: matchLength,
          message: errorPattern.message,
          shortMessage: "Grammar Error",
          rule: {
            id: "MOCK_RULE",
            description: "Mock rule description",
            category: { id: "GRAMMAR", name: "Grammar" },
          },
          replacements: [{ value: errorPattern.replacement }],
          issueType: "error",
        });
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
}
