import { describe, it, expect, beforeEach, vi } from "vitest";
import { mergeQuestionTextIntoResults } from "../../apps/web/app/lib/utils/submission";
import type { AssessmentResults } from "@writeo/shared";

// Mock window object
const mockWindow = {
  location: { href: "http://localhost" },
};

describe("submission utilities", () => {
  beforeEach(() => {
    vi.stubGlobal("window", mockWindow);
  });

  describe("mergeQuestionTextIntoResults", () => {
    it("should return results unchanged when window is undefined", () => {
      vi.stubGlobal("window", undefined);
      const results: AssessmentResults = {
        status: "success",
        template: { name: "test", version: 1 },
      };
      const merged = mergeQuestionTextIntoResults(results, "What is your opinion?");
      expect(merged).toEqual(results);
    });

    it("should return results unchanged when questionText is empty", () => {
      const results: AssessmentResults = {
        status: "success",
        template: { name: "test", version: 1 },
        meta: {
          answerTexts: { "answer-1": "My answer" },
        },
      };
      const merged = mergeQuestionTextIntoResults(results, "");
      expect(merged).toEqual(results);
    });

    it("should return results unchanged when no answerTexts in meta", () => {
      const results: AssessmentResults = {
        status: "success",
        template: { name: "test", version: 1 },
        meta: {},
      };
      const merged = mergeQuestionTextIntoResults(results, "What is your opinion?");
      expect(merged).toEqual(results);
    });

    it("should create questionTexts when it doesn't exist", () => {
      const results: AssessmentResults = {
        status: "success",
        template: { name: "test", version: 1 },
        meta: {
          answerTexts: { "answer-1": "My answer" },
        },
      };
      const merged = mergeQuestionTextIntoResults(results, "What is your opinion?");
      expect(merged.meta?.questionTexts).toEqual({
        "answer-1": "What is your opinion?",
      });
      expect(merged.meta?.answerTexts).toEqual({ "answer-1": "My answer" });
    });

    it("should add questionText to existing questionTexts", () => {
      const results: AssessmentResults = {
        status: "success",
        template: { name: "test", version: 1 },
        meta: {
          answerTexts: { "answer-1": "My answer" },
          questionTexts: { "answer-2": "Another question" },
        },
      };
      const merged = mergeQuestionTextIntoResults(results, "What is your opinion?");
      expect(merged.meta?.questionTexts).toEqual({
        "answer-2": "Another question",
        "answer-1": "What is your opinion?",
      });
    });

    it("should not overwrite existing questionText for same answerId", () => {
      const results: AssessmentResults = {
        status: "success",
        template: { name: "test", version: 1 },
        meta: {
          answerTexts: { "answer-1": "My answer" },
          questionTexts: { "answer-1": "Existing question" },
        },
      };
      const merged = mergeQuestionTextIntoResults(results, "New question");
      expect(merged.meta?.questionTexts).toEqual({
        "answer-1": "Existing question",
      });
    });

    it("should handle multiple answerIds and only add questionText for first one", () => {
      const results: AssessmentResults = {
        status: "success",
        template: { name: "test", version: 1 },
        meta: {
          answerTexts: {
            "answer-1": "First answer",
            "answer-2": "Second answer",
          },
        },
      };
      const merged = mergeQuestionTextIntoResults(results, "What is your opinion?");
      // Should use first key from answerTexts
      const firstKey = Object.keys(results.meta!.answerTexts!)[0];
      expect(merged.meta?.questionTexts?.[firstKey]).toBe("What is your opinion?");
    });

    it("should preserve all other meta properties", () => {
      const results: AssessmentResults = {
        status: "success",
        template: { name: "test", version: 1 },
        meta: {
          answerTexts: { "answer-1": "My answer" },
          wordCount: 250,
          overallScore: 7.5,
          timestamp: "2024-01-01T00:00:00Z",
        },
      };
      const merged = mergeQuestionTextIntoResults(results, "What is your opinion?");
      expect(merged.meta?.wordCount).toBe(250);
      expect(merged.meta?.overallScore).toBe(7.5);
      expect(merged.meta?.timestamp).toBe("2024-01-01T00:00:00Z");
      expect(merged.meta?.questionTexts).toBeDefined();
    });
  });
});
