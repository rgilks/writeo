import { describe, it, expect } from "vitest";
import {
  calculateErrorReduction,
  calculateScoreImprovement,
  calculateWordCountChange,
  analyzeErrorTypeFrequency,
  getTopErrorTypes,
  calculateProgressMetrics,
  generateErrorId,
  extractErrorIds,
} from "../../apps/web/app/lib/utils/progress";
import type { DraftHistory, LanguageToolError } from "@writeo/shared";

describe("progress utilities", () => {
  const createDraftHistory = (overrides: Partial<DraftHistory>): DraftHistory => ({
    draftNumber: 1,
    submissionId: "test-id",
    timestamp: "2024-01-01T00:00:00Z",
    wordCount: 250,
    errorCount: 10,
    overallScore: 6.0,
    cefrLevel: "B2",
    errorIds: [],
    ...overrides,
  });

  describe("calculateErrorReduction", () => {
    it("should calculate error reduction when errors decreased", () => {
      const previous = createDraftHistory({ errorCount: 10 });
      const current = createDraftHistory({ errorCount: 5 });
      expect(calculateErrorReduction(previous, current)).toBe(5);
    });

    it("should return negative value when errors increased", () => {
      const previous = createDraftHistory({ errorCount: 5 });
      const current = createDraftHistory({ errorCount: 10 });
      expect(calculateErrorReduction(previous, current)).toBe(-5);
    });

    it("should return null when previous draft is null", () => {
      const current = createDraftHistory({ errorCount: 5 });
      expect(calculateErrorReduction(null, current)).toBeNull();
    });

    it("should return null when errorCount is undefined", () => {
      const previous = createDraftHistory({ errorCount: 10 });
      const current = createDraftHistory({ errorCount: undefined });
      expect(calculateErrorReduction(previous, current)).toBeNull();
    });
  });

  describe("calculateScoreImprovement", () => {
    it("should calculate score improvement when score increased", () => {
      const previous = createDraftHistory({ overallScore: 6.0 });
      const current = createDraftHistory({ overallScore: 7.0 });
      expect(calculateScoreImprovement(previous, current)).toBe(1.0);
    });

    it("should return negative value when score decreased", () => {
      const previous = createDraftHistory({ overallScore: 7.0 });
      const current = createDraftHistory({ overallScore: 6.0 });
      expect(calculateScoreImprovement(previous, current)).toBe(-1.0);
    });

    it("should return null when previous draft is null", () => {
      const current = createDraftHistory({ overallScore: 7.0 });
      expect(calculateScoreImprovement(null, current)).toBeNull();
    });

    it("should return null when overallScore is undefined", () => {
      const previous = createDraftHistory({ overallScore: 6.0 });
      const current = createDraftHistory({ overallScore: undefined });
      expect(calculateScoreImprovement(previous, current)).toBeNull();
    });
  });

  describe("calculateWordCountChange", () => {
    it("should calculate word count increase", () => {
      const previous = createDraftHistory({ wordCount: 250 });
      const current = createDraftHistory({ wordCount: 300 });
      expect(calculateWordCountChange(previous, current)).toBe(50);
    });

    it("should calculate word count decrease", () => {
      const previous = createDraftHistory({ wordCount: 300 });
      const current = createDraftHistory({ wordCount: 250 });
      expect(calculateWordCountChange(previous, current)).toBe(-50);
    });

    it("should return null when previous draft is null", () => {
      const current = createDraftHistory({ wordCount: 300 });
      expect(calculateWordCountChange(null, current)).toBeNull();
    });
  });

  describe("analyzeErrorTypeFrequency", () => {
    it("should analyze error type frequency", () => {
      const errors: LanguageToolError[] = [
        {
          start: 0,
          end: 5,
          length: 5,
          category: "GRAMMAR",
          rule_id: "test1",
          message: "Error 1",
          source: "LT",
          severity: "error",
          errorType: "Subject-verb agreement",
        },
        {
          start: 10,
          end: 15,
          length: 5,
          category: "GRAMMAR",
          rule_id: "test2",
          message: "Error 2",
          source: "LT",
          severity: "error",
          errorType: "Subject-verb agreement",
        },
        {
          start: 20,
          end: 25,
          length: 5,
          category: "SPELLING",
          rule_id: "test3",
          message: "Error 3",
          source: "LT",
          severity: "error",
        },
      ];

      const frequency = analyzeErrorTypeFrequency(errors);
      expect(frequency).toHaveLength(2);
      expect(frequency[0].type).toBe("Subject-verb agreement");
      expect(frequency[0].count).toBe(2);
      expect(frequency[1].type).toBe("SPELLING");
      expect(frequency[1].count).toBe(1);
    });

    it("should return empty array for empty errors", () => {
      expect(analyzeErrorTypeFrequency([])).toEqual([]);
    });

    it("should sort by count descending", () => {
      const errors: LanguageToolError[] = [
        {
          start: 0,
          end: 5,
          length: 5,
          category: "A",
          rule_id: "test1",
          message: "Error 1",
          source: "LT",
          severity: "error",
        },
        {
          start: 10,
          end: 15,
          length: 5,
          category: "B",
          rule_id: "test2",
          message: "Error 2",
          source: "LT",
          severity: "error",
        },
        {
          start: 20,
          end: 25,
          length: 5,
          category: "B",
          rule_id: "test3",
          message: "Error 3",
          source: "LT",
          severity: "error",
        },
        {
          start: 30,
          end: 35,
          length: 5,
          category: "B",
          rule_id: "test4",
          message: "Error 4",
          source: "LT",
          severity: "error",
        },
      ];

      const frequency = analyzeErrorTypeFrequency(errors);
      expect(frequency[0].type).toBe("B");
      expect(frequency[0].count).toBe(3);
      expect(frequency[1].type).toBe("A");
      expect(frequency[1].count).toBe(1);
    });
  });

  describe("getTopErrorTypes", () => {
    it("should return top N error types", () => {
      const errors: LanguageToolError[] = [
        {
          start: 0,
          end: 5,
          length: 5,
          category: "A",
          rule_id: "1",
          message: "A1",
          source: "LT",
          severity: "error",
        },
        {
          start: 10,
          end: 15,
          length: 5,
          category: "A",
          rule_id: "2",
          message: "A2",
          source: "LT",
          severity: "error",
        },
        {
          start: 20,
          end: 25,
          length: 5,
          category: "B",
          rule_id: "3",
          message: "B1",
          source: "LT",
          severity: "error",
        },
        {
          start: 30,
          end: 35,
          length: 5,
          category: "C",
          rule_id: "4",
          message: "C1",
          source: "LT",
          severity: "error",
        },
      ];

      const top = getTopErrorTypes(errors, 2);
      expect(top).toHaveLength(2);
      expect(top[0].type).toBe("A");
      expect(top[0].count).toBe(2);
    });

    it("should default to top 3", () => {
      const errors: LanguageToolError[] = Array.from({ length: 10 }, (_, i) => ({
        start: i * 10,
        end: i * 10 + 5,
        length: 5,
        category: `Type${i % 4}`,
        rule_id: `rule${i}`,
        message: `Error ${i}`,
        source: "LT" as const,
        severity: "error" as const,
      }));

      const top = getTopErrorTypes(errors);
      expect(top).toHaveLength(3);
    });
  });

  describe("calculateProgressMetrics", () => {
    it("should calculate progress metrics from draft history", () => {
      const draftHistory: DraftHistory[] = [
        createDraftHistory({ draftNumber: 1, overallScore: 6.0, errorCount: 10, wordCount: 250 }),
        createDraftHistory({ draftNumber: 2, overallScore: 7.0, errorCount: 5, wordCount: 300 }),
      ];

      const metrics = calculateProgressMetrics(draftHistory);
      expect(metrics).not.toBeNull();
      expect(metrics?.totalDrafts).toBe(2);
      expect(metrics?.firstDraftScore).toBe(6.0);
      expect(metrics?.latestDraftScore).toBe(7.0);
      expect(metrics?.scoreImprovement).toBe(1.0);
      expect(metrics?.errorReduction).toBe(5);
      expect(metrics?.wordCountChange).toBe(50);
    });

    it("should return null for empty draft history", () => {
      expect(calculateProgressMetrics([])).toBeNull();
    });

    it("should handle single draft", () => {
      const draftHistory: DraftHistory[] = [
        createDraftHistory({ draftNumber: 1, overallScore: 6.0, errorCount: 10, wordCount: 250 }),
      ];

      const metrics = calculateProgressMetrics(draftHistory);
      expect(metrics).not.toBeNull();
      expect(metrics?.totalDrafts).toBe(1);
      // When there's only one draft, first and latest are the same, so improvement is 0
      expect(metrics?.scoreImprovement).toBe(0);
      expect(metrics?.errorReduction).toBe(0);
      expect(metrics?.wordCountChange).toBe(0);
    });

    it("should handle undefined scores", () => {
      const draftHistory: DraftHistory[] = [
        createDraftHistory({ draftNumber: 1, overallScore: undefined, errorCount: undefined }),
        createDraftHistory({ draftNumber: 2, overallScore: undefined, errorCount: undefined }),
      ];

      const metrics = calculateProgressMetrics(draftHistory);
      expect(metrics).not.toBeNull();
      expect(metrics?.scoreImprovement).toBeUndefined();
      expect(metrics?.errorReduction).toBeUndefined();
    });
  });

  describe("generateErrorId", () => {
    it("should generate error ID from error properties", () => {
      const error: LanguageToolError = {
        start: 10,
        end: 15,
        length: 5,
        category: "GRAMMAR",
        rule_id: "test",
        message: "Test error message",
        source: "LT",
        severity: "error",
      };

      const id = generateErrorId(error);
      expect(id).toContain("10");
      expect(id).toContain("15");
      expect(id).toContain("Test error message");
    });

    it("should handle error without message", () => {
      const error: LanguageToolError = {
        start: 10,
        end: 15,
        length: 5,
        category: "GRAMMAR",
        rule_id: "test",
        message: "",
        source: "LT",
        severity: "error",
      };

      const id = generateErrorId(error);
      expect(id).toBe("10-15-");
    });
  });

  describe("extractErrorIds", () => {
    it("should extract error IDs from array of errors", () => {
      const errors: LanguageToolError[] = [
        {
          start: 0,
          end: 5,
          length: 5,
          category: "GRAMMAR",
          rule_id: "test1",
          message: "Error 1",
          source: "LT",
          severity: "error",
        },
        {
          start: 10,
          end: 15,
          length: 5,
          category: "SPELLING",
          rule_id: "test2",
          message: "Error 2",
          source: "LT",
          severity: "error",
        },
      ];

      const ids = extractErrorIds(errors);
      expect(ids).toHaveLength(2);
      expect(ids[0]).toContain("0-5");
      expect(ids[1]).toContain("10-15");
    });

    it("should return empty array for empty errors", () => {
      expect(extractErrorIds([])).toEqual([]);
    });
  });
});
