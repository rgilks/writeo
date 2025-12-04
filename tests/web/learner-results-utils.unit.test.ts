import { describe, it, expect } from "vitest";
import {
  getScoreColor,
  getScoreLabel,
  getCEFRDescriptor,
  mapScoreToCEFR,
  getCEFRThresholds,
  calculateCEFRProgress,
  getCEFRLabel,
  getErrorExplanation,
} from "../../apps/web/app/components/learner-results/utils";

describe("learner results utilities", () => {
  describe("getScoreColor", () => {
    it("should return green for excellent scores (>= 7.5)", () => {
      expect(getScoreColor(7.5)).toBe("#10b981");
      expect(getScoreColor(8.0)).toBe("#10b981");
      expect(getScoreColor(9.0)).toBe("#10b981");
    });

    it("should return blue for good scores (>= 6.5, < 7.5)", () => {
      expect(getScoreColor(6.5)).toBe("#3b82f6");
      expect(getScoreColor(7.0)).toBe("#3b82f6");
      expect(getScoreColor(7.4)).toBe("#3b82f6");
    });

    it("should return orange for fair scores (>= 5.5, < 6.5)", () => {
      expect(getScoreColor(5.5)).toBe("#f59e0b");
      expect(getScoreColor(6.0)).toBe("#f59e0b");
      expect(getScoreColor(6.4)).toBe("#f59e0b");
    });

    it("should return red for needs improvement scores (< 5.5)", () => {
      expect(getScoreColor(5.4)).toBe("#ef4444");
      expect(getScoreColor(4.0)).toBe("#ef4444");
      expect(getScoreColor(0)).toBe("#ef4444");
    });
  });

  describe("getScoreLabel", () => {
    it("should return 'Excellent' for scores >= 7.5", () => {
      expect(getScoreLabel(7.5)).toBe("Excellent");
      expect(getScoreLabel(9.0)).toBe("Excellent");
    });

    it("should return 'Good' for scores >= 6.5, < 7.5", () => {
      expect(getScoreLabel(6.5)).toBe("Good");
      expect(getScoreLabel(7.0)).toBe("Good");
    });

    it("should return 'Fair' for scores >= 5.5, < 6.5", () => {
      expect(getScoreLabel(5.5)).toBe("Fair");
      expect(getScoreLabel(6.0)).toBe("Fair");
    });

    it("should return 'Needs Improvement' for scores < 5.5", () => {
      expect(getScoreLabel(5.4)).toBe("Needs Improvement");
      expect(getScoreLabel(0)).toBe("Needs Improvement");
    });
  });

  describe("getCEFRDescriptor", () => {
    it("should return descriptor for valid CEFR levels", () => {
      expect(getCEFRDescriptor("A2")).toContain("simple connected text");
      expect(getCEFRDescriptor("B1")).toContain("familiar or of personal interest");
      expect(getCEFRDescriptor("B2")).toContain("wide range of subjects");
      expect(getCEFRDescriptor("C1")).toContain("complex subjects");
      expect(getCEFRDescriptor("C2")).toContain("appropriate style");
    });

    it("should return default message for invalid level", () => {
      expect(getCEFRDescriptor("INVALID")).toBe("Writing proficiency level.");
      expect(getCEFRDescriptor("")).toBe("Writing proficiency level.");
    });
  });

  describe("mapScoreToCEFR", () => {
    it("should return C2 for scores >= 8.5", () => {
      expect(mapScoreToCEFR(8.5)).toBe("C2");
      expect(mapScoreToCEFR(9.0)).toBe("C2");
    });

    it("should return C1 for scores >= 7.0, < 8.5", () => {
      expect(mapScoreToCEFR(7.0)).toBe("C1");
      expect(mapScoreToCEFR(8.4)).toBe("C1");
    });

    it("should return B2 for scores >= 5.5, < 7.0", () => {
      expect(mapScoreToCEFR(5.5)).toBe("B2");
      expect(mapScoreToCEFR(6.9)).toBe("B2");
    });

    it("should return B1 for scores >= 4.0, < 5.5", () => {
      expect(mapScoreToCEFR(4.0)).toBe("B1");
      expect(mapScoreToCEFR(5.4)).toBe("B1");
    });

    it("should return A2 for scores < 4.0", () => {
      expect(mapScoreToCEFR(3.9)).toBe("A2");
      expect(mapScoreToCEFR(0)).toBe("A2");
    });
  });

  describe("getCEFRThresholds", () => {
    it("should return correct thresholds for all levels", () => {
      const thresholds = getCEFRThresholds();
      expect(thresholds.A2).toEqual({ min: 0, max: 4.0 });
      expect(thresholds.B1).toEqual({ min: 4.0, max: 5.5 });
      expect(thresholds.B2).toEqual({ min: 5.5, max: 7.0 });
      expect(thresholds.C1).toEqual({ min: 7.0, max: 8.5 });
      expect(thresholds.C2).toEqual({ min: 8.5, max: 9.0 });
    });
  });

  describe("calculateCEFRProgress", () => {
    it("should return 100% progress for C2 level", () => {
      const result = calculateCEFRProgress(9.0);
      expect(result.current).toBe("C2");
      expect(result.progress).toBe(100);
      expect(result.scoreToNext).toBe(0);
    });

    it("should calculate progress for A2 level", () => {
      const result = calculateCEFRProgress(2.0);
      expect(result.current).toBe("A2");
      expect(result.next).toBe("B1");
      expect(result.progress).toBeGreaterThan(0);
      expect(result.progress).toBeLessThan(100);
      expect(result.scoreToNext).toBe(2.0); // 4.0 - 2.0
    });

    it("should calculate progress for B1 level", () => {
      const result = calculateCEFRProgress(5.0);
      expect(result.current).toBe("B1");
      expect(result.next).toBe("B2");
      expect(result.progress).toBeGreaterThan(0);
      expect(result.progress).toBeLessThan(100);
      expect(result.scoreToNext).toBe(0.5); // 5.5 - 5.0
    });

    it("should calculate progress for B2 level", () => {
      const result = calculateCEFRProgress(6.0);
      expect(result.current).toBe("B2");
      expect(result.next).toBe("C1");
      expect(result.progress).toBeGreaterThan(0);
      expect(result.progress).toBeLessThan(100);
      expect(result.scoreToNext).toBe(1.0); // 7.0 - 6.0
    });

    it("should calculate progress for C1 level", () => {
      const result = calculateCEFRProgress(7.5);
      expect(result.current).toBe("C1");
      expect(result.next).toBe("C2");
      expect(result.progress).toBeGreaterThan(0);
      expect(result.progress).toBeLessThan(100);
      expect(result.scoreToNext).toBe(1.0); // 8.5 - 7.5
    });

    it("should handle edge case at level boundary", () => {
      const result = calculateCEFRProgress(4.0);
      expect(result.current).toBe("B1");
      expect(result.progress).toBe(0);
      expect(result.scoreToNext).toBe(1.5); // 5.5 - 4.0
    });
  });

  describe("getCEFRLabel", () => {
    it("should return correct labels for valid levels", () => {
      expect(getCEFRLabel("A2")).toBe("Elementary");
      expect(getCEFRLabel("B1")).toBe("Intermediate");
      expect(getCEFRLabel("B2")).toBe("Upper Intermediate");
      expect(getCEFRLabel("C1")).toBe("Advanced");
      expect(getCEFRLabel("C2")).toBe("Proficient");
    });

    it("should return level as-is for invalid level", () => {
      expect(getCEFRLabel("INVALID")).toBe("INVALID");
      expect(getCEFRLabel("")).toBe("");
    });
  });

  describe("getErrorExplanation", () => {
    it("should return explanation for known error types", () => {
      expect(getErrorExplanation("Subject-verb agreement", 1)).toContain("subject and verb");
      expect(getErrorExplanation("Verb tense", 1)).toContain("consistent verb tenses");
      expect(getErrorExplanation("Article use", 1)).toContain("a' before consonant");
      expect(getErrorExplanation("Preposition", 1)).toContain("relationships");
      expect(getErrorExplanation("Spelling", 1)).toContain("spelling carefully");
      expect(getErrorExplanation("Punctuation", 1)).toContain("punctuation marks");
      expect(getErrorExplanation("Word order", 1)).toContain("Subject-Verb-Object");
      expect(getErrorExplanation("Grammar error", 1)).toContain("grammatical mistake");
    });

    it("should return generic message for unknown error types", () => {
      const result = getErrorExplanation("Unknown Error", 1);
      expect(result).toContain("appears 1 time");
    });

    it("should handle plural correctly", () => {
      expect(getErrorExplanation("Unknown Error", 1)).toContain("1 time");
      expect(getErrorExplanation("Unknown Error", 2)).toContain("2 times");
      expect(getErrorExplanation("Unknown Error", 0)).toContain("0 times");
    });
  });
});
