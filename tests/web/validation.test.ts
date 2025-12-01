import { describe, it, expect } from "vitest";
import {
  validateEssayAnswer,
  validateWordCount,
  validateAssessmentResults,
  validateSubmissionResponse,
} from "../../apps/web/app/lib/utils/validation";

describe("validation utilities", () => {
  describe("validateEssayAnswer", () => {
    it("should return invalid for empty string", () => {
      const result = validateEssayAnswer("");
      expect(result.isValid).toBe(false);
      expect(result.error).toContain("Please write your essay");
    });

    it("should return invalid for whitespace-only string", () => {
      const result = validateEssayAnswer("   \n\t  ");
      expect(result.isValid).toBe(false);
      expect(result.error).toContain("Please write your essay");
    });

    it("should return valid for non-empty string", () => {
      const result = validateEssayAnswer("This is a valid essay.");
      expect(result.isValid).toBe(true);
      expect(result.error).toBeNull();
    });

    it("should return valid for string with content", () => {
      const result = validateEssayAnswer("Hello world");
      expect(result.isValid).toBe(true);
      expect(result.error).toBeNull();
    });
  });

  describe("validateWordCount", () => {
    it("should return invalid when word count is below minimum", () => {
      const result = validateWordCount(100, 250, 500);
      expect(result.isValid).toBe(false);
      expect(result.error).toContain("too short");
      expect(result.error).toContain("100");
      expect(result.error).toContain("250");
    });

    it("should return invalid when word count is above maximum", () => {
      const result = validateWordCount(600, 250, 500);
      expect(result.isValid).toBe(false);
      expect(result.error).toContain("too long");
      expect(result.error).toContain("600");
      expect(result.error).toContain("500");
    });

    it("should return valid when word count is within range", () => {
      const result = validateWordCount(300, 250, 500);
      expect(result.isValid).toBe(true);
      expect(result.error).toBeNull();
    });

    it("should return valid when word count equals minimum", () => {
      const result = validateWordCount(250, 250, 500);
      expect(result.isValid).toBe(true);
      expect(result.error).toBeNull();
    });

    it("should return valid when word count equals maximum", () => {
      const result = validateWordCount(500, 250, 500);
      expect(result.isValid).toBe(true);
      expect(result.error).toBeNull();
    });

    it("should handle edge case with zero words", () => {
      const result = validateWordCount(0, 250, 500);
      expect(result.isValid).toBe(false);
      expect(result.error).toContain("too short");
    });
  });

  describe("validateAssessmentResults", () => {
    it("should return invalid for null", () => {
      const result = validateAssessmentResults(null);
      expect(result.isValid).toBe(false);
      expect(result.error).toBe(
        "Invalid results format: missing required fields (status, template)",
      );
    });

    it("should return invalid for undefined", () => {
      const result = validateAssessmentResults(undefined);
      expect(result.isValid).toBe(false);
      expect(result.error).toBe(
        "Invalid results format: missing required fields (status, template)",
      );
    });

    it("should return invalid for non-object", () => {
      const result = validateAssessmentResults("not an object");
      expect(result.isValid).toBe(false);
      expect(result.error).toBe(
        "Invalid results format: missing required fields (status, template)",
      );
    });

    it("should return invalid for object without status", () => {
      const result = validateAssessmentResults({ template: { name: "test" } });
      expect(result.isValid).toBe(false);
      expect(result.error).toBe(
        "Invalid results format: missing required fields (status, template)",
      );
    });

    it("should return invalid for object without template", () => {
      const result = validateAssessmentResults({ status: "success" });
      expect(result.isValid).toBe(false);
      expect(result.error).toBe(
        "Invalid results format: missing required fields (status, template)",
      );
    });

    it("should return valid for object with status and template", () => {
      const result = validateAssessmentResults({
        status: "success",
        template: { name: "test", version: 1 },
      });
      expect(result.isValid).toBe(true);
      expect(result.error).toBeNull();
    });

    it("should return valid for object with additional properties", () => {
      const result = validateAssessmentResults({
        status: "success",
        template: { name: "test", version: 1 },
        results: { parts: [] },
        meta: {},
      });
      expect(result.isValid).toBe(true);
      expect(result.error).toBeNull();
    });
  });

  describe("validateSubmissionResponse", () => {
    it("should return invalid when submissionId is missing", () => {
      const result = validateSubmissionResponse({
        results: { status: "success", template: { name: "test" } },
      });
      expect(result.isValid).toBe(false);
      expect(result.error).toBe("No submission ID or results returned");
    });

    it("should return invalid when results is missing", () => {
      const result = validateSubmissionResponse({
        submissionId: "test-id",
      });
      expect(result.isValid).toBe(false);
      expect(result.error).toBe("No submission ID or results returned");
    });

    it("should return invalid when both are missing", () => {
      const result = validateSubmissionResponse({});
      expect(result.isValid).toBe(false);
      expect(result.error).toBe("No submission ID or results returned");
    });

    it("should return invalid when results format is invalid", () => {
      const result = validateSubmissionResponse({
        submissionId: "test-id",
        results: { invalid: "format" },
      });
      expect(result.isValid).toBe(false);
      expect(result.error).toBe(
        "Invalid results format: missing required fields (status, template)",
      );
    });

    it("should return valid when both submissionId and valid results are present", () => {
      const result = validateSubmissionResponse({
        submissionId: "test-id",
        results: {
          status: "success",
          template: { name: "test", version: 1 },
        },
      });
      expect(result.isValid).toBe(true);
      expect(result.error).toBeNull();
    });
  });
});
