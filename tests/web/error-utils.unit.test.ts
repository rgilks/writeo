import { describe, it, expect } from "vitest";
import {
  getErrorType,
  groupErrorsByType,
  getErrorCountByType,
  getErrorSeverityColor,
  getErrorCategoryIcon,
  formatErrorMessage,
  getLearningTipForErrorType,
} from "../../apps/web/app/lib/utils/error-utils";
import type { LanguageToolError } from "@writeo/shared";

describe("error utilities", () => {
  describe("getErrorType", () => {
    it("should return errorType if present", () => {
      const error: LanguageToolError = {
        start: 0,
        end: 5,
        length: 5,
        category: "GRAMMAR",
        rule_id: "test",
        message: "Test error",
        source: "LT",
        severity: "error",
        errorType: "Subject-verb agreement",
      };
      expect(getErrorType(error)).toBe("Subject-verb agreement");
    });

    it("should return category if errorType is not present", () => {
      const error: LanguageToolError = {
        start: 0,
        end: 5,
        length: 5,
        category: "GRAMMAR",
        rule_id: "test",
        message: "Test error",
        source: "LT",
        severity: "error",
      };
      expect(getErrorType(error)).toBe("GRAMMAR");
    });

    it("should return 'Other' if neither errorType nor category is present", () => {
      const error: LanguageToolError = {
        start: 0,
        end: 5,
        length: 5,
        category: "",
        rule_id: "test",
        message: "Test error",
        source: "LT",
        severity: "error",
      };
      expect(getErrorType(error)).toBe("Other");
    });
  });

  describe("groupErrorsByType", () => {
    it("should group errors by type", () => {
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

      const grouped = groupErrorsByType(errors);
      expect(grouped["Subject-verb agreement"]).toHaveLength(2);
      expect(grouped["SPELLING"]).toHaveLength(1);
    });

    it("should return empty object for empty array", () => {
      const grouped = groupErrorsByType([]);
      expect(grouped).toEqual({});
    });
  });

  describe("getErrorCountByType", () => {
    it("should count errors by type", () => {
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

      const counts = getErrorCountByType(errors);
      expect(counts["Subject-verb agreement"]).toBe(2);
      expect(counts["SPELLING"]).toBe(1);
    });

    it("should return empty object for empty array", () => {
      const counts = getErrorCountByType([]);
      expect(counts).toEqual({});
    });
  });

  describe("getErrorSeverityColor", () => {
    it("should return orange for low confidence errors", () => {
      const error: LanguageToolError = {
        start: 0,
        end: 5,
        length: 5,
        category: "GRAMMAR",
        rule_id: "test",
        message: "Test error",
        source: "LT",
        severity: "error",
        highConfidence: false,
      };
      expect(getErrorSeverityColor(error)).toBe("#f59e0b");
    });

    it("should return red for error severity", () => {
      const error: LanguageToolError = {
        start: 0,
        end: 5,
        length: 5,
        category: "GRAMMAR",
        rule_id: "test",
        message: "Test error",
        source: "LT",
        severity: "error",
      };
      expect(getErrorSeverityColor(error)).toBe("#ef4444");
    });

    it("should return orange for warning severity", () => {
      const error: LanguageToolError = {
        start: 0,
        end: 5,
        length: 5,
        category: "GRAMMAR",
        rule_id: "test",
        message: "Test error",
        source: "LT",
        severity: "warning",
      };
      expect(getErrorSeverityColor(error)).toBe("#f59e0b");
    });

    it("should return gray for default/unknown severity", () => {
      const error: LanguageToolError = {
        start: 0,
        end: 5,
        length: 5,
        category: "GRAMMAR",
        rule_id: "test",
        message: "Test error",
        source: "LT",
        severity: "info" as any,
      };
      expect(getErrorSeverityColor(error)).toBe("#6b7280");
    });
  });

  describe("getErrorCategoryIcon", () => {
    it("should return correct icons for known categories", () => {
      expect(getErrorCategoryIcon("GRAMMAR")).toBe("📝");
      expect(getErrorCategoryIcon("SPELLING")).toBe("✏️");
      expect(getErrorCategoryIcon("TYPOS")).toBe("🔤");
      expect(getErrorCategoryIcon("STYLE")).toBe("✨");
      expect(getErrorCategoryIcon("PUNCTUATION")).toBe(".");
    });

    it("should be case-insensitive", () => {
      expect(getErrorCategoryIcon("grammar")).toBe("📝");
      expect(getErrorCategoryIcon("Grammar")).toBe("📝");
    });

    it("should return OTHER icon for unknown categories", () => {
      expect(getErrorCategoryIcon("UNKNOWN")).toBe("⚠️");
      expect(getErrorCategoryIcon("")).toBe("⚠️");
    });
  });

  describe("formatErrorMessage", () => {
    it("should return message if present", () => {
      const error: LanguageToolError = {
        start: 0,
        end: 5,
        length: 5,
        category: "GRAMMAR",
        rule_id: "test",
        message: "Test error message",
        source: "LT",
        severity: "error",
      };
      expect(formatErrorMessage(error)).toBe("Test error message");
    });

    it("should format errorType if message is not present", () => {
      const error: LanguageToolError = {
        start: 0,
        end: 5,
        length: 5,
        category: "GRAMMAR",
        rule_id: "test",
        message: "",
        source: "LT",
        severity: "error",
        errorType: "Subject-verb agreement",
      };
      expect(formatErrorMessage(error)).toContain("Subject-verb agreement");
      expect(formatErrorMessage(error)).toContain("error");
    });

    it("should format category if errorType and message are not present", () => {
      const error: LanguageToolError = {
        start: 0,
        end: 5,
        length: 5,
        category: "GRAMMAR",
        rule_id: "test",
        message: "",
        source: "LT",
        severity: "error",
      };
      expect(formatErrorMessage(error)).toContain("GRAMMAR");
      expect(formatErrorMessage(error)).toContain("issue");
    });

    it("should return default message if nothing is present", () => {
      const error: LanguageToolError = {
        start: 0,
        end: 5,
        length: 5,
        category: "",
        rule_id: "test",
        message: "",
        source: "LT",
        severity: "error",
      };
      expect(formatErrorMessage(error)).toBe("Error detected");
    });
  });

  describe("getLearningTipForErrorType", () => {
    it("should return tips for known error types", () => {
      expect(getLearningTipForErrorType("GRAMMAR")).toContain("grammar rules");
      expect(getLearningTipForErrorType("SPELLING")).toContain("dictionary");
      expect(getLearningTipForErrorType("TYPOS")).toContain("carefully");
      expect(getLearningTipForErrorType("STYLE")).toContain("varied");
      expect(getLearningTipForErrorType("PUNCTUATION")).toContain("punctuation");
    });

    it("should be case-insensitive", () => {
      expect(getLearningTipForErrorType("grammar")).toBeTruthy();
      expect(getLearningTipForErrorType("Grammar")).toBeTruthy();
    });

    it("should return null for unknown error types", () => {
      expect(getLearningTipForErrorType("UNKNOWN")).toBeNull();
      expect(getLearningTipForErrorType("")).toBeNull();
    });
  });
});
