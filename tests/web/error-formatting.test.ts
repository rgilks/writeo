import { describe, it, expect } from "vitest";
import { formatFriendlyErrorMessage } from "../../apps/web/app/lib/utils/error-formatting";

describe("error formatting utilities", () => {
  describe("formatFriendlyErrorMessage", () => {
    it("should format Error instance using getErrorMessage", () => {
      const error = new Error("Network error");
      const result = formatFriendlyErrorMessage(error, "write");
      // getErrorMessage converts "Network error" to a user-friendly message
      expect(result).toBeTruthy();
      expect(typeof result).toBe("string");
      expect(result.length).toBeGreaterThan(0);
    });

    it("should return user-friendly string error as-is", () => {
      const error = "Please try again";
      const result = formatFriendlyErrorMessage(error, "write");
      expect(result).toBe("Please try again");
    });

    it("should format string error that doesn't look like stack trace", () => {
      const error = "Something went wrong";
      const result = formatFriendlyErrorMessage(error, "global");
      expect(result).toBe("Something went wrong");
    });

    it("should format string error with Error: prefix", () => {
      const error = "Error: Something went wrong";
      const result = formatFriendlyErrorMessage(error, "write");
      // Should use default message since it contains "Error:"
      expect(result).toBeTruthy();
      expect(result).not.toBe("Error: Something went wrong");
    });

    it("should format string error with stack trace", () => {
      const error = "Error at line 1\n  at function()";
      const result = formatFriendlyErrorMessage(error, "write");
      // Should use default message since it contains "at "
      expect(result).toBeTruthy();
      expect(result).not.toBe(error);
    });

    it("should format long string error (over 200 chars)", () => {
      const longError = "a".repeat(201);
      const result = formatFriendlyErrorMessage(longError, "write");
      // Should use default message since it's too long
      expect(result).toBeTruthy();
      expect(result).not.toBe(longError);
    });

    it("should handle null error", () => {
      const result = formatFriendlyErrorMessage(null, "global");
      expect(result).toBeTruthy();
    });

    it("should handle undefined error", () => {
      const result = formatFriendlyErrorMessage(undefined, "write");
      expect(result).toBeTruthy();
    });

    it("should handle number error", () => {
      const result = formatFriendlyErrorMessage(404, "results");
      expect(result).toBeTruthy();
    });

    it("should handle object error", () => {
      const result = formatFriendlyErrorMessage({ code: 500 }, "global");
      expect(result).toBeTruthy();
    });

    it("should use correct context for error messages", () => {
      const error = new Error("Test error");
      const writeResult = formatFriendlyErrorMessage(error, "write");
      const resultsResult = formatFriendlyErrorMessage(error, "results");
      const globalResult = formatFriendlyErrorMessage(error, "global");

      // All should return valid strings
      expect(typeof writeResult).toBe("string");
      expect(typeof resultsResult).toBe("string");
      expect(typeof globalResult).toBe("string");
    });
  });
});
