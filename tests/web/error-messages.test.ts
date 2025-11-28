import { describe, it, expect } from "vitest";
import {
  getErrorMessage,
  DEFAULT_ERROR_MESSAGES,
} from "../../apps/web/app/lib/utils/error-messages";

describe("error messages utilities", () => {
  describe("getErrorMessage", () => {
    it("should return default message for empty error message", () => {
      const error = new Error("");
      const result = getErrorMessage(error, "global");
      expect(result).toBe(DEFAULT_ERROR_MESSAGES.global);
    });

    it("should return default message for error without message", () => {
      const error = new Error();
      const result = getErrorMessage(error, "write");
      expect(result).toBe(DEFAULT_ERROR_MESSAGES.write);
    });

    it("should handle Server Component errors", () => {
      const error = new Error("Server Components render error");
      const result = getErrorMessage(error, "write");
      expect(result).toContain("processing your submission");
    });

    it("should handle configuration errors", () => {
      const error = new Error("API_KEY is missing");
      const result = getErrorMessage(error, "write");
      expect(result).toContain("configuration issue");
    });

    it("should handle network errors", () => {
      const error = new Error("Failed to fetch");
      const result = getErrorMessage(error, "write");
      expect(result).toContain("internet connection");
    });

    it("should handle timeout errors", () => {
      const error = new Error("Request timed out");
      const result = getErrorMessage(error, "global");
      expect(result).toContain("too long");
    });

    it("should handle server configuration errors", () => {
      const error = new Error("Server configuration error");
      const result = getErrorMessage(error, "global");
      expect(result).toContain("server configuration");
    });

    it("should handle not found errors in results context", () => {
      const error = new Error("Not found");
      const result = getErrorMessage(error, "results");
      expect(result).toContain("couldn't find the results");
    });

    it("should return user-friendly message if message is short and clean", () => {
      const error = new Error("Please try again later");
      const result = getErrorMessage(error, "global");
      expect(result).toBe("Please try again later");
    });

    it("should return default message for stack trace", () => {
      const error = new Error("Error at line 1\n  at function()");
      const result = getErrorMessage(error, "write");
      expect(result).toBe(DEFAULT_ERROR_MESSAGES.write);
    });

    it("should return default message for long error messages", () => {
      const longMessage = "a".repeat(201);
      const error = new Error(longMessage);
      const result = getErrorMessage(error, "global");
      expect(result).toBe(DEFAULT_ERROR_MESSAGES.global);
    });

    it("should use correct context for default messages", () => {
      const error = new Error("");
      expect(getErrorMessage(error, "global")).toBe(DEFAULT_ERROR_MESSAGES.global);
      expect(getErrorMessage(error, "write")).toBe(DEFAULT_ERROR_MESSAGES.write);
      expect(getErrorMessage(error, "results")).toBe(DEFAULT_ERROR_MESSAGES.results);
    });
  });
});
