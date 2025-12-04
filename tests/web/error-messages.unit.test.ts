import { describe, it, expect } from "vitest";
import {
  getErrorMessage,
  DEFAULT_ERROR_MESSAGES,
} from "../../apps/web/app/lib/utils/error-messages";

describe("error messages utilities", () => {
  describe("getErrorMessage", () => {
    it.each([
      [new Error(""), "global", DEFAULT_ERROR_MESSAGES.global],
      [new Error(), "write", DEFAULT_ERROR_MESSAGES.write],
    ])(
      "should return default message for empty/undefined error message",
      (error, context, expected) => {
        const result = getErrorMessage(error, context as any);
        expect(result).toBe(expected);
      },
    );

    it.each([
      ["Server Components render error", "write", "processing your submission"],
      ["API_KEY is missing", "write", "configuration issue"],
      ["Failed to fetch", "write", "internet connection"],
      ["Request timed out", "global", "too long"],
      ["Server configuration error", "global", "server configuration"],
      ["Not found", "results", "couldn't find the results"],
    ])("should handle specific error types: %s", (errorMessage, context, expectedContain) => {
      const error = new Error(errorMessage);
      const result = getErrorMessage(error, context as any);
      expect(result).toContain(expectedContain);
    });

    it("should return user-friendly message if message is short and clean", () => {
      const error = new Error("Please try again later");
      const result = getErrorMessage(error, "global");
      expect(result).toBe("Please try again later");
    });

    it.each([
      ["Error at line 1\n  at function()", "write", DEFAULT_ERROR_MESSAGES.write],
      ["a".repeat(201), "global", DEFAULT_ERROR_MESSAGES.global],
    ])(
      "should return default message for problematic error messages: %s",
      (errorMessage, context, expected) => {
        const error = new Error(errorMessage);
        const result = getErrorMessage(error, context as any);
        expect(result).toBe(expected);
      },
    );

    it.each([
      ["global", DEFAULT_ERROR_MESSAGES.global],
      ["write", DEFAULT_ERROR_MESSAGES.write],
      ["results", DEFAULT_ERROR_MESSAGES.results],
    ])("should use correct default message for context: %s", (context, expectedMessage) => {
      const error = new Error("");
      expect(getErrorMessage(error, context as any)).toBe(expectedMessage);
    });
  });
});
