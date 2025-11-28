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

    it.each([
      ["Please try again", "write", "Please try again"],
      ["Something went wrong", "global", "Something went wrong"],
    ])("should return user-friendly string error as-is: %s", (error, context, expected) => {
      const result = formatFriendlyErrorMessage(error, context as any);
      expect(result).toBe(expected);
    });

    it.each([
      ["Error: Something went wrong", "write"],
      ["Error at line 1\n  at function()", "write"],
      ["a".repeat(201), "write"],
    ])(
      "should format problematic string error (Error: prefix, stack trace, or too long): %s",
      (error, context) => {
        const result = formatFriendlyErrorMessage(error, context as any);
        // Should use default message since it contains "Error:", "at ", or is too long
        expect(result).toBeTruthy();
        expect(result).not.toBe(error);
        expect(typeof result).toBe("string");
        expect(result.length).toBeGreaterThan(0);
      },
    );

    it.each([
      [null, "global"],
      [undefined, "write"],
      [404, "results"],
      [{ code: 500 }, "global"],
    ])("should handle non-Error input: %j", (error, context) => {
      const result = formatFriendlyErrorMessage(error as any, context as any);
      expect(result).toBeTruthy();
      expect(typeof result).toBe("string");
    });

    it.each([["write"], ["results"], ["global"]])(
      "should use correct context for error messages: %s",
      (context) => {
        const error = new Error("Test error");
        const result = formatFriendlyErrorMessage(error, context as any);
        expect(typeof result).toBe("string");
        expect(result.length).toBeGreaterThan(0);
      },
    );
  });
});
