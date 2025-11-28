/**
 * Unit tests for shared validation utilities
 */

import { describe, it, expect } from "vitest";
import { validateWordCount } from "../../packages/shared/ts/validation";
import { MIN_ESSAY_WORDS, MAX_ESSAY_WORDS } from "../../packages/shared/ts/constants";

describe("validateWordCount", () => {
  it.each([
    [300, undefined, undefined, true],
    [MIN_ESSAY_WORDS, undefined, undefined, true],
    [MAX_ESSAY_WORDS, undefined, undefined, true],
  ])(
    "should return valid for word count: count=%d, min=%s, max=%s",
    (count, min, max, expectedValid) => {
      const result = validateWordCount(count, min, max);
      expect(result.valid).toBe(expectedValid);
      expect(result.error).toBeUndefined();
    },
  );

  it.each([
    [MIN_ESSAY_WORDS - 1, undefined, undefined, "too short", String(MIN_ESSAY_WORDS)],
    [MAX_ESSAY_WORDS + 1, undefined, undefined, "too long", String(MAX_ESSAY_WORDS)],
    [0, undefined, undefined, "too short", undefined],
    [-10, undefined, undefined, "too short", undefined],
  ])(
    "should return invalid for word count: count=%d, error contains %s",
    (count, min, max, errorContains, errorContains2) => {
      const result = validateWordCount(count, min, max);
      expect(result.valid).toBe(false);
      expect(result.error).toBeDefined();
      expect(result.error).toContain(errorContains);
      if (errorContains2) {
        expect(result.error).toContain(errorContains2);
      }
    },
  );

  it.each([
    [100, 200, 500, false, "200"],
    [600, 250, 500, false, "500"],
    [300, 200, 400, true, undefined],
  ])(
    "should use custom range: count=%d, min=%d, max=%d, valid=%s",
    (count, min, max, expectedValid, errorContains) => {
      const result = validateWordCount(count, min, max);
      expect(result.valid).toBe(expectedValid);
      if (expectedValid) {
        expect(result.error).toBeUndefined();
      } else {
        expect(result.error).toContain(errorContains);
      }
    },
  );

  it("should include current word count in error message", () => {
    const result = validateWordCount(100);
    expect(result.error).toContain("100");
  });

  it.each([
    [MIN_ESSAY_WORDS - 1, "min - 1"],
    [MAX_ESSAY_WORDS + 1, "max + 1"],
  ])("should handle edge case at boundary: count=%d (%s)", (count, description) => {
    const result = validateWordCount(count);
    expect(result.valid).toBe(false);
  });

  it("should handle very large word count", () => {
    const result = validateWordCount(10000);
    expect(result.valid).toBe(false);
    expect(result.error).toContain("too long");
  });
});
