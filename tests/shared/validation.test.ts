/**
 * Unit tests for shared validation utilities
 */

import { describe, it, expect } from "vitest";
import { validateWordCount } from "../../packages/shared/ts/validation";
import { MIN_ESSAY_WORDS, MAX_ESSAY_WORDS } from "../../packages/shared/ts/constants";

describe("validateWordCount", () => {
  it("should return valid for word count within default range", () => {
    const result = validateWordCount(300);
    expect(result.valid).toBe(true);
    expect(result.error).toBeUndefined();
  });

  it("should return valid for word count at minimum", () => {
    const result = validateWordCount(MIN_ESSAY_WORDS);
    expect(result.valid).toBe(true);
    expect(result.error).toBeUndefined();
  });

  it("should return valid for word count at maximum", () => {
    const result = validateWordCount(MAX_ESSAY_WORDS);
    expect(result.valid).toBe(true);
    expect(result.error).toBeUndefined();
  });

  it("should return invalid for word count below minimum", () => {
    const result = validateWordCount(MIN_ESSAY_WORDS - 1);
    expect(result.valid).toBe(false);
    expect(result.error).toBeDefined();
    expect(result.error).toContain("too short");
    expect(result.error).toContain(String(MIN_ESSAY_WORDS));
  });

  it("should return invalid for word count above maximum", () => {
    const result = validateWordCount(MAX_ESSAY_WORDS + 1);
    expect(result.valid).toBe(false);
    expect(result.error).toBeDefined();
    expect(result.error).toContain("too long");
    expect(result.error).toContain(String(MAX_ESSAY_WORDS));
  });

  it("should return invalid for zero words", () => {
    const result = validateWordCount(0);
    expect(result.valid).toBe(false);
    expect(result.error).toContain("too short");
  });

  it("should return invalid for negative word count", () => {
    const result = validateWordCount(-10);
    expect(result.valid).toBe(false);
    expect(result.error).toContain("too short");
  });

  it("should use custom minimum when provided", () => {
    const result = validateWordCount(100, 200, 500);
    expect(result.valid).toBe(false);
    expect(result.error).toContain("200");
  });

  it("should use custom maximum when provided", () => {
    const result = validateWordCount(600, 250, 500);
    expect(result.valid).toBe(false);
    expect(result.error).toContain("500");
  });

  it("should return valid for word count within custom range", () => {
    const result = validateWordCount(300, 200, 400);
    expect(result.valid).toBe(true);
    expect(result.error).toBeUndefined();
  });

  it("should include current word count in error message", () => {
    const result = validateWordCount(100);
    expect(result.error).toContain("100");
  });

  it("should handle edge case at boundary (min - 1)", () => {
    const result = validateWordCount(MIN_ESSAY_WORDS - 1);
    expect(result.valid).toBe(false);
  });

  it("should handle edge case at boundary (max + 1)", () => {
    const result = validateWordCount(MAX_ESSAY_WORDS + 1);
    expect(result.valid).toBe(false);
  });

  it("should handle very large word count", () => {
    const result = validateWordCount(10000);
    expect(result.valid).toBe(false);
    expect(result.error).toContain("too long");
  });
});
