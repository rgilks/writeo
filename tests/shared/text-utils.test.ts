/**
 * Unit tests for shared text utility functions
 */

import { describe, it, expect } from "vitest";
import { countWords } from "../../packages/shared/ts/text-utils";

describe("countWords", () => {
  it("should count words in simple text", () => {
    expect(countWords("Hello world")).toBe(2);
  });

  it("should count words with multiple spaces", () => {
    expect(countWords("Hello   world")).toBe(2);
  });

  it("should trim leading and trailing whitespace", () => {
    expect(countWords("  Hello world  ")).toBe(2);
  });

  it("should handle tabs and newlines", () => {
    expect(countWords("Hello\tworld\nfoo")).toBe(3);
  });

  it("should return 0 for empty string", () => {
    expect(countWords("")).toBe(0);
  });

  it("should return 0 for whitespace-only string", () => {
    expect(countWords("   \n\t  ")).toBe(0);
  });

  it("should handle single word", () => {
    expect(countWords("Hello")).toBe(1);
  });

  it("should handle multiple words", () => {
    expect(countWords("The quick brown fox jumps over the lazy dog")).toBe(9);
  });

  it("should handle words with punctuation", () => {
    expect(countWords("Hello, world! How are you?")).toBe(5);
  });

  it("should handle words with numbers", () => {
    expect(countWords("I have 3 cats and 2 dogs")).toBe(7); // Numbers are counted as words
  });

  it("should handle mixed whitespace", () => {
    expect(countWords("Hello\t  world\n  foo")).toBe(3);
  });

  it("should return 0 for null", () => {
    expect(countWords(null as any)).toBe(0);
  });

  it("should return 0 for undefined", () => {
    expect(countWords(undefined as any)).toBe(0);
  });

  it("should return 0 for non-string input", () => {
    expect(countWords(123 as any)).toBe(0);
    expect(countWords({} as any)).toBe(0);
    expect(countWords([] as any)).toBe(0);
  });

  it("should handle empty words between delimiters", () => {
    expect(countWords("Hello  world")).toBe(2);
  });

  it("should handle unicode characters", () => {
    expect(countWords("Hello 世界 world")).toBe(3);
  });

  it("should handle emoji", () => {
    expect(countWords("Hello 😀 world 🌍")).toBe(4); // Emojis are counted as separate words
  });

  it("should handle very long text", () => {
    const longText = "word ".repeat(1000);
    expect(countWords(longText)).toBe(1000);
  });

  it("should handle text with only punctuation", () => {
    expect(countWords("... !!! ???")).toBe(3); // Punctuation marks separated by spaces are counted as words
  });

  it("should handle text with mixed punctuation and words", () => {
    expect(countWords("Hello, world! How are you? I'm fine.")).toBe(7);
  });
});
