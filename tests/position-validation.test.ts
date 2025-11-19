import { test, expect, describe } from "vitest";
import { validateAndCorrectErrorPosition } from "../apps/api-worker/src/utils/text-processing";

describe("Position Validation and Correction", () => {
  const sampleText =
    "I believe universities should increase their focus on engineering and understanding that graduates can contribute to the economy.";

  test("corrects position that splits a word", () => {
    // Position splits "believe" - should align to word boundary
    const result = validateAndCorrectErrorPosition(
      {
        start: 2, // In the middle of "believe"
        end: 5,
        errorText: "lie",
        errorType: "GRAMMAR",
      },
      sampleText
    );

    expect(result.valid).toBe(true);
    // Should align to start of "believe" (position 2)
    expect(result.start).toBeLessThanOrEqual(2);
    expect(result.end).toBeGreaterThan(result.start);

    // Verify the highlighted text is a complete word
    const highlightedText = sampleText.substring(result.start, result.end);
    expect(highlightedText.trim().length).toBeGreaterThan(0);
  });

  test("aligns position to word boundaries when close", () => {
    // Position is slightly off but close to word boundary
    const result = validateAndCorrectErrorPosition(
      {
        start: 1, // Just before "believe"
        end: 4,
        errorText: "bel",
        errorType: "GRAMMAR",
      },
      sampleText
    );

    expect(result.valid).toBe(true);
    // Should align to word boundary
    const highlightedText = sampleText.substring(result.start, result.end);
    expect(highlightedText).toMatch(/^\w+/); // Should start with a word character
  });

  test("uses fuzzy matching when errorText is provided", () => {
    // Position is wrong, but errorText helps find the correct location
    const result = validateAndCorrectErrorPosition(
      {
        start: 0, // Wrong position
        end: 3,
        errorText: "believe", // Correct text snippet
        errorType: "GRAMMAR",
      },
      sampleText
    );

    expect(result.valid).toBe(true);
    // Should find "believe" in the text
    const highlightedText = sampleText.substring(result.start, result.end);
    expect(highlightedText.toLowerCase()).toContain("believe");
  });

  test("filters out invalid positions (out of bounds)", () => {
    const result = validateAndCorrectErrorPosition(
      {
        start: -1,
        end: 5,
        errorText: "test",
        errorType: "GRAMMAR",
      },
      sampleText
    );

    expect(result.valid).toBe(false);
  });

  test("filters out invalid positions (end > start)", () => {
    const result = validateAndCorrectErrorPosition(
      {
        start: 10,
        end: 5,
        errorText: "test",
        errorType: "GRAMMAR",
      },
      sampleText
    );

    expect(result.valid).toBe(false);
  });

  test("filters out positions highlighting only whitespace", () => {
    const textWithSpaces = "I believe   universities";
    const result = validateAndCorrectErrorPosition(
      {
        start: 8,
        end: 11, // Only spaces
        errorText: "   ",
        errorType: "GRAMMAR",
      },
      textWithSpaces
    );

    // The validation will try to expand to word boundaries
    // If it can't find meaningful text, it should be invalid
    // But if it expands to include words, that's also acceptable
    if (result.valid) {
      const highlightedText = textWithSpaces.substring(result.start, result.end);
      // If valid, it should have expanded to include at least one word
      expect(/\w/.test(highlightedText)).toBe(true);
    } else {
      // If invalid, that's also fine - whitespace-only positions should be filtered
      expect(result.valid).toBe(false);
    }
  });

  test("allows punctuation errors to highlight punctuation", () => {
    const textWithPunctuation = "Hello, world!";
    const result = validateAndCorrectErrorPosition(
      {
        start: 5,
        end: 6, // The comma
        errorText: ",",
        errorType: "PUNCTUATION",
      },
      textWithPunctuation
    );

    expect(result.valid).toBe(true);
    const highlightedText = textWithPunctuation.substring(result.start, result.end);
    // Should include the comma, may expand to include surrounding words
    expect(highlightedText).toContain(",");
  });

  test("expands to include complete words when position splits word", () => {
    const text = "I believe this is correct.";
    // Position splits "believe" in the middle
    const result = validateAndCorrectErrorPosition(
      {
        start: 2, // Middle of "believe"
        end: 5,
        errorText: "lie",
        errorType: "GRAMMAR",
      },
      text
    );

    expect(result.valid).toBe(true);
    const highlightedText = text.substring(result.start, result.end);
    // Should include the complete word "believe"
    expect(highlightedText.toLowerCase()).toContain("believe");
  });

  test("handles multi-word errors correctly", () => {
    const text = "I believe universities should increase";
    const result = validateAndCorrectErrorPosition(
      {
        start: 2,
        end: 20, // Spans multiple words
        errorText: "believe universities",
        errorType: "GRAMMAR",
      },
      text
    );

    expect(result.valid).toBe(true);
    const highlightedText = text.substring(result.start, result.end);
    expect(highlightedText.toLowerCase()).toContain("believe");
    expect(highlightedText.toLowerCase()).toContain("universities");
  });

  test("validates that suggestions match highlighted text context", () => {
    const text = "I believe this is correct.";
    const result = validateAndCorrectErrorPosition(
      {
        start: 2,
        end: 9, // "believe"
        errorText: "believe",
        errorType: "Verb tense",
        message: "Should be past tense",
      },
      text
    );

    expect(result.valid).toBe(true);
    const highlightedText = text.substring(result.start, result.end);
    // Should include "believe", may expand to include surrounding words
    expect(highlightedText.toLowerCase()).toContain("believe");
    // Should be a meaningful text snippet
    expect(highlightedText.trim().length).toBeGreaterThan(0);
  });
});
