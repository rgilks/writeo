/**
 * Parsing utilities for AI assessment responses
 */

import type { LLMErrorInput } from "./validation";

export function parsePipeDelimitedResponse(text: string): LLMErrorInput[] {
  const trimmedText = text.trim();

  if (trimmedText === "NO_ERRORS" || trimmedText.toLowerCase().includes("no errors")) {
    return [];
  }

  const lines = trimmedText.split("\n").filter((line) => line.trim().length > 0);
  const errors: LLMErrorInput[] = [];

  for (const line of lines) {
    const trimmedLine = line.trim();
    if (
      !trimmedLine ||
      trimmedLine.startsWith("#") ||
      trimmedLine.toLowerCase().includes("format:") ||
      trimmedLine.toLowerCase().includes("example:")
    ) {
      continue;
    }

    const parts = trimmedLine.split("|").map((p) => p.trim());

    if (parts.length < 9) {
      console.warn(
        `[getLLMAssessment] Skipping malformed line (expected 9 parts, got ${parts.length}):`,
        trimmedLine.substring(0, 100)
      );
      continue;
    }

    const [
      errorText,
      wordBefore,
      wordAfter,
      category,
      message,
      suggestionsStr,
      errorType,
      explanation,
      severity,
    ] = parts;

    if (!errorText || errorText.trim().length === 0) {
      console.warn(
        `[getLLMAssessment] Skipping line with empty errorText:`,
        trimmedLine.substring(0, 100)
      );
      continue;
    }

    const suggestions = (suggestionsStr || "")
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s.length > 0);

    errors.push({
      errorText: errorText.trim(),
      wordBefore: wordBefore && wordBefore.trim().length > 0 ? wordBefore.trim() : null,
      wordAfter: wordAfter && wordAfter.trim().length > 0 ? wordAfter.trim() : null,
      category: category || "GRAMMAR",
      message: message || "Error detected",
      suggestions,
      errorType: errorType || "Grammar error",
      explanation: explanation || message || "Error detected",
      severity: (severity === "warning" ? "warning" : "error") as "warning" | "error",
    });
  }

  return errors;
}
