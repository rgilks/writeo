/**
 * Parsing utilities for AI assessment responses
 */

import type { LLMErrorInput } from "./validation";

const NO_ERROR_TOKENS = ["no_errors", "no errors"];
const IGNORE_LINE_PATTERNS = ["#", "format:", "example:"];
const PIPE_FIELDS = [
  "errorText",
  "wordBefore",
  "wordAfter",
  "category",
  "message",
  "suggestions",
  "errorType",
  "explanation",
  "severity",
] as const;

const EXPECTED_PART_COUNT = PIPE_FIELDS.length;

const cleanOrNull = (value?: string): string | null => {
  const trimmed = value?.trim();
  return trimmed && trimmed.length > 0 ? trimmed : null;
};

const normalizeSeverity = (value?: string): "warning" | "error" => {
  return value?.trim().toLowerCase() === "warning" ? "warning" : "error";
};

const parseSuggestions = (suggestionsStr?: string): string[] => {
  return (suggestionsStr ?? "")
    .split(",")
    .map((suggestion) => suggestion.trim())
    .filter(Boolean);
};

const shouldIgnoreLine = (line: string): boolean => {
  if (!line) {
    return true;
  }

  const lowerLine = line.toLowerCase();
  return IGNORE_LINE_PATTERNS.some((pattern) => lowerLine.startsWith(pattern));
};

export function parsePipeDelimitedResponse(text: string): LLMErrorInput[] {
  const trimmedText = text.trim();

  if (!trimmedText) {
    return [];
  }

  const normalizedText = trimmedText.toLowerCase();
  if (NO_ERROR_TOKENS.some((token) => normalizedText.includes(token))) {
    return [];
  }

  const lines = trimmedText.split("\n").filter((line) => line.trim().length > 0);
  const errors: LLMErrorInput[] = [];

  for (const line of lines) {
    const trimmedLine = line.trim();
    if (shouldIgnoreLine(trimmedLine)) {
      continue;
    }

    const parts = trimmedLine.split("|").map((p) => p.trim());

    if (parts.length < EXPECTED_PART_COUNT) {
      console.warn(
        `[getLLMAssessment] Skipping malformed line (expected ${EXPECTED_PART_COUNT} parts, got ${parts.length})`,
        trimmedLine.substring(0, 100),
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

    const cleanErrorText = cleanOrNull(errorText);
    if (!cleanErrorText) {
      console.warn(
        "[getLLMAssessment] Skipping line with empty errorText",
        trimmedLine.substring(0, 100),
      );
      continue;
    }

    errors.push({
      errorText: cleanErrorText,
      wordBefore: cleanOrNull(wordBefore),
      wordAfter: cleanOrNull(wordAfter),
      category: category || "GRAMMAR",
      message: message || "Error detected",
      suggestions: parseSuggestions(suggestionsStr),
      errorType: errorType || "Grammar error",
      explanation: explanation || message || "Error detected",
      severity: normalizeSeverity(severity),
    });
  }

  return errors;
}
