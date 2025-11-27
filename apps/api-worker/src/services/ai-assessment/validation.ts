/**
 * Error validation and position finding utilities
 */

import type { LanguageToolError } from "@writeo/shared";
import { validateAndCorrectErrorPosition, findTextWithContext } from "../../utils/text-processing";
import { getMaxTextLength } from "./prompts";

export interface LLMErrorInput {
  errorText: string;
  wordBefore?: string | null;
  wordAfter?: string | null;
  category?: string;
  message?: string;
  errorType?: string;
  suggestions?: string[];
  severity?: "warning" | "error";
  explanation?: string;
}

export function validateAndProcessError(
  err: LLMErrorInput,
  answerText: string,
): LanguageToolError | null {
  if (!err.errorText || err.errorText.trim().length === 0) {
    console.warn(`[getLLMAssessment] Skipping error with empty errorText`);
    return null;
  }

  const reportedErrorText = err.errorText.trim();
  const wordBefore = err.wordBefore || null;
  const wordAfter = err.wordAfter || null;

  const maxValidPosition = Math.min(answerText.length, getMaxTextLength());
  const searchText = answerText.substring(0, maxValidPosition);

  let foundPosition = findTextWithContext(reportedErrorText, wordBefore, wordAfter, searchText);

  if (!foundPosition) {
    const errorTextLower = reportedErrorText.toLowerCase();
    const searchTextLower = searchText.toLowerCase();
    const foundIndex = searchTextLower.indexOf(errorTextLower);

    if (foundIndex === -1) {
      console.warn(
        `[getLLMAssessment] Could not find error text "${reportedErrorText}" in answer text`,
        {
          wordBefore,
          wordAfter,
          category: err.category,
        },
      );
      return null;
    }

    foundPosition = {
      start: foundIndex,
      end: foundIndex + reportedErrorText.length,
    };
  }

  const validationInput = {
    start: foundPosition.start,
    end: foundPosition.end,
    errorText: reportedErrorText,
    message: err.message,
    errorType: err.errorType || err.category,
  };

  const validated = validateAndCorrectErrorPosition(validationInput, answerText);

  if (!validated.valid) {
    console.warn(`[getLLMAssessment] Rejected error after validation:`, {
      errorText: reportedErrorText,
      wordBefore,
      wordAfter,
      foundPosition,
      category: err.category,
    });
    return null;
  }

  const actualErrorText = answerText.substring(validated.start, validated.end);

  let suggestions = Array.isArray(err.suggestions) ? err.suggestions : [];
  suggestions = suggestions.filter((s) => s && s.trim() !== actualErrorText.trim());

  let message = err.message || "Error detected";
  let explanation = err.explanation || err.message || "Error detected";

  return {
    start: validated.start,
    end: validated.end,
    length: validated.end - validated.start,
    category: (err.category || "GRAMMAR").toUpperCase(),
    rule_id: `LLM_${err.errorType?.replace(/\s+/g, "_").toUpperCase() || "ERROR"}`,
    message: message,
    suggestions: suggestions.slice(0, 5),
    source: "LLM" as const,
    severity: (err.severity || "error") as "warning" | "error",
    confidenceScore: 0.75,
    highConfidence: false,
    mediumConfidence: true,
    errorType: err.errorType || "Grammar error",
    explanation: explanation,
    example: suggestions[0] ? `Try: "${suggestions[0]}"` : undefined,
  };
}
