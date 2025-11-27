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

const DEFAULT_CATEGORY = "GRAMMAR";
const DEFAULT_MESSAGE = "Error detected";
const DEFAULT_RULE_PREFIX = "LLM";
const DEFAULT_CONFIDENCE_SCORE = 0.75;
const HIGH_CONFIDENCE_THRESHOLD = 0.85;
const MEDIUM_CONFIDENCE_THRESHOLD = 0.5;

const normalizeCategory = (category?: string): string => {
  return (category || DEFAULT_CATEGORY).toUpperCase();
};

const formatRuleId = (errorType?: string): string => {
  const normalized = errorType?.trim().replace(/\s+/g, "_").toUpperCase();
  return `${DEFAULT_RULE_PREFIX}_${normalized || "ERROR"}`;
};

const getMessage = (message?: string): string => {
  return message?.trim() || DEFAULT_MESSAGE;
};

const getExplanation = (explanation?: string, fallbackMessage?: string): string => {
  return explanation?.trim() || fallbackMessage?.trim() || DEFAULT_MESSAGE;
};

const filterSuggestions = (actualText: string, rawSuggestions?: string[]): string[] => {
  if (!Array.isArray(rawSuggestions)) {
    return [];
  }

  const trimmedActual = actualText.trim();
  return rawSuggestions
    .map((suggestion) => suggestion?.trim())
    .filter(
      (suggestion): suggestion is string => Boolean(suggestion) && suggestion !== trimmedActual,
    )
    .slice(0, 5);
};

const deriveConfidence = (score: number) => ({
  confidenceScore: score,
  highConfidence: score >= HIGH_CONFIDENCE_THRESHOLD,
  mediumConfidence: score >= MEDIUM_CONFIDENCE_THRESHOLD,
});

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

  const suggestions = filterSuggestions(actualErrorText, err.suggestions);
  const message = getMessage(err.message);
  const explanation = getExplanation(err.explanation, err.message);
  const category = normalizeCategory(err.category);
  const ruleId = formatRuleId(err.errorType || err.category);
  const confidence = deriveConfidence(DEFAULT_CONFIDENCE_SCORE);

  return {
    start: validated.start,
    end: validated.end,
    length: validated.end - validated.start,
    category,
    rule_id: ruleId,
    message,
    suggestions,
    source: "LLM" as const,
    severity: (err.severity || "error") as "warning" | "error",
    confidenceScore: confidence.confidenceScore,
    highConfidence: confidence.highConfidence,
    mediumConfidence: confidence.mediumConfidence,
    errorType: err.errorType || "Grammar error",
    explanation,
    example: suggestions[0] ? `Try: "${suggestions[0]}"` : undefined,
  };
}
