import type { LanguageToolError } from "@writeo/shared";
import { callLLMAPI, type LLMProvider } from "./llm";
import {
  validateAndCorrectErrorPosition,
  truncateEssayText,
  truncateQuestionText,
} from "../utils/text-processing";
import { MAX_TOKENS_GRAMMAR_CHECK } from "../utils/constants";

// Note: With pipe-delimited format, we don't need JSON repair anymore
// Each line is independent, so truncation only affects incomplete lines at the end

/**
 * Retry helper with exponential backoff for LLM calls
 */
async function retryLLMCall<T>(
  fn: () => Promise<T>,
  maxAttempts: number = 3,
  baseDelayMs: number = 500
): Promise<T> {
  let lastError: Error | undefined;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      // Don't retry on client errors (4xx) - these are not transient
      if (error instanceof Error && error.message.includes("4")) {
        throw error;
      }

      // Don't retry on last attempt
      if (attempt === maxAttempts - 1) {
        break;
      }

      // Exponential backoff: 500ms, 1000ms, 2000ms
      const delay = baseDelayMs * Math.pow(2, attempt);
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }

  throw lastError || new Error("LLM call failed after retries");
}

export async function getLLMAssessment(
  llmProvider: LLMProvider,
  apiKey: string,
  questionText: string,
  answerText: string,
  modelName: string
): Promise<LanguageToolError[]> {
  // Truncate long texts to reduce token usage and costs
  // Keep enough context for meaningful grammar checking (~2000 words / ~12000 chars)
  // NOTE: Only the first 12,000 characters are processed, so errors beyond this point
  // will not be detected. Errors near the truncation boundary are handled carefully
  // to ensure they aren't incorrectly filtered out.
  const MAX_TEXT_LENGTH_FOR_GRAMMAR_CHECK = 12000;
  const truncatedAnswerText =
    answerText.length > MAX_TEXT_LENGTH_FOR_GRAMMAR_CHECK
      ? answerText.slice(0, MAX_TEXT_LENGTH_FOR_GRAMMAR_CHECK) + "\n\n[... text continues ...]"
      : answerText;
  const truncatedQuestionText = truncateQuestionText(questionText);

  // More concise prompt to reduce token usage
  const prompt = `Find ALL grammar, spelling, style, and punctuation errors in the student's answer.

IMPORTANT: Check the ENTIRE text systematically from beginning to end. Do not focus only on the beginning - make sure to check the middle and end sections equally thoroughly.

Focus on:
- Tense errors (past-time indicators → past tense verbs)
- Grammar (subject-verb agreement, articles, prepositions, word order)
- Spelling, style, punctuation, confused words

Question: ${truncatedQuestionText}
Answer: ${truncatedAnswerText}

Return format: ONE ERROR PER LINE, pipe-delimited.
Format: start|end|errorText|category|message|suggestions|errorType|explanation|severity

Where:
- start/end: character positions (0-based from start of answer)
- errorText: exact text at that position
- category: GRAMMAR, SPELLING, STYLE, PUNCTUATION, TYPOS, or CONFUSED_WORDS
- message: error description
- suggestions: comma-separated corrections (e.g., "went,goes")
- errorType: error type (e.g., "Verb tense", "Subject-verb agreement")
- explanation: brief explanation
- severity: "error" or "warning"

Example:
15|20|go to|GRAMMAR|Verb tense error|went|Verb tense|Use past tense|error
34|37|was|GRAMMAR|Subject-verb agreement|were|Subject-verb agreement|We requires were|error

Return ONLY error lines (one per line), no headers/explanations.
If no errors: NO_ERRORS`;

  try {
    // Use retry logic for reliability
    const text = await retryLLMCall(() =>
      callLLMAPI(
        llmProvider,
        apiKey,
        modelName,
        [
          {
            role: "system",
            content:
              "You are an expert English grammar checker. Check the ENTIRE text systematically from beginning to end. Return ALL errors as pipe-delimited lines. Format: start|end|errorText|category|message|suggestions|errorType|explanation|severity. No headers/explanations. Make sure to check the middle and end sections of the text as thoroughly as the beginning.",
          },
          {
            role: "user",
            content: prompt,
          },
        ],
        MAX_TOKENS_GRAMMAR_CHECK
      )
    );

    if (!text) {
      return [];
    }

    const trimmedText = text.trim();

    // Handle "NO_ERRORS" response
    if (trimmedText === "NO_ERRORS" || trimmedText.toLowerCase().includes("no errors")) {
      console.log(`[getLLMAssessment] ${llmProvider} reported no errors`);
      return [];
    }

    // Parse pipe-delimited format: start|end|errorText|category|message|suggestions|errorType|explanation|severity
    const lines = trimmedText.split("\n").filter((line) => line.trim().length > 0);
    const errors: any[] = [];

    for (const line of lines) {
      const trimmedLine = line.trim();
      // Skip empty lines or lines that look like headers/explanations
      if (
        !trimmedLine ||
        trimmedLine.startsWith("#") ||
        trimmedLine.toLowerCase().includes("format:") ||
        trimmedLine.toLowerCase().includes("example:")
      ) {
        continue;
      }

      const parts = trimmedLine.split("|").map((p) => p.trim());

      // Need at least 9 parts (start|end|errorText|category|message|suggestions|errorType|explanation|severity)
      if (parts.length < 9) {
        console.warn(
          `[getLLMAssessment] Skipping malformed line (expected 9 parts, got ${parts.length}):`,
          trimmedLine.substring(0, 100)
        );
        continue;
      }

      const [
        startStr,
        endStr,
        errorText,
        category,
        message,
        suggestionsStr,
        errorType,
        explanation,
        severity,
      ] = parts;

      const start = parseInt(startStr, 10);
      const end = parseInt(endStr, 10);

      if (isNaN(start) || isNaN(end)) {
        console.warn(
          `[getLLMAssessment] Skipping line with invalid positions: start=${startStr}, end=${endStr}`
        );
        continue;
      }

      // Parse suggestions (comma-separated)
      const suggestions = suggestionsStr
        .split(",")
        .map((s) => s.trim())
        .filter((s) => s.length > 0);

      errors.push({
        start,
        end,
        errorText,
        category: category || "GRAMMAR",
        message: message || "Error detected",
        suggestions,
        errorType: errorType || "Grammar error",
        explanation: explanation || message || "Error detected",
        severity: severity || "error",
      });
    }

    console.log(
      `[getLLMAssessment] ${llmProvider} returned ${errors.length} errors from pipe-delimited parsing`
    );

    if (errors.length === 0 && llmProvider === "groq") {
      console.log(
        `[getLLMAssessment] Groq returned empty errors array. Response preview:`,
        trimmedText.substring(0, 200)
      );
    }

    const processedErrors = errors
      .map((err: any): LanguageToolError | null => {
        if (typeof err.start !== "number" || typeof err.end !== "number") {
          return null;
        }

        // Get the error text from the response or extract it from the answer text
        const reportedErrorText = err.errorText || "";
        // Positions are relative to truncatedAnswerText, but we validate against original answerText
        // (positions should be valid for first 12k chars of original text)
        // Cap positions at the truncation boundary to prevent issues with errors near the end
        const maxValidPosition = Math.min(answerText.length, MAX_TEXT_LENGTH_FOR_GRAMMAR_CHECK);

        // Ensure error positions don't exceed the truncation boundary
        // Errors beyond this point weren't processed by the LLM
        if (err.start >= maxValidPosition) {
          // This error is beyond the truncation boundary - skip it
          console.log(
            `[getLLMAssessment] Skipping error beyond truncation boundary: start=${err.start}, max=${maxValidPosition}`
          );
          return null;
        }

        const extractedErrorText = answerText.substring(
          Math.max(0, err.start),
          Math.min(maxValidPosition, err.end)
        );

        // Validate and correct the position
        // For Groq, be more lenient - try to find the errorText even if positions are off
        // Cap end position at truncation boundary to prevent validation issues
        let validationInput = {
          start: err.start || 0,
          end: Math.min(err.end || 0, maxValidPosition),
          errorText: reportedErrorText || extractedErrorText,
          message: err.message,
          errorType: err.errorType || err.category,
        };

        // If Groq and we have errorText but positions seem wrong, try to find it first
        if (llmProvider === "groq" && reportedErrorText && reportedErrorText.trim().length > 0) {
          // Try to find the errorText in the answer text, but only within the truncation boundary
          const searchStart = Math.max(0, err.start - 100);
          const searchEnd = Math.min(maxValidPosition, err.end + 100);
          const searchArea = answerText.substring(searchStart, searchEnd);
          const foundIndex = searchArea
            .toLowerCase()
            .indexOf(reportedErrorText.toLowerCase().trim());

          if (foundIndex !== -1) {
            // Found it! Use the found position
            const correctedStart = searchStart + foundIndex;
            const correctedEnd = correctedStart + reportedErrorText.trim().length;
            validationInput.start = correctedStart;
            validationInput.end = correctedEnd;
            console.log(
              `[getLLMAssessment] Groq: Found errorText "${reportedErrorText}" at corrected position ${correctedStart}-${correctedEnd} (original: ${err.start}-${err.end})`
            );
          }
        }

        // Validate against original answerText (positions should be valid for first part)
        const validated = validateAndCorrectErrorPosition(validationInput, answerText);

        if (!validated.valid) {
          // For Groq, if we have errorText, try one more time with a more aggressive search
          if (llmProvider === "groq" && reportedErrorText && reportedErrorText.trim().length > 0) {
            // Search within the truncation boundary for the errorText
            const searchText = answerText.substring(0, maxValidPosition).toLowerCase();
            const errorTextLower = reportedErrorText.toLowerCase().trim();
            const foundIndex = searchText.indexOf(errorTextLower);

            if (foundIndex !== -1) {
              // Found it! Create a new validation with the found position
              const retryValidated = validateAndCorrectErrorPosition(
                {
                  start: foundIndex,
                  end: foundIndex + errorTextLower.length,
                  errorText: reportedErrorText,
                  message: err.message,
                  errorType: err.errorType || err.category,
                },
                answerText
              );

              if (retryValidated.valid) {
                console.log(
                  `[getLLMAssessment] Groq: Retry validation succeeded for "${reportedErrorText}" at ${retryValidated.start}-${retryValidated.end}`
                );
                // Use the retry validated position
                const actualErrorText = answerText.substring(
                  retryValidated.start,
                  retryValidated.end
                );
                let suggestions = Array.isArray(err.suggestions) ? err.suggestions : [];
                suggestions = suggestions.filter(
                  (s: string) => s && s.trim() !== actualErrorText.trim()
                );

                return {
                  start: retryValidated.start,
                  end: retryValidated.end,
                  length: retryValidated.end - retryValidated.start,
                  category: (err.category || "GRAMMAR").toUpperCase(),
                  rule_id: `LLM_${err.errorType?.replace(/\s+/g, "_").toUpperCase() || "ERROR"}`,
                  message: err.message || "Error detected",
                  suggestions: suggestions.slice(0, 5),
                  source: "LLM" as const,
                  severity: (err.severity || "error") as "warning" | "error",
                  confidenceScore: 0.75,
                  highConfidence: false,
                  mediumConfidence: true,
                  errorType: err.errorType || "Grammar error",
                  explanation: err.explanation || err.message || "Error detected",
                  example: suggestions[0] ? `Try: "${suggestions[0]}"` : undefined,
                };
              }
            }
          }

          // Skip invalid positions
          console.warn(`[getLLMAssessment] Rejected error for ${llmProvider}:`, {
            originalStart: err.start,
            originalEnd: err.end,
            errorText: reportedErrorText,
            extractedText: extractedErrorText,
            category: err.category,
            validationInput: validationInput,
          });
          return null;
        }

        if (
          llmProvider === "groq" &&
          (validated.start !== err.start || validated.end !== err.end)
        ) {
          console.log(`[getLLMAssessment] Groq position corrected:`, {
            original: `${err.start}-${err.end}`,
            corrected: `${validated.start}-${validated.end}`,
            errorText: reportedErrorText,
          });
        }

        // Get the actual text at the validated position
        const actualErrorText = answerText.substring(validated.start, validated.end);

        // Validate that suggestions make sense for the actual text
        // Filter out suggestions that don't match the error type or are clearly wrong
        let suggestions = Array.isArray(err.suggestions) ? err.suggestions : [];

        // If we have suggestions, ensure they're reasonable
        // Remove suggestions that are identical to the error text (no change)
        suggestions = suggestions.filter((s: string) => s && s.trim() !== actualErrorText.trim());

        // Ensure error message and explanation match the actual error
        let message = err.message || "Error detected";
        let explanation = err.explanation || err.message || "Error detected";

        // If the explanation doesn't mention the actual error text, try to improve it
        if (!explanation.toLowerCase().includes(actualErrorText.toLowerCase().substring(0, 5))) {
          // The explanation might be generic, but we'll keep it if it's reasonable
          // The frontend will show the actual highlighted text anyway
        }

        return {
          start: validated.start,
          end: validated.end,
          length: validated.end - validated.start,
          category: (err.category || "GRAMMAR").toUpperCase(),
          rule_id: `LLM_${err.errorType?.replace(/\s+/g, "_").toUpperCase() || "ERROR"}`,
          message: message,
          suggestions: suggestions.slice(0, 5), // Limit to 5 suggestions
          source: "LLM" as const,
          severity: (err.severity || "error") as "warning" | "error",
          confidenceScore: 0.75,
          highConfidence: false,
          mediumConfidence: true,
          errorType: err.errorType || "Grammar error",
          explanation: explanation,
          example: suggestions[0] ? `Try: "${suggestions[0]}"` : undefined,
        };
      })
      .filter((err: LanguageToolError | null): err is LanguageToolError => err !== null);

    // Log error distribution to help identify if errors are biased toward the beginning
    if (processedErrors.length > 0) {
      const textLength = answerText.length;
      const firstHalf = processedErrors.filter((e) => e.start < textLength / 2).length;
      const secondHalf = processedErrors.length - firstHalf;
      console.log(
        `[getLLMAssessment] ${llmProvider} processed ${processedErrors.length} valid errors (from ${errors.length} raw errors) - Distribution: ${firstHalf} in first half, ${secondHalf} in second half (text length: ${textLength} chars)`
      );
    } else {
      console.log(
        `[getLLMAssessment] ${llmProvider} processed ${processedErrors.length} valid errors (from ${errors.length} raw errors)`
      );
    }

    return processedErrors;
  } catch (error) {
    console.error(`[getLLMAssessment] Error in getLLMAssessment for ${llmProvider}:`, error);
    return [];
  }
}
