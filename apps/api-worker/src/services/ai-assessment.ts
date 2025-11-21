import type { LanguageToolError } from "@writeo/shared";
import { callLLMAPI, type LLMProvider } from "./llm";
import {
  validateAndCorrectErrorPosition,
  truncateEssayText,
  truncateQuestionText,
  findTextWithContext,
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
Format: errorText|wordBefore|wordAfter|category|message|suggestions|errorType|explanation|severity

Where:
- errorText: the exact text that contains the error (the word or phrase that's wrong)
- wordBefore: the word immediately before the error (or empty if at start of sentence)
- wordAfter: the word immediately after the error (or empty if at end of sentence)
- category: GRAMMAR, SPELLING, STYLE, PUNCTUATION, TYPOS, or CONFUSED_WORDS
- message: error description
- suggestions: comma-separated corrections (e.g., "went,goes")
- errorType: error type (e.g., "Verb tense", "Subject-verb agreement")
- explanation: brief explanation
- severity: "error" or "warning"

Example:
go to|I|the|GRAMMAR|Verb tense error|went|Verb tense|Use past tense|error
was|they|happy|GRAMMAR|Subject-verb agreement|were|Subject-verb agreement|They requires were|error

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
              "You are an expert English grammar checker. Check the ENTIRE text systematically from beginning to end. Return ALL errors as pipe-delimited lines. Format: errorText|wordBefore|wordAfter|category|message|suggestions|errorType|explanation|severity. Include the word before and after the error for context (or empty if at sentence boundaries). No headers/explanations. Make sure to check the middle and end sections of the text as thoroughly as the beginning.",
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

    // Parse pipe-delimited format: errorText|wordBefore|wordAfter|category|message|suggestions|errorType|explanation|severity
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

      // Need at least 9 parts (errorText|wordBefore|wordAfter|category|message|suggestions|errorType|explanation|severity)
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

      // Validate errorText is not empty
      if (!errorText || errorText.trim().length === 0) {
        console.warn(
          `[getLLMAssessment] Skipping line with empty errorText:`,
          trimmedLine.substring(0, 100)
        );
        continue;
      }

      // Parse suggestions (comma-separated)
      const suggestions = suggestionsStr
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
        // Validate required fields
        if (!err.errorText || err.errorText.trim().length === 0) {
          console.warn(`[getLLMAssessment] Skipping error with empty errorText`);
          return null;
        }

        const reportedErrorText = err.errorText.trim();
        const wordBefore = err.wordBefore || null;
        const wordAfter = err.wordAfter || null;

        // Cap search at truncation boundary - errors beyond this weren't processed by LLM
        const maxValidPosition = Math.min(answerText.length, MAX_TEXT_LENGTH_FOR_GRAMMAR_CHECK);
        const searchText = answerText.substring(0, maxValidPosition);

        // Use context-aware text matching to find the position
        let foundPosition = findTextWithContext(
          reportedErrorText,
          wordBefore,
          wordAfter,
          searchText
        );

        if (!foundPosition) {
          // Try fallback: simple search without context
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
              }
            );
            return null;
          }

          // Use simple found position
          foundPosition = {
            start: foundIndex,
            end: foundIndex + reportedErrorText.length,
          };
        }

        // Validate and correct the position to align with word boundaries
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
