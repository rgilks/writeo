/**
 * AI Assessment service - main exports
 */

import type { LanguageToolError } from "@writeo/shared";
import { callLLMAPI, type LLMProvider } from "./llm";
import { MAX_TOKENS_GRAMMAR_CHECK } from "../utils/constants";
import { retryWithBackoff } from "@writeo/shared";
import { buildAssessmentPrompt } from "./ai-assessment/prompts";
import { parsePipeDelimitedResponse } from "./ai-assessment/parser";
import { validateAndProcessError } from "./ai-assessment/validation";

const SYSTEM_PROMPT =
  "You are an expert English grammar checker. Check the ENTIRE text systematically from beginning to end. Return ALL errors as pipe-delimited lines. Format: errorText|wordBefore|wordAfter|category|message|suggestions|errorType|explanation|severity. Include the word before and after the error for context (or empty if at sentence boundaries). No headers/explanations. Make sure to check the middle and end sections of the text as thoroughly as the beginning.";

function calculateErrorDistribution(
  errors: LanguageToolError[],
  textLength: number,
): { firstHalf: number; secondHalf: number } {
  const midpoint = textLength / 2;
  const firstHalf = errors.filter((e) => e.start < midpoint).length;
  return { firstHalf, secondHalf: errors.length - firstHalf };
}

function logResults(
  llmProvider: LLMProvider,
  processedErrors: LanguageToolError[],
  rawErrors: number,
  answerTextLength: number,
): void {
  const baseMessage = `[getLLMAssessment] ${llmProvider} processed ${processedErrors.length} valid errors (from ${rawErrors} raw errors)`;

  if (processedErrors.length > 0) {
    const { firstHalf, secondHalf } = calculateErrorDistribution(processedErrors, answerTextLength);
    console.log(
      `${baseMessage} - Distribution: ${firstHalf} in first half, ${secondHalf} in second half (text length: ${answerTextLength} chars)`,
    );
  } else {
    console.log(baseMessage);
  }
}

export async function getLLMAssessment(
  llmProvider: LLMProvider,
  apiKey: string,
  questionText: string,
  answerText: string,
  modelName: string,
  useMockServices?: boolean,
): Promise<LanguageToolError[]> {
  const prompt = buildAssessmentPrompt(questionText, answerText);

  try {
    // Check if we should use mocks - pass undefined to let callLLMAPI check internally
    // This maintains backward compatibility while allowing explicit control
    const text = await retryWithBackoff(() =>
      callLLMAPI(
        llmProvider,
        apiKey,
        modelName,
        [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: prompt },
        ],
        MAX_TOKENS_GRAMMAR_CHECK,
        useMockServices, // Pass through the mock flag
      ),
    );

    if (!text) {
      return [];
    }

    const errors = parsePipeDelimitedResponse(text);

    if (errors.length === 0 && llmProvider === "groq") {
      console.log(
        `[getLLMAssessment] Groq returned empty errors array. Response preview:`,
        text.trim().substring(0, 200),
      );
    }

    const processedErrors = errors
      .map((err) => validateAndProcessError(err, answerText))
      .filter((err): err is LanguageToolError => err !== null);

    logResults(llmProvider, processedErrors, errors.length, answerText.length);

    return processedErrors;
  } catch (error) {
    console.error(`[getLLMAssessment] Error in getLLMAssessment for ${llmProvider}:`, error);
    return [];
  }
}
