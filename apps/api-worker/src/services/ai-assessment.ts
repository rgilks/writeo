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

export async function getLLMAssessment(
  llmProvider: LLMProvider,
  apiKey: string,
  questionText: string,
  answerText: string,
  modelName: string
): Promise<LanguageToolError[]> {
  const prompt = buildAssessmentPrompt(questionText, answerText);

  try {
    const text = await retryWithBackoff(() =>
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

    const errors = parsePipeDelimitedResponse(text);

    if (errors.length === 0 && llmProvider === "groq") {
      console.log(
        `[getLLMAssessment] Groq returned empty errors array. Response preview:`,
        text.trim().substring(0, 200)
      );
    }

    const processedErrors = errors
      .map((err): LanguageToolError | null => validateAndProcessError(err, answerText))
      .filter((err: LanguageToolError | null): err is LanguageToolError => err !== null);

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
