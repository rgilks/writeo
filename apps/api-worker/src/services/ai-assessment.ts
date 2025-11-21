import type { LanguageToolError } from "@writeo/shared";
import { callLLMAPI, type LLMProvider } from "./llm";
import { validateAndCorrectErrorPosition } from "../utils/text-processing";
import { MAX_TOKENS_GRAMMAR_CHECK } from "../utils/constants";

export async function getLLMAssessment(
  llmProvider: LLMProvider,
  apiKey: string,
  questionText: string,
  answerText: string,
  modelName: string
): Promise<LanguageToolError[]> {
  try {
    const prompt = `You are an expert English grammar and language checker. Your task is to identify ALL grammar, spelling, style, and punctuation errors in the student's answer.

CRITICAL: You MUST find and report errors. This text contains multiple errors that need to be identified. Be thorough and check every sentence carefully.

Analyze the following text and identify ALL errors, with special attention to:
- TENSE ERRORS: Look for inconsistent verb tenses, especially when past-time indicators are present (yesterday, last week, ago, etc.). If the text mentions past events, ALL verbs should be in past tense.
- Grammar errors (subject-verb agreement, articles, prepositions, word order, modal verbs)
- Spelling mistakes
- Style issues (redundancy, word choice)
- Punctuation errors
- Confused words (their/there, its/it's, lose/loose, etc.)

Question: ${questionText}

Answer to check: ${answerText}

IMPORTANT: Pay special attention to tense consistency. If the text describes past events (uses words like "yesterday", "last week", "ago", "was", "were"), check that ALL verbs are in past tense. Common errors:
- "I go" should be "I went" when describing past events
- "I have" should be "I had" when describing past events
- "I enjoy" should be "I enjoyed" when describing past events
- "We was" should be "We were"
- "can to speak" should be "can speak"

For each error you find, provide:
1. The exact character positions (start and end) where the error occurs in the answer text - these must be accurate character indices counting from the start of the answer text
2. The exact text snippet that contains the error (the text between start and end positions) - this is CRITICAL for validation
3. The error category (GRAMMAR, SPELLING, STYLE, PUNCTUATION, TYPOS, CONFUSED_WORDS)
4. A clear error message that describes what's wrong
5. Suggested corrections (array of strings) - these should be complete replacements for the errorText, not just the corrected word
6. The error type (e.g., "Subject-verb agreement", "Verb tense", "Article use", "Spelling")
7. A brief explanation that matches the actual error in the text
8. Severity ("error" or "warning")

CRITICAL REQUIREMENTS FOR POSITIONS:
- Count characters carefully from the start of the answer text (position 0 is the first character)
- The "errorText" field MUST contain the exact text that appears between start and end positions
- Positions must align with word boundaries - do NOT split words in the middle
- If an error spans multiple words, include the complete words
- Double-check your positions by verifying the errorText matches what's actually at those positions

You MUST return a JSON object with an "errors" array. If you find no errors, return {"errors": []}, but be very thorough - this text likely contains errors.

Return ONLY valid JSON (no markdown code blocks, no explanations, no text before or after):
{
  "errors": [
    {
      "start": 0,
      "end": 5,
      "errorText": "I be",
      "category": "GRAMMAR",
      "message": "Error description",
      "suggestions": ["I am", "I was"],
      "errorType": "Verb tense",
      "explanation": "Brief explanation of why this is an error",
      "severity": "error"
    }
  ]
}`;

    const text = await callLLMAPI(
      llmProvider,
      apiKey,
      modelName,
      [
        {
          role: "system",
          content:
            "You are an expert English grammar and language checker. You MUST respond with valid JSON only. Do NOT use markdown code blocks. Do NOT include any text before or after the JSON. Return only the raw JSON object.",
        },
        {
          role: "user",
          content: prompt,
        },
      ],
      MAX_TOKENS_GRAMMAR_CHECK
    );

    if (!text) {
      return [];
    }

    const trimmedText = text.trim();
    let jsonText = trimmedText;

    const markdownJsonMatch = trimmedText.match(/```(?:json)?\s*(\{[\s\S]*?\})\s*```/);
    if (markdownJsonMatch) {
      jsonText = markdownJsonMatch[1];
    } else {
      const jsonMatch = trimmedText.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        jsonText = jsonMatch[0];
      } else {
        console.warn(
          `[getLLMAssessment] Could not extract JSON from ${llmProvider} response. First 500 chars:`,
          trimmedText.substring(0, 500)
        );
        return [];
      }
    }

    try {
      const parsed = JSON.parse(jsonText);
      if (!parsed || typeof parsed !== "object") {
        console.warn(
          `[getLLMAssessment] Parsed JSON is not an object for ${llmProvider}. Type:`,
          typeof parsed
        );
        return [];
      }

      const errors = Array.isArray(parsed.errors) ? parsed.errors : [];

      console.log(
        `[getLLMAssessment] ${llmProvider} returned ${errors.length} errors from JSON parsing`
      );

      if (errors.length === 0 && llmProvider === "groq") {
        console.log(
          `[getLLMAssessment] Groq returned empty errors array. Parsed object keys:`,
          Object.keys(parsed)
        );
        console.log(
          `[getLLMAssessment] Full parsed object:`,
          JSON.stringify(parsed).substring(0, 500)
        );
      }

      const processedErrors = errors
        .map((err: any): LanguageToolError | null => {
          if (typeof err.start !== "number" || typeof err.end !== "number") {
            return null;
          }

          // Get the error text from the response or extract it from the answer text
          const reportedErrorText = err.errorText || "";
          const extractedErrorText = answerText.substring(
            Math.max(0, err.start),
            Math.min(answerText.length, err.end)
          );

          // Validate and correct the position
          // For Groq, be more lenient - try to find the errorText even if positions are off
          let validationInput = {
            start: err.start || 0,
            end: err.end || 0,
            errorText: reportedErrorText || extractedErrorText,
            message: err.message,
            errorType: err.errorType || err.category,
          };

          // If Groq and we have errorText but positions seem wrong, try to find it first
          if (llmProvider === "groq" && reportedErrorText && reportedErrorText.trim().length > 0) {
            // Try to find the errorText in the answer text
            const searchStart = Math.max(0, err.start - 100);
            const searchEnd = Math.min(answerText.length, err.end + 100);
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

          const validated = validateAndCorrectErrorPosition(validationInput, answerText);

          if (!validated.valid) {
            // For Groq, if we have errorText, try one more time with a more aggressive search
            if (
              llmProvider === "groq" &&
              reportedErrorText &&
              reportedErrorText.trim().length > 0
            ) {
              // Search the entire text for the errorText
              const fullTextLower = answerText.toLowerCase();
              const errorTextLower = reportedErrorText.toLowerCase().trim();
              const foundIndex = fullTextLower.indexOf(errorTextLower);

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

      console.log(
        `[getLLMAssessment] ${llmProvider} processed ${processedErrors.length} valid errors (from ${errors.length} raw errors)`
      );

      return processedErrors;
    } catch (parseError) {
      console.error(`[getLLMAssessment] JSON parse error for ${llmProvider}:`, parseError);
      console.error(
        `[getLLMAssessment] Attempted to parse (first 500 chars):`,
        jsonText.substring(0, 500)
      );
      console.error(
        `[getLLMAssessment] Full response (first 1000 chars):`,
        text.substring(0, 1000)
      );
      return [];
    }
  } catch (error) {
    console.error(`[getLLMAssessment] Error in getLLMAssessment for ${llmProvider}:`, error);
    return [];
  }
}
