/**
 * Combined feedback generation
 */

import { callLLMAPI, type LLMProvider } from "../llm";
import { MAX_TOKENS_DETAILED_FEEDBACK } from "../../utils/constants";
import { buildCombinedFeedbackPrompt } from "./prompts-combined";
import { parseFeedbackResponse } from "./parser";
import type { CombinedFeedback } from "./types";

export async function getCombinedFeedback(
  llmProvider: LLMProvider,
  apiKey: string,
  questionText: string,
  answerText: string,
  modelName: string,
  essayScores?: {
    overall?: number;
    dimensions?: { TA?: number; CC?: number; Vocab?: number; Grammar?: number; Overall?: number };
    label?: string;
  },
  languageToolErrors?: Array<{
    message: string;
    category: string;
    suggestions?: string[];
    start: number;
    end: number;
  }>,
  llmErrors?: Array<{
    message: string;
    category: string;
    suggestions?: string[];
    start: number;
    end: number;
    errorType?: string;
  }>,
  relevanceCheck?: { addressesQuestion: boolean; score: number; threshold: number }
): Promise<CombinedFeedback> {
  const prompt = buildCombinedFeedbackPrompt(
    questionText,
    answerText,
    essayScores,
    languageToolErrors,
    llmErrors,
    relevanceCheck
  );

  const responseText = await callLLMAPI(
    llmProvider,
    apiKey,
    modelName,
    [
      {
        role: "system",
        content:
          "You are an expert English language tutor specializing in academic argumentative writing. Always respond with valid JSON only, no markdown, no explanations. Focus on helping students improve their essay writing skills without mentioning technical terms like , CEFR, or band scores.",
      },
      {
        role: "user",
        content: prompt,
      },
    ],
    MAX_TOKENS_DETAILED_FEEDBACK
  );

  return parseFeedbackResponse(responseText);
}
