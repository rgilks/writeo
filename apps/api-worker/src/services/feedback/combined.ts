/**
 * Combined feedback generation
 */

import { callLLMAPI, type LLMProvider } from "../llm";
import { MAX_TOKENS_DETAILED_FEEDBACK } from "../../utils/constants";
import { buildCombinedFeedbackPrompt } from "./prompts-combined";
import { parseFeedbackResponse } from "./parser";
import { normalizeEssayScores } from "./context";
import type { CombinedFeedback, EssayScores, FeedbackError, RelevanceCheck } from "./types";

const COMBINED_FEEDBACK_SYSTEM_PROMPT =
  "You are an expert English language tutor specializing in academic argumentative writing. Always respond with valid JSON only, no markdown, no explanations. Focus on helping students improve their essay writing skills without mentioning technical terms like CEFR or band scores.";

export interface CombinedFeedbackParams {
  llmProvider: LLMProvider;
  apiKey: string;
  questionText: string;
  answerText: string;
  modelName: string;
  essayScores?: EssayScores;
  languageToolErrors?: FeedbackError[];
  llmErrors?: FeedbackError[];
  relevanceCheck?: RelevanceCheck;
  useMockServices?: boolean;
}

export async function getCombinedFeedback({
  llmProvider,
  apiKey,
  questionText,
  answerText,
  modelName,
  essayScores,
  languageToolErrors,
  llmErrors,
  relevanceCheck,
  useMockServices,
}: CombinedFeedbackParams): Promise<CombinedFeedback> {
  const prompt = buildCombinedFeedbackPrompt(
    questionText,
    answerText,
    normalizeEssayScores(essayScores),
    languageToolErrors,
    llmErrors,
    relevanceCheck,
  );

  const responseText = await callLLMAPI(
    llmProvider,
    apiKey,
    modelName,
    [
      {
        role: "system",
        content: COMBINED_FEEDBACK_SYSTEM_PROMPT,
      },
      {
        role: "user",
        content: prompt,
      },
    ],
    MAX_TOKENS_DETAILED_FEEDBACK,
    useMockServices,
  );

  const trimmedResponse = responseText.trim();
  if (!trimmedResponse) {
    throw new Error("LLM returned an empty response for combined feedback");
  }

  try {
    return parseFeedbackResponse(trimmedResponse);
  } catch (error) {
    throw new Error(
      `Combined feedback parsing failed: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}
