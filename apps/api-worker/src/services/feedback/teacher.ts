/**
 * Teacher feedback generation
 */

import { callLLMAPI, type LLMProvider } from "../llm";
import {
  MAX_TOKENS_TEACHER_FEEDBACK_INITIAL,
  MAX_TOKENS_TEACHER_FEEDBACK_EXPLANATION,
} from "../../utils/constants";
import { buildTeacherFeedbackPrompt } from "./prompts-teacher";
import { getLowestDimension, getFocusArea, normalizeEssayScores } from "./context";
import type { TeacherFeedback, EssayScores, FeedbackError, RelevanceCheck } from "./types";

const SYSTEM_MESSAGE_INITIAL =
  "You are a professional writing tutor specializing in academic argumentative writing. Always respond with clear, direct feedback. Never mention technical terms like CEFR or band scores. Focus on actionable improvements.";

const SYSTEM_MESSAGE_EXPLANATION =
  "You are an experienced writing instructor providing detailed analysis for another teacher. Use markdown formatting to structure your analysis. Be comprehensive, specific, and cite examples from the student's work. Never mention technical terms like CEFR or band scores.";

const MODE_CONFIG: Record<
  "initial" | "clues" | "explanation",
  { systemMessage: string; maxTokens: number }
> = {
  initial: {
    systemMessage: SYSTEM_MESSAGE_INITIAL,
    maxTokens: MAX_TOKENS_TEACHER_FEEDBACK_INITIAL,
  },
  clues: {
    systemMessage: SYSTEM_MESSAGE_INITIAL,
    maxTokens: MAX_TOKENS_TEACHER_FEEDBACK_INITIAL,
  },
  explanation: {
    systemMessage: SYSTEM_MESSAGE_EXPLANATION,
    maxTokens: MAX_TOKENS_TEACHER_FEEDBACK_EXPLANATION,
  },
};

export async function getTeacherFeedback(
  llmProvider: LLMProvider,
  apiKey: string,
  questionText: string,
  answerText: string,
  modelName: string,
  mode: "initial" | "clues" | "explanation" = "initial",
  essayScores?: EssayScores,
  languageToolErrors?: FeedbackError[],
  llmErrors?: FeedbackError[],
  relevanceCheck?: RelevanceCheck,
): Promise<TeacherFeedback> {
  const normalizedScores = essayScores ? normalizeEssayScores(essayScores) : undefined;
  const { systemMessage, maxTokens } = MODE_CONFIG[mode];

  const prompt = buildTeacherFeedbackPrompt(
    questionText,
    answerText,
    mode,
    normalizedScores,
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
        content: systemMessage,
      },
      {
        role: "user",
        content: prompt,
      },
    ],
    maxTokens,
  );

  const trimmedResponseText = responseText.trim();

  let focusArea: string | undefined;
  if (mode === "initial" && normalizedScores?.dimensions) {
    const lowestDim = getLowestDimension(normalizedScores);
    focusArea = lowestDim ? getFocusArea(lowestDim[0]) : undefined;
  }

  return {
    message: trimmedResponseText,
    ...(mode === "initial" && { focusArea }),
    ...(mode === "clues" && { clues: trimmedResponseText }),
    ...(mode === "explanation" && { explanation: trimmedResponseText }),
  };
}
