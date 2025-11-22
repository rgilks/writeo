/**
 * Teacher feedback generation
 */

import { callLLMAPI, type LLMProvider } from "../llm";
import {
  MAX_TOKENS_TEACHER_FEEDBACK_INITIAL,
  MAX_TOKENS_TEACHER_FEEDBACK_EXPLANATION,
} from "../../utils/constants";
import { buildTeacherFeedbackPrompt } from "./prompts-teacher";
import { getLowestDimension, getFocusArea } from "./context";
import type { TeacherFeedback } from "./types";

export async function getTeacherFeedback(
  llmProvider: LLMProvider,
  apiKey: string,
  questionText: string,
  answerText: string,
  modelName: string,
  mode: "initial" | "clues" | "explanation" = "initial",
  essayScores?: {
    overall?: number;
    dimensions?: { TA?: number; CC?: number; Vocab?: number; Grammar?: number; Overall?: number };
  },
  languageToolErrors?: Array<{
    message: string;
    category: string;
    suggestions?: string[];
    start: number;
    end: number;
    errorType?: string;
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
): Promise<TeacherFeedback> {
  const prompt = buildTeacherFeedbackPrompt(
    questionText,
    answerText,
    mode,
    essayScores,
    languageToolErrors,
    llmErrors,
    relevanceCheck
  );

  const systemMessage =
    mode === "explanation"
      ? "You are an experienced writing instructor providing detailed analysis for another teacher. Use markdown formatting to structure your analysis. Be comprehensive, specific, and cite examples from the student's work. Never mention technical terms like CEFR or band scores."
      : "You are a professional writing tutor specializing in academic argumentative writing. Always respond with clear, direct feedback. Never mention technical terms like CEFR or band scores. Focus on actionable improvements.";

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
    mode === "explanation"
      ? MAX_TOKENS_TEACHER_FEEDBACK_EXPLANATION
      : MAX_TOKENS_TEACHER_FEEDBACK_INITIAL
  );

  const trimmedResponseText = responseText.trim();

  let focusArea: string | undefined;
  if (mode === "initial" && essayScores?.dimensions) {
    const lowestDim = getLowestDimension(essayScores);
    focusArea = lowestDim ? getFocusArea(lowestDim[0]) : undefined;
  }

  return {
    message: trimmedResponseText,
    focusArea: mode === "initial" ? focusArea : undefined,
    clues: mode === "clues" ? trimmedResponseText : undefined,
    explanation: mode === "explanation" ? trimmedResponseText : undefined,
  };
}
