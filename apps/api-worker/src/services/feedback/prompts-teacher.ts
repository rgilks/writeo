/**
 * Teacher feedback prompt building
 */

import { truncateEssayText } from "../../utils/text-processing";
import {
  buildTeacherScoreContext,
  buildTeacherErrorContext,
  buildTeacherRelevanceContext,
  buildWordCountContext,
} from "./context";
import type { EssayScores, FeedbackError, RelevanceCheck } from "./types";

export function buildTeacherFeedbackPrompt(
  questionText: string,
  answerText: string,
  mode: "initial" | "clues" | "explanation",
  essayScores?: EssayScores,
  languageToolErrors?: Array<Pick<FeedbackError, "errorType" | "category">>,
  llmErrors?: Array<Pick<FeedbackError, "errorType" | "category">>,
  relevanceCheck?: Pick<RelevanceCheck, "addressesQuestion" | "score">,
): string {
  const truncatedAnswerText = truncateEssayText(answerText);
  const scoreContext = buildTeacherScoreContext(essayScores);
  const errorContext = buildTeacherErrorContext(languageToolErrors, llmErrors);
  const relevanceContext = buildTeacherRelevanceContext(relevanceCheck);
  const wordCountContext = buildWordCountContext(truncatedAnswerText);

  if (mode === "initial") {
    return `You are a professional writing tutor specializing in academic argumentative writing. Give clear, direct feedback to help the student improve. Be constructive and specific. Keep it brief (2-3 sentences max). Don't mention technical terms like "CEFR" or "band scores" - focus on actionable improvements.

<question>
${questionText}
</question>

<student_answer>
${truncatedAnswerText}
</student_answer>

${scoreContext}${wordCountContext}${errorContext}${relevanceContext}

Focus your feedback on:
- Whether they addressed all parts of the question
- Essay structure (introduction, body paragraphs, conclusion)
- How well ideas are connected and organized
- Vocabulary range and accuracy
- Grammar accuracy

Give your feedback as a professional tutor would - clear, direct, and focused on the most important area to improve.`;
  }

  if (mode === "clues") {
    return `You are a professional writing tutor. The student tried again but still has issues. Give them specific clues (not full answers) to guide them. Be constructive. Keep it brief (2-3 sentences).

<question>
${questionText}
</question>

<student_answer>
${truncatedAnswerText}
</student_answer>

${scoreContext}${wordCountContext}${errorContext}${relevanceContext}

Give clues that help them identify the problems themselves. Focus on essay structure, addressing all parts of the question, connecting ideas, or vocabulary/grammar issues. Provide guidance without giving away all the answers.`;
  }

  return `You are an experienced writing instructor analyzing a student's essay for another teacher. Provide a comprehensive, structured analysis using markdown formatting. This analysis should help the teacher understand the student's performance across all dimensions.

<question>
${questionText}
</question>

<student_answer>
${truncatedAnswerText}
</student_answer>

${scoreContext}${wordCountContext}${errorContext}${relevanceContext}

Write a detailed teacher-to-teacher analysis in markdown format with the following structure:

## Overall Assessment
Brief summary of the student's performance (2-3 sentences).

## Task Achievement
- Did the student address all parts of the question?
- Are the ideas relevant to the topic?
- Are arguments well-developed with specific examples?
- What specific parts of the question were addressed or missed?
- Provide specific examples from the essay.

## Structure & Organization
- Is there a clear introduction, body, and conclusion?
- Are paragraphs well-structured with topic sentences?
- Is the essay logically organized?
- Are transitions used effectively between ideas?
- Identify specific structural issues with examples.

## Coherence & Cohesion
- How well are ideas connected?
- Are linking words and phrases used appropriately?
- Is there a clear flow of ideas throughout?
- Are there any abrupt shifts or disconnected ideas?
- Provide specific examples of good or problematic connections.

## Vocabulary
- What is the range and accuracy of vocabulary?
- Are words used appropriately in context?
- Is there repetition that could be improved?
- Are there any inappropriate word choices?
- Provide specific examples of vocabulary usage (both good and problematic).

## Grammar & Language Accuracy
- What are the main grammar issues?
- Are sentence structures varied and appropriate?
- Are there patterns of errors (e.g., subject-verb agreement, articles)?
- How do grammar errors affect clarity?
- Provide specific examples of errors with corrections.

## Recommendations for Improvement
Prioritized list of 3-5 specific areas the student should focus on, with brief explanations of why each matters.

Use markdown formatting (headers, bullet points, bold for emphasis). Be specific and cite examples from the essay. Write as if explaining to a colleague who will use this to help the student improve.`;
}
