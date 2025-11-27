/**
 * Combined feedback prompt building
 */

import { truncateEssayText, truncateQuestionText } from "../../utils/text-processing";
import {
  buildEssayContext,
  buildGrammarContext,
  buildRelevanceContext,
  getLowestDimension,
  getFocusArea,
} from "./context";
import type { EssayScores, FeedbackError, RelevanceCheck } from "./types";

export function buildCombinedFeedbackPrompt(
  questionText: string,
  answerText: string,
  essayScores?: EssayScores,
  languageToolErrors?: FeedbackError[],
  llmErrors?: FeedbackError[],
  relevanceCheck?: RelevanceCheck,
): string {
  const truncatedAnswerText = truncateEssayText(answerText);
  const truncatedQuestionText = truncateQuestionText(questionText);
  const essayContext = buildEssayContext(essayScores);
  const grammarContext = buildGrammarContext(languageToolErrors, llmErrors);
  const relevanceContext = buildRelevanceContext(relevanceCheck);
  const lowestDim = getLowestDimension(essayScores);
  const focusArea = getFocusArea(lowestDim?.[0]);

  return `You are an expert English language tutor specializing in academic argumentative writing. Analyze the following essay answer and provide TWO types of feedback:

<question>
${truncatedQuestionText}
</question>

<answer>
${truncatedAnswerText}
</answer>

${essayContext}${grammarContext}${relevanceContext}

Provide feedback in TWO formats:

1. DETAILED FEEDBACK:
Analyze the essay across these key areas:

- Task Achievement:
* Does the answer address ALL parts of the question?
* Is there a clear position/opinion stated (for opinion questions)?
* Are ideas fully developed with specific examples and explanations?
* For discussion questions: Are both sides addressed before giving an opinion?
* For problem-solution: Are problems clearly identified and solutions provided?

- Structure & Organization:
* Is there a clear introduction that presents the topic and position?
* Are body paragraphs well-organized with topic sentences?
* Does each paragraph focus on one main idea?
* Is there a conclusion that summarizes the main points?
* Is the essay logically structured and easy to follow?

- Coherence & Cohesion:
* Are ideas connected smoothly using linking words (however, furthermore, therefore, etc.)?
* Do paragraphs flow logically from one to the next?
* Are pronouns and referencing used correctly?
* Is the writing cohesive and easy to follow?

- Vocabulary:
* Is there a good range of vocabulary (avoiding repetition)?
* Are words used accurately and appropriately?
* Are there attempts at less common vocabulary?
* Is word choice precise and natural?

- Grammar:
* Is grammar accurate?
* Is there a variety of sentence structures (simple, compound, complex)?
* Are tenses used correctly?

Provide:
- Relevance: Does the answer fully address the question? (true/false, score 0.0-1.0, brief explanation)
- Strengths: 2-3 specific things the student did well (be specific about what worked)
- Improvements: 2-3 specific, actionable areas to improve (prioritize the most important issues)
- Overall: A brief summary comment (1-2 sentences) that focuses on improvement and next steps

2. TEACHER FEEDBACK (simple, direct, professional):
- Give clear, direct feedback as a professional writing tutor would
- Be honest and constructive
- Keep it brief (2-3 sentences max)
- Don't mention technical terms like "CEFR" or "band scores"
- Focus on the most important area to improve
- Provide one specific, actionable suggestion

Respond ONLY with valid JSON (no markdown, no explanations):
{
"detailed": {
"relevance": {
"addressesQuestion": true/false,
"score": 0.0-1.0,
"explanation": "brief explanation of how well the answer addresses the question, including which parts were addressed and which might be missing"
},
"feedback": {
"strengths": ["specific strength 1 (e.g., 'Clear position stated in introduction' or 'Good use of linking words like however and furthermore')", "specific strength 2", "specific strength 3"],
"improvements": ["specific improvement 1 (e.g., 'Develop your second paragraph with a concrete example' or 'Use more varied vocabulary instead of repeating 'important' - try 'crucial', 'significant', 'vital')", "specific improvement 2", "specific improvement 3"],
"overall": "overall comment focusing on improvement and next steps"
}
},
"teacher": {
"message": "clear, direct teacher feedback (2-3 sentences, professional and constructive, focusing on the most important improvement)",
"focusArea": "${focusArea}"
}
}`;
}
