/**
 * Prompt building utilities for AI assessment
 */

import { truncateQuestionText } from "../../utils/text-processing";

const MAX_TEXT_LENGTH_FOR_GRAMMAR_CHECK = 12000;

export function truncateAnswerText(answerText: string): string {
  return answerText.length > MAX_TEXT_LENGTH_FOR_GRAMMAR_CHECK
    ? answerText.slice(0, MAX_TEXT_LENGTH_FOR_GRAMMAR_CHECK) + "\n\n[... text continues ...]"
    : answerText;
}

export function buildAssessmentPrompt(questionText: string, answerText: string): string {
  const truncatedAnswerText = truncateAnswerText(answerText);
  const truncatedQuestionText = truncateQuestionText(questionText);

  return `Find ALL grammar, spelling, style, and punctuation errors in the student's answer.

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
}

export function getMaxTextLength(): number {
  return MAX_TEXT_LENGTH_FOR_GRAMMAR_CHECK;
}
