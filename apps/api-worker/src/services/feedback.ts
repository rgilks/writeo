import { callLLMAPI, parseLLMProvider, getDefaultModel, getAPIKey, type LLMProvider } from "./llm";
import { truncateEssayText, truncateQuestionText } from "../utils/text-processing";
import {
  MAX_TOKENS_DETAILED_FEEDBACK,
  MAX_TOKENS_TEACHER_FEEDBACK_INITIAL,
  MAX_TOKENS_TEACHER_FEEDBACK_EXPLANATION,
} from "../utils/constants";

// Types for AI feedback
export interface AIFeedback {
  relevance: {
    addressesQuestion: boolean;
    score: number;
    explanation: string;
  };
  feedback: {
    strengths: string[];
    improvements: string[];
    overall: string;
  };
}

export interface TeacherFeedback {
  message: string;
  focusArea?: string;
  clues?: string;
  explanation?: string;
}

export interface CombinedFeedback {
  detailed: AIFeedback;
  teacher: TeacherFeedback;
}

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
  // Truncate very long essays to keep costs under control (max ~2500 words or ~15000 chars)
  // This prevents excessive token usage while preserving enough context for meaningful feedback
  const truncatedAnswerText = truncateEssayText(answerText);
  const truncatedQuestionText = truncateQuestionText(questionText);

  let essayContext = "";
  if (essayScores) {
    essayContext = `\n\nAssessment Results:
- Overall Score: ${essayScores.overall ?? essayScores.dimensions?.Overall ?? "N/A"} / 9.0
- CEFR Level: ${essayScores.label ?? "N/A"}
- Task Achievement (TA): ${essayScores.dimensions?.TA ?? "N/A"} / 9.0
- Coherence & Cohesion (CC): ${essayScores.dimensions?.CC ?? "N/A"} / 9.0
- Vocabulary: ${essayScores.dimensions?.Vocab ?? "N/A"} / 9.0
- Grammar: ${essayScores.dimensions?.Grammar ?? "N/A"} / 9.0`;
  }

  const allErrors = [...(languageToolErrors || []), ...(llmErrors || [])];
  let grammarContext = "";
  if (allErrors.length > 0) {
    const errorSummary = allErrors
      .slice(0, 10)
      .map((err, idx) => {
        const source = languageToolErrors?.includes(err as any) ? "LanguageTool" : "AI Assessment";
        return `${idx + 1}. [${source}] ${err.message} (${err.category})${err.suggestions ? ` - Suggestions: ${err.suggestions.slice(0, 2).join(", ")}` : ""}`;
      })
      .join("\n");
    grammarContext = `\n\nGrammar & Language Issues Found (${allErrors.length} total):\n${errorSummary}${allErrors.length > 10 ? `\n... and ${allErrors.length - 10} more issues` : ""}`;
  } else if (languageToolErrors !== undefined || llmErrors !== undefined) {
    grammarContext = "\n\nGrammar & Language: No errors detected by LanguageTool or AI assessment.";
  }

  // Add relevance check context if available
  let relevanceContext = "";
  if (relevanceCheck) {
    relevanceContext = `\n\nAnswer Relevance: ${relevanceCheck.addressesQuestion ? "The answer addresses the question" : "The answer may not fully address the question"} (similarity score: ${relevanceCheck.score.toFixed(2)}, threshold: ${relevanceCheck.threshold}).`;
  }

  const overall = essayScores?.overall ?? essayScores?.dimensions?.Overall ?? 0;
  const lowestDim = essayScores?.dimensions
    ? Object.entries(essayScores.dimensions)
        .filter(([k]) => k !== "Overall")
        .sort(([, a], [, b]) => (a ?? 0) - (b ?? 0))[0]
    : undefined;

  const prompt = `You are an expert English language tutor specializing in academic argumentative writing. Analyze the following essay answer and provide TWO types of feedback:

Question: ${truncatedQuestionText}

Answer: ${truncatedAnswerText}${essayContext}${grammarContext}${relevanceContext}

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
"focusArea": "${lowestDim ? (lowestDim[0] === "TA" ? "answering all parts of the question completely" : lowestDim[0] === "CC" ? "organizing your ideas into clear paragraphs and connecting them smoothly" : lowestDim[0] === "Vocab" ? "using a wider range of vocabulary and avoiding repetition" : "improving grammar accuracy and using more varied sentence structures") : "this area"}"
}
}`;

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

  const trimmedResponseText = responseText.trim();
  const jsonMatch =
    trimmedResponseText.match(/```(?:json)?\s*(\{[\s\S]*\})\s*```/) ||
    trimmedResponseText.match(/(\{[\s\S]*\})/);

  if (jsonMatch) {
    try {
      const parsed = JSON.parse(jsonMatch[1]) as CombinedFeedback;
      if (!parsed.detailed || !parsed.teacher) {
        throw new Error("Missing required fields");
      }
      if (!parsed.detailed.relevance || !parsed.detailed.feedback) {
        throw new Error("Detailed feedback missing required fields");
      }
      if (!parsed.teacher.message) {
        throw new Error("Teacher feedback missing message");
      }
      return parsed;
    } catch (parseError) {
      throw new Error(
        `Failed to parse AI JSON response: ${parseError instanceof Error ? parseError.message : String(parseError)}`
      );
    }
  }

  throw new Error(`Could not extract JSON from AI response`);
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export async function getCombinedFeedbackWithRetry(
  params: {
    llmProvider: LLMProvider;
    apiKey: string;
    questionText: string;
    answerText: string;
    modelName: string;
    essayScores?: {
      overall?: number;
      dimensions?: { TA?: number; CC?: number; Vocab?: number; Grammar?: number; Overall?: number };
      label?: string;
    };
    languageToolErrors?: Array<{
      message: string;
      category: string;
      suggestions?: string[];
      start: number;
      end: number;
    }>;
    llmErrors?: Array<{
      message: string;
      category: string;
      suggestions?: string[];
      start: number;
      end: number;
      errorType?: string;
    }>;
    relevanceCheck?: { addressesQuestion: boolean; score: number; threshold: number };
  },
  options?: { maxAttempts?: number; baseDelayMs?: number }
): Promise<CombinedFeedback> {
  const maxAttempts = options?.maxAttempts ?? 3;
  const baseDelayMs = options?.baseDelayMs ?? 300;
  let attempt = 0;
  let lastError: Error | undefined;

  while (attempt < maxAttempts) {
    attempt += 1;
    try {
      const feedback = await getCombinedFeedback(
        params.llmProvider,
        params.apiKey,
        params.questionText,
        params.answerText,
        params.modelName,
        params.essayScores,
        params.languageToolErrors,
        params.llmErrors,
        params.relevanceCheck
      );

      if (!feedback?.detailed || !feedback?.teacher) {
        throw new Error("Combined feedback returned incomplete data");
      }

      return feedback;
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      if (attempt < maxAttempts) {
        const delay = baseDelayMs * attempt;
        await sleep(delay);
      }
    }
  }

  throw new Error(
    `Failed after ${maxAttempts} attempts${lastError ? `: ${lastError.message}` : ""}`
  );
}

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
  // Truncate very long essays to keep context concise (max ~2500 words or ~15000 chars)
  // This keeps prompts efficient while preserving enough context for meaningful feedback
  const truncatedAnswerText = truncateEssayText(answerText);

  let scoreContext = "";
  if (essayScores) {
    const overall = essayScores.overall ?? essayScores.dimensions?.Overall ?? 0;
    const lowestDim = Object.entries(essayScores.dimensions || {})
      .filter(([k]) => k !== "Overall")
      .sort(([, a], [, b]) => (a ?? 0) - (b ?? 0))[0];

    scoreContext = `\n\nStudent's performance:\n- Overall score: ${overall.toFixed(1)} / 9.0${lowestDim ? `\n- Weakest area: ${lowestDim[0]} (${lowestDim[1]?.toFixed(1)} / 9.0)` : ""}`;
  }

  // Combine errors from both sources for comprehensive feedback
  // Keep error context minimal but informative - count + most common types
  const allErrors = [...(languageToolErrors || []), ...(llmErrors || [])];
  let errorContext = "";
  if (allErrors.length > 0) {
    // Identify most common error types (helpful for teacher to know what to focus on)
    const errorTypes = new Map<string, number>();
    allErrors.forEach((err) => {
      const errWithType = err as { errorType?: string; category: string };
      const type = errWithType.errorType || errWithType.category || "Other";
      errorTypes.set(type, (errorTypes.get(type) || 0) + 1);
    });
    const topErrorTypes = Array.from(errorTypes.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 2)
      .map(([type, count]) => `${type} (${count})`)
      .join(", ");

    errorContext = `\n\nFound ${allErrors.length} grammar/language issue${allErrors.length > 1 ? "s" : ""}${topErrorTypes ? `, mostly: ${topErrorTypes}` : ""}.`;
  }

  // Add relevance check context if available (helps with Task Achievement feedback)
  let relevanceContext = "";
  if (relevanceCheck) {
    const relevancePercent = (relevanceCheck.score * 100).toFixed(0);
    if (relevanceCheck.addressesQuestion) {
      relevanceContext = `\n\nThe answer addresses the question well (relevance: ${relevancePercent}%).`;
    } else {
      relevanceContext = `\n\nThe answer may not fully address the question (relevance: ${relevancePercent}% - consider if all parts of the question were answered).`;
    }
  }

  // Add word count context (helpful for teacher to assess length appropriateness)
  const wordCount = truncatedAnswerText.split(/\s+/).filter((w) => w.length > 0).length;
  const wordCountContext =
    wordCount > 0 ? `\n\nEssay length: ${wordCount} word${wordCount !== 1 ? "s" : ""}.` : "";

  let prompt = "";
  if (mode === "initial") {
    prompt = `You are a professional writing tutor specializing in academic argumentative writing. Give clear, direct feedback to help the student improve. Be constructive and specific. Keep it brief (2-3 sentences max). Don't mention technical terms like "CEFR" or "band scores" - focus on actionable improvements.

Question: ${questionText}

Student's answer: ${truncatedAnswerText}${scoreContext}${wordCountContext}${errorContext}${relevanceContext}

Focus your feedback on:
- Whether they addressed all parts of the question
- Essay structure (introduction, body paragraphs, conclusion)
- How well ideas are connected and organized
- Vocabulary range and accuracy
- Grammar accuracy

Give your feedback as a professional tutor would - clear, direct, and focused on the most important area to improve.`;
  } else if (mode === "clues") {
    prompt = `You are a professional writing tutor. The student tried again but still has issues. Give them specific clues (not full answers) to guide them. Be constructive. Keep it brief (2-3 sentences).

Question: ${questionText}

Student's answer: ${truncatedAnswerText}${scoreContext}${wordCountContext}${errorContext}${relevanceContext}

Give clues that help them identify the problems themselves. Focus on essay structure, addressing all parts of the question, connecting ideas, or vocabulary/grammar issues. Provide guidance without giving away all the answers.`;
  } else {
    // Explanation mode: Detailed, structured analysis for teachers
    prompt = `You are an experienced writing instructor analyzing a student's essay for another teacher. Provide a comprehensive, structured analysis using markdown formatting. This analysis should help the teacher understand the student's performance across all dimensions.

Question: ${questionText}

Student's answer: ${truncatedAnswerText}${scoreContext}${wordCountContext}${errorContext}${relevanceContext}

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
    const lowestDim = Object.entries(essayScores.dimensions)
      .filter(([k]) => k !== "Overall")
      .sort(([, a], [, b]) => (a ?? 0) - (b ?? 0))[0];

    if (lowestDim) {
      const simpleNames: Record<string, string> = {
        TA: "answering all parts of the question completely and developing your ideas fully",
        CC: "organizing your ideas into clear paragraphs and connecting them smoothly with linking words",
        Vocab: "using a wider range of vocabulary and avoiding repetition",
        Grammar: "improving grammar accuracy and using more varied sentence structures",
      };
      focusArea = simpleNames[lowestDim[0]] || "this area";
    }
  }

  return {
    message: trimmedResponseText,
    focusArea: mode === "initial" ? focusArea : undefined,
    clues: mode === "clues" ? trimmedResponseText : undefined,
    explanation: mode === "explanation" ? trimmedResponseText : undefined,
  };
}
