import type {
  AssessmentResults,
  LanguageToolError,
  AssessorResult,
  AssessmentPart,
  AnswerResult,
  ModalRequest,
} from "@writeo/shared";
import type { AIFeedback, TeacherFeedback } from "./feedback";
import type { RelevanceCheck } from "./relevance";

const MAX_LLM_ERRORS = 10;
const ASSESSOR_IDS = {
  LANGUAGE_TOOL: "T-GEC-LT",
  LLM: "T-GEC-LLM",
  AI_FEEDBACK: "T-AI-FEEDBACK",
  RELEVANCE: "T-RELEVANCE-CHECK",
  TEACHER: "T-TEACHER-FEEDBACK",
} as const;

function extractEssayAssessorResults(
  essay: AssessmentResults | null,
  partId: string | number,
): AssessorResult[] {
  if (!essay?.results?.parts) {
    console.warn(
      `[merge-results] Essay assessment missing for part ${partId} - essay assessor results will be missing`,
    );
    return [];
  }

  const essayPart = essay.results.parts.find((p) => String(p.part) === String(partId));
  if (!essayPart?.answers?.length) {
    console.warn(
      `[merge-results] Essay part ${partId} has no answers - essay assessor results will be missing`,
    );
    return [];
  }

  const firstAnswer = essayPart.answers[0];
  const assessorResults = firstAnswer?.["assessor-results"];
  if (!assessorResults) {
    console.warn(
      `[merge-results] Essay answer ${firstAnswer?.id} has no assessor-results - essay assessor results will be missing`,
    );
    return [];
  }

  console.log(
    `[merge-results] Extracted ${assessorResults.length} essay assessor result(s) for part ${partId}`,
  );
  return [...assessorResults];
}

function sortLLMErrorsBySignificance(errors: LanguageToolError[]): LanguageToolError[] {
  return [...errors].sort((a, b) => {
    // First, prioritize errors over warnings
    if (a.severity !== b.severity) {
      return a.severity === "error" ? -1 : 1;
    }
    // Then by confidence score (higher is better)
    const aConfidence = a.confidenceScore ?? 0;
    const bConfidence = b.confidenceScore ?? 0;
    if (aConfidence !== bConfidence) {
      return bConfidence - aConfidence;
    }
    // Finally, by position (earlier errors first)
    return a.start - b.start;
  });
}

function createLanguageToolAssessor(errors: LanguageToolError[], language: string): AssessorResult {
  return {
    id: ASSESSOR_IDS.LANGUAGE_TOOL,
    name: "LanguageTool (OSS)",
    type: "feedback",
    errors,
    meta: {
      language,
      engine: "LT-OSS",
      errorCount: errors.length,
    },
  };
}

function createLLMAssessor(
  errors: LanguageToolError[],
  llmProvider: string,
  aiModel: string,
): AssessorResult {
  const sortedErrors = sortLLMErrorsBySignificance(errors);
  const topErrors = sortedErrors.slice(0, MAX_LLM_ERRORS);

  return {
    id: ASSESSOR_IDS.LLM,
    name: "AI Assessment",
    type: "feedback",
    errors: topErrors,
    meta: {
      engine: llmProvider === "groq" ? "Groq" : "OpenAI",
      model: aiModel,
      errorCount: topErrors.length,
      totalErrorsFound: errors.length,
    },
  };
}

function createAIFeedbackAssessor(
  feedback: AIFeedback,
  llmProvider: string,
  aiModel: string,
): AssessorResult {
  return {
    id: ASSESSOR_IDS.AI_FEEDBACK,
    name: "AI Tutor Feedback",
    type: "feedback",
    meta: {
      relevance: feedback.relevance,
      feedback: feedback.feedback,
      engine: llmProvider === "groq" ? "Groq" : "OpenAI",
      model: aiModel,
    },
  };
}

function createRelevanceAssessor(relevance: RelevanceCheck): AssessorResult {
  return {
    id: ASSESSOR_IDS.RELEVANCE,
    name: "Answer Relevance Check",
    type: "feedback",
    meta: {
      addressesQuestion: relevance.addressesQuestion,
      similarityScore: relevance.score,
      threshold: relevance.threshold,
      engine: "Cloudflare Workers AI",
      model: "bge-base-en-v1.5",
      method: "embeddings + cosine similarity",
    },
  };
}

function createTeacherFeedbackAssessor(
  teacher: TeacherFeedback,
  llmProvider: string,
  aiModel: string,
): AssessorResult {
  return {
    id: ASSESSOR_IDS.TEACHER,
    name: "Teacher's Feedback",
    type: "feedback",
    meta: {
      message: teacher.message,
      focusArea: teacher.focusArea,
      engine: llmProvider === "groq" ? "Groq" : "OpenAI",
      model: aiModel,
    },
  };
}

function buildAssessorResults(
  answerId: string,
  essayAssessorResults: AssessorResult[],
  ltErrors: LanguageToolError[],
  llmErrors: LanguageToolError[],
  llmFeedback: AIFeedback | undefined,
  relevance: RelevanceCheck | undefined,
  teacher: TeacherFeedback | undefined,
  language: string,
  llmProvider: string,
  aiModel: string,
): AssessorResult[] {
  const assessorResults: AssessorResult[] = [...essayAssessorResults];

  if (ltErrors.length > 0) {
    assessorResults.push(createLanguageToolAssessor(ltErrors, language));
  }

  if (llmErrors.length > 0) {
    assessorResults.push(createLLMAssessor(llmErrors, llmProvider, aiModel));
  } else {
    console.log(
      `[merge-results] LLM assessor not added for answer ${answerId}: no errors found (provider: ${llmProvider})`,
    );
  }

  if (llmFeedback) {
    assessorResults.push(createAIFeedbackAssessor(llmFeedback, llmProvider, aiModel));
  }

  if (relevance) {
    assessorResults.push(createRelevanceAssessor(relevance));
  }

  if (teacher) {
    assessorResults.push(createTeacherFeedbackAssessor(teacher, llmProvider, aiModel));
  }

  return assessorResults;
}

function extractQuestionTexts(request: ModalRequest): Map<string, string> {
  const questionTexts = new Map<string, string>();
  for (const part of request.parts) {
    for (const answer of part.answers) {
      if (answer.question_text) {
        questionTexts.set(answer.id, answer.question_text);
      }
    }
  }
  return questionTexts;
}

export function mergeAssessmentResults(
  essay: AssessmentResults | null,
  ltErrors: Map<string, LanguageToolError[]>,
  llmErrors: Map<string, LanguageToolError[]>,
  answerTexts: Map<string, string>,
  request: ModalRequest,
  language: string,
  aiModel: string = "gpt-4o-mini",
  llmProvider: string = "openai",
  llmFeedback: Map<string, AIFeedback> = new Map(),
  relevanceChecks: Map<string, RelevanceCheck> = new Map(),
  teacherFeedback: Map<string, TeacherFeedback> = new Map(),
): AssessmentResults {
  const parts: AssessmentPart[] = [];

  for (const part of request.parts) {
    const essayAssessorResults = extractEssayAssessorResults(essay, part.part);
    const answerResults: AnswerResult[] = [];

    for (const answer of part.answers) {
      const assessorResults = buildAssessorResults(
        answer.id,
        essayAssessorResults,
        ltErrors.get(answer.id) ?? [],
        llmErrors.get(answer.id) ?? [],
        llmFeedback.get(answer.id),
        relevanceChecks.get(answer.id),
        teacherFeedback.get(answer.id),
        language,
        llmProvider,
        aiModel,
      );

      answerResults.push({
        id: answer.id,
        "assessor-results": assessorResults,
      });
    }

    parts.push({
      part: part.part,
      status: answerResults.some((ar) => ar["assessor-results"].length > 0) ? "success" : "error",
      answers: answerResults,
    });
  }

  const questionTexts = extractQuestionTexts(request);
  const meta: Record<string, unknown> = {
    answerTexts: Object.fromEntries(answerTexts),
    questionTexts: questionTexts.size > 0 ? Object.fromEntries(questionTexts) : undefined,
  };

  return {
    status: parts.some((p) => p.status === "success") ? "success" : "error",
    results: { parts },
    template: request.template,
    meta,
  };
}
