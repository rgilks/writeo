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
  teacherFeedback: Map<string, TeacherFeedback> = new Map()
): AssessmentResults {
  const parts: AssessmentPart[] = [];

  for (const part of request.parts) {
    // Get essay assessor results for this part (will be added to all answers)
    const essayAssessorResults: AssessorResult[] = [];
    if (essay?.results?.parts) {
      const essayPart = essay.results.parts.find((p) => p.part === part.part);
      if (essayPart?.answers && essayPart.answers.length > 0) {
        // Extract assessor-results from the first answer (scores are typically the same for all answers in a part)
        const firstAnswer = essayPart.answers[0];
        if (firstAnswer?.["assessor-results"]) {
          essayAssessorResults.push(...firstAnswer["assessor-results"]);
        }
      }
    }

    // Create answer results for each answer in the part
    const answerResults: AnswerResult[] = [];

    for (const answer of part.answers) {
      const assessorResults: AssessorResult[] = [];

      // Add essay assessor results (shared across all answers in the part)
      assessorResults.push(...essayAssessorResults);

      // Add LanguageTool errors for this answer
      const answerLtErrors = ltErrors.get(answer.id) ?? [];
      if (answerLtErrors.length > 0) {
        assessorResults.push({
          id: "T-GEC-LT",
          name: "LanguageTool (OSS)",
          type: "feedback",
          errors: answerLtErrors,
          meta: {
            language,
            engine: "LT-OSS",
            errorCount: answerLtErrors.length,
          },
        });
      }

      // Add LLM errors for this answer
      const answerLlmErrors = llmErrors.get(answer.id) ?? [];
      if (answerLlmErrors.length > 0) {
        assessorResults.push({
          id: "T-GEC-LLM",
          name: "AI Assessment",
          type: "feedback",
          errors: answerLlmErrors,
          meta: {
            engine: llmProvider === "groq" ? "Groq" : "OpenAI",
            model: aiModel,
            errorCount: answerLlmErrors.length,
          },
        });
      } else {
        // Log when LLM assessor is not added due to empty errors
        console.log(
          `[merge-results] LLM assessor not added for answer ${answer.id}: no errors found (provider: ${llmProvider})`
        );
      }

      // Add LLM feedback for this answer
      const feedback = llmFeedback.get(answer.id);
      if (feedback) {
        assessorResults.push({
          id: "T-AI-FEEDBACK",
          name: "AI Tutor Feedback",
          type: "feedback",
          meta: {
            relevance: feedback.relevance,
            feedback: feedback.feedback,
            engine: llmProvider === "groq" ? "Groq" : "OpenAI",
            model: aiModel,
          },
        });
      }

      // Add relevance check for this answer
      const relevance = relevanceChecks.get(answer.id);
      if (relevance) {
        assessorResults.push({
          id: "T-RELEVANCE-CHECK",
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
        });
      }

      // Add teacher feedback for this answer
      const teacher = teacherFeedback.get(answer.id);
      if (teacher) {
        assessorResults.push({
          id: "T-TEACHER-FEEDBACK",
          name: "Teacher's Feedback",
          type: "feedback",
          meta: {
            message: teacher.message,
            focusArea: teacher.focusArea,
            engine: llmProvider === "groq" ? "Groq" : "OpenAI",
            model: aiModel,
          },
        });
      }

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

  const questionTexts = new Map<string, string>();
  for (const part of request.parts) {
    for (const answer of part.answers) {
      if ((answer as any).question_text) {
        questionTexts.set(answer.id, (answer as any).question_text);
      }
    }
  }

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
