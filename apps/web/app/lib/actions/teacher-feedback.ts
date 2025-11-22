/**
 * Teacher feedback server actions
 */

"use server";

import { getApiBase, getApiKey } from "../api-config";
import { retryWithBackoff } from "../utils/retry-utils";
import { getErrorMessage, makeSerializableError } from "../utils/error-handling";

export async function getTeacherFeedback(
  submissionId: string,
  answerId: string,
  mode: "clues" | "explanation",
  answerText: string,
  questionText?: string,
  assessmentData?: any
): Promise<{ message: string; focusArea?: string }> {
  try {
    if (!submissionId || !answerId || !mode || !answerText) {
      throw new Error("Missing required fields: submissionId, answerId, mode, answerText");
    }

    if (mode !== "clues" && mode !== "explanation") {
      throw new Error("Mode must be 'clues' or 'explanation'");
    }

    const apiBase = getApiBase();
    const apiKey = getApiKey();

    if (apiKey === "MISSING_API_KEY" || apiBase === "MISSING_API_BASE_URL") {
      throw new Error("Server configuration error: API credentials not set");
    }

    const requestBody: any = {
      answerId,
      mode,
      answerText,
    };

    if (questionText) {
      requestBody.questionText = questionText;
    }
    if (assessmentData) {
      requestBody.assessmentData = assessmentData;
    }

    const response = await retryWithBackoff(async () => {
      const res = await fetch(`${apiBase}/text/submissions/${submissionId}/teacher-feedback`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Token ${apiKey}`,
        },
        body: JSON.stringify(requestBody),
      });

      if (!res.ok && res.status >= 500) {
        throw new Error(`Server error: HTTP ${res.status}. Please try again.`);
      }

      return res;
    });

    if (!response.ok) {
      throw new Error(`Failed to get feedback: ${await getErrorMessage(response)}`);
    }

    const data = await response.json();
    const feedbackResult = {
      message: data.message || "",
      focusArea: data.focusArea,
    };

    if (typeof window !== "undefined") {
      try {
        const storedResults = localStorage.getItem(`results_${submissionId}`);
        if (storedResults) {
          const results = JSON.parse(storedResults) as any;

          if (results.results?.parts?.[0]?.answers?.[0]) {
            const firstAnswer = results.results.parts[0].answers[0];
            if (!firstAnswer["assessor-results"]) {
              firstAnswer["assessor-results"] = [];
            }

            let teacherAssessor = firstAnswer["assessor-results"].find(
              (a: any) => a.id === "T-TEACHER-FEEDBACK"
            );

            if (!teacherAssessor) {
              teacherAssessor = {
                id: "T-TEACHER-FEEDBACK",
                name: "Teacher's Feedback",
                type: "feedback",
                meta: {},
              };
              firstAnswer["assessor-results"].push(teacherAssessor);
            }

            const existingMeta = (teacherAssessor.meta || {}) as Record<string, any>;
            teacherAssessor.meta = {
              ...existingMeta,
              message: existingMeta.message || feedbackResult.message,
              focusArea: feedbackResult.focusArea || existingMeta.focusArea,
              ...(mode === "clues" && { cluesMessage: feedbackResult.message }),
              ...(mode === "explanation" && { explanationMessage: feedbackResult.message }),
            };

            localStorage.setItem(`results_${submissionId}`, JSON.stringify(results));
          }
        }
      } catch (storageError) {
        console.warn("[getTeacherFeedback] Failed to store feedback locally:", storageError);
      }
    }

    return feedbackResult;
  } catch (error) {
    console.error("[getTeacherFeedback] Error:", error);
    throw makeSerializableError(error);
  }
}
