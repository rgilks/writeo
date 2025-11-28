"use server";

import type { AssessmentData } from "../types/assessment";
import { retryWithBackoff } from "@writeo/shared";
import { getApiBase, getApiKey } from "../api-config";
import { getErrorMessage, makeSerializableError } from "../utils/error-handling";

type FeedbackMode = "clues" | "explanation";

interface RequestBody {
  answerId: string;
  mode: FeedbackMode;
  answerText: string;
  questionText?: string;
  assessmentData?: AssessmentData;
}

interface TeacherFeedbackResponse {
  message: string;
  focusArea?: string;
}

function validateRequest(
  submissionId: string,
  answerId: string,
  mode: string,
  answerText: string,
): void {
  if (!submissionId || !answerId || !mode || !answerText) {
    throw new Error("Missing required fields: submissionId, answerId, mode, answerText");
  }

  if (mode !== "clues" && mode !== "explanation") {
    throw new Error("Mode must be 'clues' or 'explanation'");
  }
}

export async function getTeacherFeedback(
  submissionId: string,
  answerId: string,
  mode: FeedbackMode,
  answerText: string,
  questionText?: string,
  assessmentData?: AssessmentData,
): Promise<TeacherFeedbackResponse> {
  try {
    validateRequest(submissionId, answerId, mode, answerText);

    const apiBase = getApiBase();
    const apiKey = getApiKey();

    if (apiKey === "MISSING_API_KEY" || apiBase === "MISSING_API_BASE_URL") {
      throw new Error("Server configuration error: API credentials not set");
    }

    const requestBody: RequestBody = {
      answerId,
      mode,
      answerText,
      ...(questionText && { questionText }),
      ...(assessmentData && { assessmentData }),
    };

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
    return {
      message: data.message || "",
      focusArea: data.focusArea,
    };
  } catch (error) {
    console.error("[getTeacherFeedback] Error:", error);
    throw makeSerializableError(error);
  }
}
