"use server";

import type { AssessmentResults } from "@writeo/shared";
import { retryWithBackoff } from "@writeo/shared";
import { generateUUID } from "../utils/uuid-utils";
import { apiRequest } from "../utils/api-client";
import { getErrorMessage, makeSerializableError } from "../utils/error-handling";
import { z } from "zod";

const SubmissionEnvelopeSchema = z.object({
  status: z.string(),
  results: z
    .object({
      status: z.string(),
      results: z.unknown(),
    })
    .optional(),
  error: z.string().optional(),
  message: z.string().optional(),
});

interface AnswerPayload {
  id: string;
  questionId: string;
  text: string;
  questionText?: string | null;
}

interface SubmissionBody {
  submission: Array<{
    part: number;
    answers: AnswerPayload[];
  }>;
  template: { name: string; version: number };
  storeResults: boolean;
}

interface CreateSubmissionResult {
  submissionId: string;
  results: unknown;
}

export async function createSubmission(
  questionText: string,
  answerText: string,
  storeResults: boolean = false,
): Promise<CreateSubmissionResult> {
  const submissionId = generateUUID();
  const questionId = generateUUID();
  const answerId = generateUUID();

  const trimmedQuestionText = questionText?.trim();

  const answerPayload: AnswerPayload = {
    id: answerId,
    questionId: questionId,
    text: answerText,
    ...(trimmedQuestionText ? { questionText: trimmedQuestionText } : { questionText: null }),
  };

  const body: SubmissionBody = {
    submission: [
      {
        part: 1,
        answers: [answerPayload],
      },
    ],
    template: { name: "generic", version: 1 },
    storeResults,
  };

  const response = await apiRequest({
    endpoint: `/v1/text/submissions`,
    method: "POST",
    body: {
      submissionId,
      ...body,
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to create submission: ${await getErrorMessage(response)}`);
  }

  let data;
  try {
    data = await response.json();
  } catch (error) {
    console.error("[createSubmission] Failed to parse JSON:", error);
    console.error("[createSubmission] Response status:", response.status);
    console.error(
      "[createSubmission] Response headers:",
      Object.fromEntries(response.headers.entries()),
    );
    throw new Error(
      `Invalid response format: ${error instanceof Error ? error.message : "Unknown error"}`,
    );
  }

  // API returns AssessmentResults directly (with optional requestId field)
  // The response is the assessment results object itself, not wrapped in an envelope
  // Ensure we have a valid structure
  if (!data || typeof data !== "object") {
    throw new Error("Invalid response: expected an object");
  }

  return { submissionId, results: data };
}

function createNotFoundError(): Error & { status: number } {
  const error = new Error("Submission not found on server") as Error & { status: number };
  error.status = 404;
  return error;
}

export async function getSubmissionResults(
  submissionId: string,
): Promise<AssessmentResults | unknown> {
  const response = await retryWithBackoff(async () => {
    const res = await apiRequest({
      endpoint: `/v1/text/submissions/${submissionId}`,
      method: "GET",
    });

    if (res.status === 404) {
      return res;
    }

    if (!res.ok && res.status >= 500) {
      throw new Error(`Server error: HTTP ${res.status}. Please try again.`);
    }

    return res;
  });

  if (!response.ok) {
    if (response.status === 404) {
      throw createNotFoundError();
    }
    throw new Error(await getErrorMessage(response));
  }

  const data = await response.json();
  const parsed = SubmissionEnvelopeSchema.safeParse(data);

  if (parsed.success) {
    if (parsed.data.status === "success" && parsed.data.results) {
      return parsed.data.results;
    }
    return parsed.data;
  }

  return data;
}

function calculatePollingInterval(attempt: number, initialIntervalMs: number): number {
  return Math.min(initialIntervalMs * Math.pow(2, attempt), 10000);
}

export async function pollSubmissionResults(
  submissionId: string,
  maxAttempts: number = 20,
  initialIntervalMs: number = 1000,
  parentSubmissionId?: string,
): Promise<AssessmentResults | unknown> {
  for (let attempts = 0; attempts < maxAttempts; attempts++) {
    try {
      const data = await getSubmissionResults(submissionId);

      if (typeof data === "object" && data !== null && "status" in data) {
        const assessmentData = data as AssessmentResults;
        if (assessmentData.status !== "pending") {
          if (parentSubmissionId) {
            const { getSubmissionResultsWithDraftTracking } = await import("./draft");
            return await getSubmissionResultsWithDraftTracking(submissionId, parentSubmissionId);
          }
          return data;
        }
      }
    } catch (error) {
      if (attempts >= maxAttempts - 1) {
        throw makeSerializableError(error);
      }
    }

    const intervalMs = calculatePollingInterval(attempts, initialIntervalMs);
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }

  throw new Error("Request timed out. Please try again.");
}
