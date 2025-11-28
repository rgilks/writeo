"use server";

import type { AssessmentResults } from "@writeo/shared";
import { retryWithBackoff } from "@writeo/shared";
import { generateUUID } from "../utils/uuid-utils";
import { apiRequest } from "../utils/api-client";
import { getErrorMessage, makeSerializableError } from "../utils/error-handling";
import { getApiBase, getApiKey } from "../api-config";
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
  "question-number": number;
  "question-id": string;
  text: string;
  "question-text"?: string;
}

interface SubmissionBody {
  submission: Array<{
    part: string;
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
    "question-number": 1,
    "question-id": questionId,
    text: answerText,
    ...(trimmedQuestionText && { "question-text": trimmedQuestionText }),
  };

  const body: SubmissionBody = {
    submission: [
      {
        part: "1",
        answers: [answerPayload],
      },
    ],
    template: { name: "generic", version: 1 },
    storeResults,
  };

  const response = await apiRequest(`/text/submissions/${submissionId}`, "PUT", body);

  if (!response.ok) {
    throw new Error(`Failed to create submission: ${await getErrorMessage(response)}`);
  }

  const results = await response.json();
  return { submissionId, results };
}

async function extractErrorMessage(response: Response): Promise<string> {
  const errorText = await response.text();
  let errorMessage = `Failed to fetch results: HTTP ${response.status}`;

  try {
    const errorJson = JSON.parse(errorText);
    errorMessage = errorJson.error || errorJson.message || errorMessage;
  } catch {
    errorMessage = errorText || errorMessage;
  }

  return errorMessage;
}

function createNotFoundError(): Error & { status: number } {
  const error = new Error("Submission not found on server") as Error & { status: number };
  error.status = 404;
  return error;
}

export async function getSubmissionResults(
  submissionId: string,
): Promise<AssessmentResults | unknown> {
  const apiBase = getApiBase();
  const apiKey = getApiKey();

  if (apiKey === "MISSING_API_KEY" || apiBase === "MISSING_API_BASE_URL") {
    throw new Error("Server configuration error: API credentials not set");
  }

  const response = await retryWithBackoff(async () => {
    const res = await fetch(`${apiBase}/text/submissions/${submissionId}`, {
      headers: { Authorization: `Token ${apiKey}` },
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
    const errorMessage = await extractErrorMessage(response);
    throw new Error(errorMessage);
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
