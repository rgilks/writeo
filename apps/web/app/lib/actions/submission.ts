/**
 * Submission-related server actions
 */

"use server";

import { generateUUID } from "../utils/uuid-utils";
import { apiRequest } from "../utils/api-client";
import { getErrorMessage, makeSerializableError } from "../utils/error-handling";
import { retryWithBackoff } from "@writeo/shared";
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

export async function createSubmission(
  questionText: string,
  answerText: string,
  storeResults: boolean = false,
): Promise<{ submissionId: string; results: any }> {
  const submissionId = generateUUID();
  const questionId = generateUUID();
  const answerId = generateUUID();

  const trimmedQuestionText = questionText?.trim();

  const answerPayload: Record<string, unknown> = {
    id: answerId,
    "question-number": 1,
    "question-id": questionId,
    text: answerText,
  };

  if (trimmedQuestionText) {
    answerPayload["question-text"] = trimmedQuestionText;
  }

  const body: any = {
    submission: [
      {
        part: "1",
        answers: [
          answerPayload,
        ],
      },
    ],
    template: { name: "generic", version: 1 },
    storeResults: storeResults,
  };

  const response = await apiRequest(`/text/submissions/${submissionId}`, "PUT", body);

  if (!response.ok) {
    throw new Error(`Failed to create submission: ${await getErrorMessage(response)}`);
  }

  const results = await response.json();
  return { submissionId, results };
}

export async function getSubmissionResults(submissionId: string) {
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
      const error = new Error("Submission not found on server");
      (error as any).status = 404;
      throw error;
    }
    const errorText = await response.text();
    let errorMessage = `Failed to fetch results: HTTP ${response.status}`;
    try {
      const errorJson = JSON.parse(errorText);
      errorMessage = errorJson.error || errorJson.message || errorMessage;
    } catch {
      errorMessage = errorText || errorMessage;
    }
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

export async function pollSubmissionResults(
  submissionId: string,
  maxAttempts: number = 20,
  initialIntervalMs: number = 1000,
  parentSubmissionId?: string,
): Promise<any> {
  for (let attempts = 0; attempts < maxAttempts; attempts++) {
    try {
      const data = await getSubmissionResults(submissionId);
      if (data.status !== "pending") {
        if (parentSubmissionId) {
          const { getSubmissionResultsWithDraftTracking } = await import("./draft");
          return await getSubmissionResultsWithDraftTracking(submissionId, parentSubmissionId);
        }
        return data;
      }
    } catch (error) {
      if (attempts >= maxAttempts - 1) {
        throw makeSerializableError(error);
      }
    }

    const intervalMs = Math.min(initialIntervalMs * Math.pow(2, attempts), 10000);
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }

  throw new Error("Request timed out. Please try again.");
}
