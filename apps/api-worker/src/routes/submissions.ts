import type { Context } from "hono";
import type { Env } from "../types/env";
import { processSubmission } from "../services/submission-processor";
import { errorResponse, ERROR_CODES } from "../utils/errors";
import { safeLogError, sanitizeError } from "../utils/logging";
import { getServices } from "../utils/context";
import { uuidStringSchema, formatZodMessage } from "../utils/zod";
import type { CreateSubmissionRequest } from "@writeo/shared";
import { z } from "zod";

// Schema for POST request body (includes submission_id)
const createSubmissionSchema = z.object({
  submissionId: z.string().uuid("Invalid submissionId format"),
  submission: z.array(z.any()),
  assessors: z.array(z.string()).optional(),
  storeResults: z.boolean().optional(),
});

export async function createSubmissionHandler(
  c: Context<{ Bindings: Env; Variables: { requestId?: string } }>,
) {
  try {
    const body = await c.req.json();
    const parsed = createSubmissionSchema.safeParse(body);

    if (!parsed.success) {
      return errorResponse(
        400,
        formatZodMessage(parsed.error, "Invalid request body"),
        c,
        ERROR_CODES.INVALID_SUBMISSION_FORMAT,
      );
    }

    // Extract submission_id from body
    const submissionId = parsed.data.submissionId;

    // Check if submission already exists
    const { storage } = getServices(c);
    const existing = await storage.getSubmission(submissionId);
    if (existing) {
      return errorResponse(
        409,
        "Submission already exists. Each submission must have a unique submissionId.",
        c,
        ERROR_CODES.SUBMISSION_EXISTS_DIFFERENT_CONTENT,
      );
    }

    // Create submission body
    const submissionBody: CreateSubmissionRequest = {
      submission: parsed.data.submission,
      assessors: parsed.data.assessors,
      storeResults: parsed.data.storeResults,
    };

    // Call processSubmission with the submission_id and body
    return await processSubmission(c, submissionId, submissionBody);
  } catch (error) {
    const sanitized = sanitizeError(error);
    safeLogError("Error creating submission", sanitized, c);
    return errorResponse(500, "Internal server error", c, ERROR_CODES.INTERNAL_SERVER_ERROR);
  }
}

export async function getSubmissionHandler(
  c: Context<{ Bindings: Env; Variables: { requestId?: string } }>,
) {
  const submissionIdResult = uuidStringSchema("submission_id").safeParse(
    c.req.param("submission_id"),
  );
  if (!submissionIdResult.success) {
    return errorResponse(
      400,
      formatZodMessage(submissionIdResult.error, "Invalid submission_id format"),
      c,
      ERROR_CODES.INVALID_UUID_FORMAT,
      "submission_id",
    );
  }
  const submissionId = submissionIdResult.data;

  try {
    const { storage } = getServices(c);
    const result = await storage.getResults(submissionId);

    if (!result) {
      const submission = await storage.getSubmission(submissionId);
      if (!submission) {
        return errorResponse(
          404,
          "Submission not found. Results are stored in your browser by default. If you enabled server storage, the results may have expired (90-day retention).",
          c,
          ERROR_CODES.SUBMISSION_NOT_FOUND,
        );
      }
      return c.json({ status: "pending" });
    }

    return c.json(result);
  } catch (error) {
    const sanitized = sanitizeError(error);
    safeLogError("Error fetching submission", sanitized, c);
    return errorResponse(500, "Internal server error", c, ERROR_CODES.INTERNAL_SERVER_ERROR);
  }
}
