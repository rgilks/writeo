import type { Context } from "hono";
import type { Env } from "../types/env";
import { processSubmission } from "../services/submission-processor";
import { errorResponse } from "../utils/errors";
import { safeLogError, sanitizeError } from "../utils/logging";
import { StorageService } from "../services/storage";
import { uuidStringSchema, formatZodMessage } from "../utils/zod";

export async function processSubmissionHandler(c: Context<{ Bindings: Env }>) {
  return processSubmission(c);
}

export async function getSubmissionHandler(c: Context<{ Bindings: Env }>) {
  const submissionIdResult = uuidStringSchema("submission_id").safeParse(
    c.req.param("submission_id"),
  );
  if (!submissionIdResult.success) {
    return errorResponse(
      400,
      formatZodMessage(submissionIdResult.error, "Invalid submission_id format"),
      c,
    );
  }
  const submissionId = submissionIdResult.data;

  try {
    const storage = new StorageService(c.env.WRITEO_DATA, c.env.WRITEO_RESULTS);
    const result = await storage.getResults(submissionId);

    if (!result) {
      const submission = await storage.getSubmission(submissionId);
      if (!submission) {
        return errorResponse(
          404,
          "Submission not found. Results are stored in your browser by default. If you enabled server storage, the results may have expired (90-day retention).",
          c,
        );
      }
      return c.json({ status: "pending" });
    }

    return c.json(result);
  } catch (error) {
    const sanitized = sanitizeError(error);
    safeLogError("Error fetching submission", sanitized);
    return errorResponse(500, "Internal server error", c);
  }
}
