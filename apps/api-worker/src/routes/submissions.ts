import type { Context } from "hono";
import type { Env } from "../types/env";
import { processSubmission } from "../services/submission-processor";

export async function processSubmissionHandler(c: Context<{ Bindings: Env }>) {
  return processSubmission(c);
}
