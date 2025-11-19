import { Hono } from "hono";
import { cors } from "hono/cors";
import type { Env, ExecutionContext } from "./types/env";
import { authenticate } from "./middleware/auth";
import { rateLimit } from "./middleware/rate-limit";
import { securityHeaders, getCorsOrigin } from "./middleware/security";
import { questionsRouter } from "./routes/questions";
import { healthRouter } from "./routes/health";
import { feedbackRouter } from "./routes/feedback";
import { errorResponse } from "./utils/errors";
import { safeLogError, sanitizeError } from "./utils/logging";
import { isValidUUID } from "@writeo/shared";
import { StorageService } from "./services/storage";
import { processSubmissionHandler } from "./routes/submissions";

const app = new Hono<{
  Bindings: Env;
  Variables: { executionCtx?: ExecutionContext };
}>();

// CORS
app.use(
  "*",
  cors({
    origin: (origin, c) => getCorsOrigin(origin, c.env.ALLOWED_ORIGINS),
    allowMethods: ["GET", "PUT", "POST", "OPTIONS"],
    allowHeaders: ["Content-Type", "Authorization"],
    maxAge: 86400,
  })
);

// Security headers
app.use("*", securityHeaders);

// Rate limiting
app.use("*", rateLimit);

// Execution context middleware
app.use("*", async (c, next) => {
  const ctx = (c.env as any).__executionCtx as ExecutionContext | undefined;
  if (ctx) {
    c.set("executionCtx", ctx);
  }
  await next();
});

// Public routes (before auth)
app.route("/", healthRouter);

// Authentication middleware
app.use("*", authenticate);

// Protected routes
app.route("/", questionsRouter);
app.route("/", feedbackRouter);

// Submission routes
app.put("/text/submissions/:submission_id", processSubmissionHandler);

app.get("/text/submissions/:submission_id", async (c) => {
  const submissionId = c.req.param("submission_id");
  if (!isValidUUID(submissionId)) {
    return errorResponse(400, "Invalid submission_id format", c);
  }

  try {
    const storage = new StorageService(c.env.WRITEO_DATA, c.env.WRITEO_RESULTS);
    const result = await storage.getResults(submissionId);

    if (!result) {
      const submission = await storage.getSubmission(submissionId);
      if (!submission) {
        return errorResponse(404, "Submission not found", c);
      }
      return c.json({ status: "pending" });
    }

    return c.json(result);
  } catch (error) {
    const sanitized = sanitizeError(error);
    safeLogError("Error fetching submission", sanitized);
    return errorResponse(500, "Internal server error", c);
  }
});

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    (env as any).__executionCtx = ctx;
    return app.fetch(request, env, ctx as any);
  },
};
