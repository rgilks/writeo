import { Hono } from "hono";
import { cors } from "hono/cors";
import type { Env } from "./types/env";
import { authenticate } from "./middleware/auth";
import { rateLimit } from "./middleware/rate-limit";
import { securityHeaders, getCorsOrigin } from "./middleware/security";
import { questionsRouter } from "./routes/questions";
import { healthRouter } from "./routes/health";
import { feedbackRouter } from "./routes/feedback";
import { processSubmissionHandler, getSubmissionHandler } from "./routes/submissions";

const app = new Hono<{ Bindings: Env }>();

// CORS
app.use(
  "*",
  cors({
    origin: (origin, c) => getCorsOrigin(origin, c.env.ALLOWED_ORIGINS),
    allowMethods: ["GET", "PUT", "POST", "OPTIONS"],
    allowHeaders: ["Content-Type", "Authorization"],
    maxAge: 86400,
  }),
);

// Security headers
app.use("*", securityHeaders);

// Public routes (before auth)
app.route("/", healthRouter);

// Authentication middleware
app.use("*", authenticate);

// Rate limiting (runs after auth to use user identity)
app.use("*", rateLimit);

// Protected routes
app.route("/", questionsRouter);
app.route("/", feedbackRouter);

// Submission routes (must come after feedbackRouter to avoid route conflicts)
app.put("/text/submissions/:submission_id", processSubmissionHandler);
app.get("/text/submissions/:submission_id", getSubmissionHandler);

export default {
  async fetch(
    request: Request,
    env: Env,
    ctx: import("@cloudflare/workers-types").ExecutionContext,
  ): Promise<Response> {
    return app.fetch(request, env, ctx);
  },
};
