/**
 * Feedback endpoint paths
 */

import {
  submissionIdParam,
  answerIdProperty,
  answerTextProperty,
  badRequestResponse,
  unauthorizedResponse,
  notFoundResponse,
  internalServerErrorResponse,
} from "../utils";

// Descriptions
const AI_FEEDBACK_DESCRIPTION = `Generates AI-powered feedback for an answer using Server-Sent Events (SSE).
The feedback is streamed in real-time as it's generated, providing a smooth user experience.

The feedback includes:
- Relevance assessment
- Strengths identification
- Improvement suggestions
- Overall summary

Uses essay scores and LanguageTool errors as context for more relevant feedback.`;

const TEACHER_FEEDBACK_DESCRIPTION = `Provides interactive learning feedback from a helpful teacher.
Supports two modes:
- **clues**: Provides hints to help the student improve without giving away the answer
- **explanation**: Provides detailed explanations of what could be improved

Uses essay scores and LanguageTool errors as context for targeted feedback.`;

const SSE_STREAM_DESCRIPTION = `Server-Sent Events stream with the following event types:
- \`start\`: Initial event indicating feedback generation has started
- \`chunk\`: Text chunk of the feedback (sent multiple times)
- \`done\`: Final event indicating completion
- \`error\`: Error event if generation fails

Each event is in the format: \`data: {JSON}\\n\\n\``;

const SSE_EXAMPLE = `data: {"type":"start","message":"Starting AI feedback generation..."}

data: {"type":"chunk","text":"Your answer demonstrates "}

data: {"type":"chunk","text":"good understanding of the question. "}

data: {"type":"done","message":"Feedback generation complete"}`;

export const feedbackPaths = {
  "/v1/text/submissions/{submission_id}/ai-feedback/stream": {
    post: {
      tags: ["Feedback"],
      summary: "Stream AI feedback generation",
      description: AI_FEEDBACK_DESCRIPTION,
      operationId: "streamAIFeedback",
      parameters: [submissionIdParam],
      requestBody: {
        required: true,
        content: {
          "application/json": {
            schema: {
              type: "object" as const,
              required: ["answerId", "answerText"],
              properties: {
                answerId: {
                  ...answerIdProperty,
                  description: "UUID of the answer to generate feedback for",
                },
                answerText: answerTextProperty,
              },
            },
          },
        },
      },
      responses: {
        "200": {
          description: "Streaming feedback (Server-Sent Events)",
          headers: {
            "X-Request-Id": {
              description: "Unique request identifier for debugging",
              schema: { type: "string", example: "req-1234567890" },
            },
            "X-RateLimit-Limit": {
              description: "Maximum number of requests allowed per time window",
              schema: { type: "integer", example: 30 },
            },
            "X-RateLimit-Remaining": {
              description: "Number of requests remaining in the current time window",
              schema: { type: "integer", example: 25 },
            },
            "X-RateLimit-Reset": {
              description: "Unix timestamp (seconds) when the rate limit window resets",
              schema: { type: "integer", example: 1705689600 },
            },
          },
          content: {
            "text/event-stream": {
              schema: {
                type: "string" as const,
                description: SSE_STREAM_DESCRIPTION,
              },
              example: SSE_EXAMPLE,
            },
          },
        },
        "400": badRequestResponse(
          "",
          "Missing required fields: answerId, answerText",
          "MISSING_REQUIRED_FIELD",
        ),
        "401": unauthorizedResponse,
        "404": notFoundResponse("Submission or answer", "SUBMISSION_NOT_FOUND"),
        "500": internalServerErrorResponse,
      },
    },
  },
  "/v1/text/submissions/{submission_id}/teacher-feedback": {
    post: {
      tags: ["Feedback"],
      summary: "Get Teacher feedback (clues or explanation)",
      description: TEACHER_FEEDBACK_DESCRIPTION,
      operationId: "getTeacherFeedback",
      parameters: [submissionIdParam],
      requestBody: {
        required: true,
        content: {
          "application/json": {
            schema: {
              type: "object" as const,
              required: ["answerId", "mode", "answerText"],
              properties: {
                answerId: {
                  ...answerIdProperty,
                  description: "UUID of the answer to get feedback for",
                },
                mode: {
                  type: "string" as const,
                  enum: ["clues", "explanation"],
                  description:
                    "Feedback mode - 'clues' provides hints, 'explanation' provides detailed feedback",
                  example: "clues",
                },
                answerText: answerTextProperty,
              },
            },
          },
        },
      },
      responses: {
        "200": {
          description: "Teacher feedback response",
          headers: {
            "X-Request-Id": {
              description: "Unique request identifier for debugging",
              schema: { type: "string", example: "req-1234567890" },
            },
            "X-RateLimit-Limit": {
              description: "Maximum number of requests allowed per time window",
              schema: { type: "integer", example: 30 },
            },
            "X-RateLimit-Remaining": {
              description: "Number of requests remaining in the current time window",
              schema: { type: "integer", example: 25 },
            },
            "X-RateLimit-Reset": {
              description: "Unix timestamp (seconds) when the rate limit window resets",
              schema: { type: "integer", example: 1705689600 },
            },
          },
          content: {
            "application/json": {
              schema: {
                type: "object" as const,
                required: ["message"],
                properties: {
                  message: {
                    type: "string" as const,
                    description: "The feedback message from the Teacher",
                    example:
                      "Consider using more varied sentence structures to improve your coherence score.",
                  },
                  focusArea: {
                    type: "string" as const,
                    description:
                      "Optional focus area for the feedback (e.g., 'Coherence & Cohesion', 'Vocabulary')",
                    example: "Coherence & Cohesion",
                  },
                },
              },
              example: {
                message:
                  "Consider using more varied sentence structures to improve your coherence score. Try connecting your ideas with transition words like 'however' or 'furthermore'.",
                focusArea: "Coherence & Cohesion",
              },
            },
          },
        },
        "400": badRequestResponse(
          "",
          "Mode must be 'clues' or 'explanation'",
          "INVALID_FIELD_VALUE",
        ),
        "401": unauthorizedResponse,
        "404": notFoundResponse("Submission or answer", "SUBMISSION_NOT_FOUND"),
        "500": internalServerErrorResponse,
      },
    },
  },
};
