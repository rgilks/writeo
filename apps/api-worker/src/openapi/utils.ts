/**
 * Shared OpenAPI utilities and common definitions
 */

// Common parameter definitions
export const submissionIdParam = {
  name: "submission_id",
  in: "path" as const,
  required: true,
  description: "Unique identifier for the submission (UUID format)",
  schema: {
    type: "string" as const,
    format: "uuid" as const,
    example: "770e8400-e29b-41d4-a716-446655440000",
  },
};

export const questionIdParam = {
  name: "question_id",
  in: "path" as const,
  required: true,
  description: "Unique identifier for the question (UUID format)",
  schema: {
    type: "string" as const,
    format: "uuid" as const,
    example: "550e8400-e29b-41d4-a716-446655440000",
  },
};

// Common request body properties
export const answerIdProperty = {
  type: "string" as const,
  format: "uuid" as const,
  description: "UUID of the answer",
  example: "660e8400-e29b-41d4-a716-446655440000",
};

export const answerTextProperty = {
  type: "string" as const,
  description: "The answer text to analyze",
  example: "I went to the park yesterday and played football with my friends.",
};

// Common error response helper with structured format
export const errorResponse = (description: string, example: string, code?: string) => ({
  description,
  content: {
    "application/json": {
      schema: {
        type: "object" as const,
        properties: {
          error: {
            type: "object" as const,
            properties: {
              code: {
                type: "string" as const,
                description: "Machine-readable error code",
                example: code || "INVALID_FIELD_VALUE",
              },
              message: {
                type: "string" as const,
                description: "Human-readable error message",
                example,
              },
              requestId: {
                type: "string" as const,
                description: "Request ID for debugging (optional)",
                example: "req-1234567890",
              },
            },
            required: ["code", "message"],
          },
        },
      },
      example: {
        error: {
          code: code || "INVALID_FIELD_VALUE",
          message: example,
          requestId: "req-1234567890",
        },
      },
    },
  },
});

// Standard error responses with appropriate error codes
export const badRequestResponse = (description: string, example: string, code?: string) =>
  errorResponse(
    description ? `Bad request - ${description}` : "Bad request",
    example,
    code || "INVALID_FIELD_VALUE",
  );

export const unauthorizedResponse = errorResponse("Unauthorized", "Unauthorized", "UNAUTHORIZED");

export const notFoundResponse = (resource: string = "Resource", code?: string) =>
  errorResponse(
    `${resource} not found`,
    `${resource} not found`,
    code ||
      (resource.toLowerCase().includes("submission")
        ? "SUBMISSION_NOT_FOUND"
        : "QUESTION_NOT_FOUND"),
  );

export const conflictResponse = (description: string, example: string, code?: string) =>
  errorResponse(
    `Conflict - ${description}`,
    example,
    code || "SUBMISSION_EXISTS_DIFFERENT_CONTENT",
  );

export const payloadTooLargeResponse = errorResponse(
  "Payload too large (max 1MB)",
  "Request body too large (max 1MB)",
  "PAYLOAD_TOO_LARGE",
);

export const internalServerErrorResponse = errorResponse(
  "Internal server error",
  "Internal server error",
  "INTERNAL_SERVER_ERROR",
);

// No authentication required
export const noAuth = [] as const;
