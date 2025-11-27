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

// Common error response helper
export const errorResponse = (description: string, example: string) => ({
  description,
  content: {
    "application/json": {
      schema: {
        type: "object" as const,
        properties: {
          error: {
            type: "string" as const,
            example,
          },
        },
      },
    },
  },
});

// Standard error responses
export const badRequestResponse = (description: string, example: string) =>
  errorResponse(description ? `Bad request - ${description}` : "Bad request", example);

export const unauthorizedResponse = errorResponse("Unauthorized", "Unauthorized");

export const notFoundResponse = (resource: string = "Resource") =>
  errorResponse(`${resource} not found`, `${resource} not found`);

export const conflictResponse = (description: string, example: string) =>
  errorResponse(`Conflict - ${description}`, example);

export const payloadTooLargeResponse = errorResponse(
  "Payload too large (max 1MB)",
  "Request body too large (max 1MB)",
);

export const internalServerErrorResponse = errorResponse(
  "Internal server error",
  "Internal server error",
);

// No authentication required
export const noAuth = [] as const;
