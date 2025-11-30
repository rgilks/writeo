/**
 * Questions endpoint paths
 */

import {
  questionIdParam,
  badRequestResponse,
  conflictResponse,
  payloadTooLargeResponse,
  internalServerErrorResponse,
} from "../utils";

export const questionsPath = {
  "/v1/text/questions/{question_id}": {
    put: {
      tags: ["Questions"],
      summary: "Create or update a question",
      description:
        "Creates a new question or updates an existing one if the content is identical. Questions are stored in R2 storage.",
      operationId: "createQuestion",
      parameters: [questionIdParam],
      requestBody: {
        required: true,
        content: {
          "application/json": {
            schema: {
              type: "object" as const,
              required: ["text"],
              properties: {
                text: {
                  type: "string" as const,
                  description: "The question text",
                  example: "Describe your weekend. What did you do?",
                },
              },
            },
            example: {
              text: "Describe your weekend. What did you do?",
            },
          },
        },
      },
      responses: {
        "201": {
          description: "Question created successfully",
          headers: {
            Location: {
              description: "URL of the created question resource",
              schema: {
                type: "string",
                example: "/v1/text/questions/550e8400-e29b-41d4-a716-446655440000",
              },
            },
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
        },
        "204": {
          description: "Question already exists with identical content",
        },
        "400": badRequestResponse(
          "invalid question_id format or missing text field",
          "Invalid question_id format",
          "INVALID_UUID_FORMAT",
        ),
        "409": conflictResponse(
          "question exists with different content",
          "Question already exists with different content",
          "QUESTION_EXISTS_DIFFERENT_CONTENT",
        ),
        "413": payloadTooLargeResponse,
        "500": internalServerErrorResponse,
      },
    },
  },
} as const;
