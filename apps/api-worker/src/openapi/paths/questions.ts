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
  "/text/questions/{question_id}": {
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
        },
        "204": {
          description: "Question already exists with identical content",
        },
        "400": badRequestResponse(
          "invalid question_id format or missing text field",
          "Invalid question_id format",
        ),
        "409": conflictResponse(
          "question exists with different content",
          "Question already exists with different content",
        ),
        "413": payloadTooLargeResponse,
        "500": internalServerErrorResponse,
      },
    },
  },
} as const;
