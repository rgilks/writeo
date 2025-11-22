/**
 * Questions endpoint paths
 */

export const questionsPath = {
  "/text/questions/{question_id}": {
    put: {
      tags: ["Questions"],
      summary: "Create or update a question",
      description:
        "Creates a new question or updates an existing one if the content is identical. Questions are stored in R2 storage.",
      operationId: "createQuestion",
      parameters: [
        {
          name: "question_id",
          in: "path",
          required: true,
          description: "Unique identifier for the question (UUID format)",
          schema: {
            type: "string",
            format: "uuid",
            example: "550e8400-e29b-41d4-a716-446655440000",
          },
        },
      ],
      requestBody: {
        required: true,
        content: {
          "application/json": {
            schema: {
              type: "object",
              required: ["text"],
              properties: {
                text: {
                  type: "string",
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
        "400": {
          description: "Bad request - invalid question_id format or missing text field",
          content: {
            "application/json": {
              schema: {
                type: "object",
                properties: {
                  error: {
                    type: "string",
                    example: "Invalid question_id format",
                  },
                },
              },
            },
          },
        },
        "409": {
          description: "Conflict - question exists with different content",
          content: {
            "application/json": {
              schema: {
                type: "object",
                properties: {
                  error: {
                    type: "string",
                    example: "Question already exists with different content",
                  },
                },
              },
            },
          },
        },
        "413": {
          description: "Payload too large (max 1MB)",
          content: {
            "application/json": {
              schema: {
                type: "object",
                properties: {
                  error: {
                    type: "string",
                    example: "Request body too large (max 1MB)",
                  },
                },
              },
            },
          },
        },
        "500": {
          description: "Internal server error",
          content: {
            "application/json": {
              schema: {
                type: "object",
                properties: {
                  error: {
                    type: "string",
                    example: "Internal server error",
                  },
                },
              },
            },
          },
        },
      },
    },
  },
};
