/**
 * Feedback endpoint paths
 */

export const feedbackPaths = {
  "/text/submissions/{submission_id}/ai-feedback/stream": {
    post: {
      tags: ["Feedback"],
      summary: "Stream AI feedback generation",
      description: `Generates AI-powered feedback for an answer using Server-Sent Events (SSE).
The feedback is streamed in real-time as it's generated, providing a smooth user experience.

The feedback includes:
- Relevance assessment
- Strengths identification
- Improvement suggestions
- Overall summary

Uses essay scores and LanguageTool errors as context for more relevant feedback.`,
      operationId: "streamAIFeedback",
      parameters: [
        {
          name: "submission_id",
          in: "path",
          required: true,
          description: "Unique identifier for the submission (UUID format)",
          schema: {
            type: "string",
            format: "uuid",
            example: "770e8400-e29b-41d4-a716-446655440000",
          },
        },
      ],
      requestBody: {
        required: true,
        content: {
          "application/json": {
            schema: {
              type: "object",
              required: ["answerId", "answerText"],
              properties: {
                answerId: {
                  type: "string",
                  format: "uuid",
                  description: "UUID of the answer to generate feedback for",
                  example: "660e8400-e29b-41d4-a716-446655440000",
                },
                answerText: {
                  type: "string",
                  description: "The answer text to analyze",
                  example: "I went to the park yesterday and played football with my friends.",
                },
              },
            },
          },
        },
      },
      responses: {
        "200": {
          description: "Streaming feedback (Server-Sent Events)",
          content: {
            "text/event-stream": {
              schema: {
                type: "string",
                description: `Server-Sent Events stream with the following event types:
- \`start\`: Initial event indicating feedback generation has started
- \`chunk\`: Text chunk of the feedback (sent multiple times)
- \`done\`: Final event indicating completion
- \`error\`: Error event if generation fails

Each event is in the format: \`data: {JSON}\\n\\n\``,
              },
              example: `data: {"type":"start","message":"Starting AI feedback generation..."}

data: {"type":"chunk","text":"Your answer demonstrates "}

data: {"type":"chunk","text":"good understanding of the question. "}

data: {"type":"done","message":"Feedback generation complete"}`,
            },
          },
        },
        "400": {
          description: "Bad request",
          content: {
            "application/json": {
              schema: {
                type: "object",
                properties: {
                  error: {
                    type: "string",
                    example: "Missing required fields: answerId, answerText",
                  },
                },
              },
            },
          },
        },
        "401": {
          description: "Unauthorized",
          content: {
            "application/json": {
              schema: {
                type: "object",
                properties: {
                  error: {
                    type: "string",
                    example: "Unauthorized",
                  },
                },
              },
            },
          },
        },
        "404": {
          description: "Submission or answer not found",
          content: {
            "application/json": {
              schema: {
                type: "object",
                properties: {
                  error: {
                    type: "string",
                    example: "Submission not found",
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
  "/text/submissions/{submission_id}/teacher-feedback": {
    post: {
      tags: ["Feedback"],
      summary: "Get Teacher feedback (clues or explanation)",
      description: `Provides interactive learning feedback from a helpful teacher.
Supports two modes:
- **clues**: Provides hints to help the student improve without giving away the answer
- **explanation**: Provides detailed explanations of what could be improved

Uses essay scores and LanguageTool errors as context for targeted feedback.`,
      operationId: "getTeacherFeedback",
      parameters: [
        {
          name: "submission_id",
          in: "path",
          required: true,
          description: "Unique identifier for the submission (UUID format)",
          schema: {
            type: "string",
            format: "uuid",
            example: "770e8400-e29b-41d4-a716-446655440000",
          },
        },
      ],
      requestBody: {
        required: true,
        content: {
          "application/json": {
            schema: {
              type: "object",
              required: ["answerId", "mode", "answerText"],
              properties: {
                answerId: {
                  type: "string",
                  format: "uuid",
                  description: "UUID of the answer to get feedback for",
                  example: "660e8400-e29b-41d4-a716-446655440000",
                },
                mode: {
                  type: "string",
                  enum: ["clues", "explanation"],
                  description:
                    "Feedback mode - 'clues' provides hints, 'explanation' provides detailed feedback",
                  example: "clues",
                },
                answerText: {
                  type: "string",
                  description: "The answer text to analyze",
                  example: "I went to the park yesterday and played football with my friends.",
                },
              },
            },
          },
        },
      },
      responses: {
        "200": {
          description: "Teacher feedback response",
          content: {
            "application/json": {
              schema: {
                type: "object",
                required: ["message"],
                properties: {
                  message: {
                    type: "string",
                    description: "The feedback message from the Teacher",
                    example:
                      "Consider using more varied sentence structures to improve your coherence score.",
                  },
                  focusArea: {
                    type: "string",
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
        "400": {
          description: "Bad request",
          content: {
            "application/json": {
              schema: {
                type: "object",
                properties: {
                  error: {
                    type: "string",
                    example: "Mode must be 'clues' or 'explanation'",
                  },
                },
              },
            },
          },
        },
        "401": {
          description: "Unauthorized",
          content: {
            "application/json": {
              schema: {
                type: "object",
                properties: {
                  error: {
                    type: "string",
                    example: "Unauthorized",
                  },
                },
              },
            },
          },
        },
        "404": {
          description: "Submission or answer not found",
          content: {
            "application/json": {
              schema: {
                type: "object",
                properties: {
                  error: {
                    type: "string",
                    example: "Submission not found",
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
