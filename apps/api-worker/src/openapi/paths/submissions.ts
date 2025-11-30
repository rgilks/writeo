/**
 * Submissions endpoint paths
 */

import {
  submissionIdParam,
  badRequestResponse,
  conflictResponse,
  payloadTooLargeResponse,
  notFoundResponse,
  internalServerErrorResponse,
} from "../utils";

// Descriptions
const PUT_DESCRIPTION = `Creates a submission and processes it synchronously for assessment. Results are returned immediately in the response body.

**Processing Mode:**
- **Synchronous**: API processes the submission immediately and returns results in the PUT response
- **Typical processing time**: 3-10 seconds
- **Maximum processing time**: <20 seconds

**Submission Format:**

- **Answers**: Must always be sent inline with the submission (using the \`text\` field). Answers cannot be referenced by ID.
- **Questions**: Can be sent inline (with \`question-text\`) OR referenced by ID (using \`question-id\` only, question must already exist via PUT /text/questions/{question_id}).

**Example with inline question:**
\`\`\`json
{
    "submission": [{
    "part": "1",
    "answers": [{
      "id": "answer-uuid",
      "question-number": 1,
      "question-id": "question-uuid",
      "question-text": "Describe your weekend.",
      "text": "I went to the park..."
    }]
  }],
  "template": {"name": "essay-task-2", "version": 1}
}
\`\`\`

**Example with referenced question:**
\`\`\`json
{
    "submission": [{
    "part": "1",
    "answers": [{
      "id": "answer-uuid",
      "question-number": 1,
      "question-id": "question-uuid",
      "text": "I went to the park..."
    }]
  }],
  "template": {"name": "essay-task-2", "version": 1}
}
\`\`\`

**Response:**
Returns \`200 OK\` with assessment results in the response body. Results include band scores, CEFR levels, grammar errors, and AI feedback.`;

export const submissionsPath = {
  "/text/submissions/{submission_id}": {
    put: {
      tags: ["Submissions"],
      summary: "Create a submission for assessment",
      description: PUT_DESCRIPTION,
      operationId: "createSubmission",
      parameters: [submissionIdParam],
      requestBody: {
        required: true,
        content: {
          "application/json": {
            schema: {
              type: "object" as const,
              required: ["submission", "template"],
              properties: {
                submission: {
                  type: "array",
                  description:
                    "Array of submission parts. Answers must always be sent inline with the submission (using the text field). Questions can be sent inline (with question-text) or referenced by ID (question must already exist).",
                  items: {
                    type: "object" as const,
                    required: ["part", "answers"],
                    properties: {
                      part: {
                        type: "string" as const,
                        description: 'Part identifier (typically "1" or "2")',
                        example: "1",
                      },
                      answers: {
                        type: "array",
                        description:
                          "Array of answers. Each answer must include: id, question-number, question-id, and text. Optionally include question-text to create/update the question inline.",
                        items: {
                          type: "object" as const,
                          required: ["id", "question-number", "question-id", "text"],
                          properties: {
                            id: {
                              type: "string" as const,
                              format: "uuid" as const,
                              description: "Answer ID (UUID format)",
                              example: "660e8400-e29b-41d4-a716-446655440000",
                            },
                            "question-number": {
                              type: "integer" as const,
                              description: "Question number within the part",
                              example: 1,
                            },
                            "question-id": {
                              type: "string" as const,
                              format: "uuid" as const,
                              description:
                                "Question ID (required - will auto-create question if question-text is provided, otherwise question must exist)",
                              example: "550e8400-e29b-41d4-a716-446655440000",
                            },
                            "question-text": {
                              type: "string" as const,
                              description:
                                "Question text (optional - if provided, will create/update question; if omitted, question must exist)",
                              example: "Describe your weekend. What did you do?",
                            },
                            text: {
                              type: "string" as const,
                              description:
                                "Answer text (required - answers must always be sent inline)",
                              example:
                                "I went to the park yesterday and played football with my friends.",
                            },
                          },
                        },
                      },
                    },
                  },
                },
                template: {
                  type: "object" as const,
                  required: ["name", "version"],
                  properties: {
                    name: {
                      type: "string" as const,
                      description: "Template name",
                      example: "essay-task-2",
                    },
                    version: {
                      type: "integer" as const,
                      description: "Template version",
                      example: 1,
                    },
                  },
                },
                storeResults: {
                  type: "boolean" as const,
                  description:
                    "Opt-in server storage flag. Default: false (no server storage). When false, results are returned immediately but not stored on the server. Results are stored only in the user's browser (localStorage). When true, results are stored on the server for 90 days, allowing access from any device.",
                  default: false,
                  example: false,
                },
              },
            },
          },
        },
      },
      responses: {
        "200": {
          description:
            "Submission processed successfully - results returned immediately in response body",
          content: {
            "application/json": {
              schema: {
                type: "object" as const,
                required: ["status", "template"],
                properties: {
                  status: {
                    type: "string" as const,
                    enum: ["success", "error", "pending", "bypassed"],
                    example: "success",
                  },
                  results: {
                    type: "object" as const,
                    properties: {
                      parts: {
                        type: "array",
                        items: {
                          type: "object",
                          required: ["part", "status", "answers"],
                          properties: {
                            part: { type: "integer", example: 1 },
                            status: {
                              type: "string",
                              enum: ["success", "error"],
                              example: "success",
                            },
                            answers: {
                              type: "array",
                              items: {
                                type: "object",
                                required: ["id", "assessor-results"],
                                properties: {
                                  id: {
                                    type: "string",
                                    format: "uuid",
                                    example: "550e8400-e29b-41d4-a716-446655440000",
                                  },
                                  "assessor-results": {
                                    type: "array",
                                    items: {
                                      type: "object",
                                      required: ["id", "name", "type"],
                                      properties: {
                                        id: {
                                          type: "string",
                                          example: "T-AES-ESSAY",
                                        },
                                        name: {
                                          type: "string",
                                          example: "Essay scorer",
                                        },
                                        type: {
                                          type: "string",
                                          enum: ["grader", "conf", "ard", "feedback"],
                                          example: "grader",
                                        },
                                        overall: {
                                          type: "number",
                                          minimum: 0,
                                          maximum: 9,
                                          example: 6.5,
                                        },
                                        label: {
                                          type: "string",
                                          enum: ["A2", "B1", "B2", "C1", "C2"],
                                          example: "B2",
                                        },
                                        dimensions: {
                                          type: "object",
                                          properties: {
                                            TA: {
                                              type: "number",
                                              minimum: 0,
                                              maximum: 9,
                                              example: 6.0,
                                            },
                                            CC: {
                                              type: "number",
                                              minimum: 0,
                                              maximum: 9,
                                              example: 6.5,
                                            },
                                            Vocab: {
                                              type: "number",
                                              minimum: 0,
                                              maximum: 9,
                                              example: 6.5,
                                            },
                                            Grammar: {
                                              type: "number",
                                              minimum: 0,
                                              maximum: 9,
                                              example: 6.0,
                                            },
                                            Overall: {
                                              type: "number",
                                              minimum: 0,
                                              maximum: 9,
                                              example: 6.5,
                                            },
                                          },
                                        },
                                        errors: {
                                          type: "array",
                                          items: { type: "object" },
                                          description:
                                            "Grammar and language errors (for T-GEC-LT and T-GEC-LLM assessors)",
                                        },
                                        meta: {
                                          type: "object",
                                          description: "Assessor-specific metadata",
                                          additionalProperties: true,
                                        },
                                      },
                                    },
                                  },
                                },
                              },
                            },
                          },
                        },
                      },
                    },
                    template: {
                      type: "object" as const,
                      required: ["name", "version"],
                      properties: {
                        name: { type: "string" as const, example: "essay-task-2" },
                        version: { type: "integer" as const, example: 1 },
                      },
                    },
                    error_message: {
                      type: "string" as const,
                      description: "Error message if status is 'error'",
                      example: "Assessment failed",
                    },
                    meta: {
                      type: "object" as const,
                      description:
                        "Additional metadata (wordCount, errorCount, overallScore, timestamp, etc.)",
                      properties: {
                        wordCount: { type: "integer" as const, example: 150 },
                        errorCount: { type: "integer" as const, example: 3 },
                        overallScore: { type: "number" as const, example: 6.5 },
                        timestamp: {
                          type: "string" as const,
                          format: "date-time" as const,
                          example: "2025-01-18T16:00:00Z",
                        },
                      },
                      additionalProperties: true,
                    },
                  },
                },
                example: {
                  status: "success",
                  results: {
                    parts: [
                    {
                      part: "1",
                        status: "success",
                        answers: [
                          {
                            id: "550e8400-e29b-41d4-a716-446655440000",
                            "assessor-results": [
                              {
                                id: "T-AES-ESSAY",
                                name: "Essay scorer",
                                type: "grader",
                                overall: 6.5,
                                label: "B2",
                                dimensions: {
                                  TA: 6.0,
                                  CC: 6.5,
                                  Vocab: 6.5,
                                  Grammar: 6.0,
                                  Overall: 6.5,
                                },
                              },
                            ],
                          },
                        ],
                      },
                    ],
                  },
                  template: {
                    name: "essay-task-2",
                    version: 1,
                  },
                  meta: {
                    wordCount: 150,
                    errorCount: 3,
                    overallScore: 6.5,
                    timestamp: "2025-01-18T16:00:00Z",
                  },
                },
              },
            },
          },
        },
        "204": {
          description: "Submission already exists with identical content",
        },
        "400": badRequestResponse(
          "invalid format or missing required fields",
          "Answer text is required. Answers must be sent inline with the submission.",
        ),
        "409": conflictResponse(
          "submission exists with different content",
          "Submission already exists with different content",
        ),
        "413": payloadTooLargeResponse,
        "500": internalServerErrorResponse,
      },
    },
    get: {
      tags: ["Submissions"],
      summary: "Get submission results",
      description:
        "Retrieves stored assessment results for a submission. Note: PUT endpoint returns results immediately, so GET is primarily useful for retrieving previously stored results or checking status.",
      operationId: "getSubmissionResults",
      parameters: [submissionIdParam],
      responses: {
        "200": {
          description: "Submission results (success, error, or pending)",
          content: {
            "application/json": {
              schema: {
                oneOf: [
                  {
                    type: "object" as const,
                    properties: {
                      status: {
                        type: "string" as const,
                        enum: ["pending"],
                        example: "pending",
                      },
                    },
                  },
                  {
                    type: "object" as const,
                    properties: {
                      status: {
                        type: "string" as const,
                        enum: ["success", "error", "bypassed"],
                        example: "success",
                      },
                      results: {
                        type: "object" as const,
                        properties: {
                          parts: {
                            type: "array",
                            items: {
                              type: "object",
                              properties: {
                                part: { type: "integer", example: 1 },
                                status: {
                                  type: "string",
                                  enum: ["success", "error"],
                                  example: "success",
                                },
                                answers: {
                                  type: "array",
                                  items: {
                                    type: "object",
                                    properties: {
                                      id: {
                                        type: "string",
                                        format: "uuid",
                                        example: "550e8400-e29b-41d4-a716-446655440000",
                                      },
                                      "assessor-results": {
                                        type: "array",
                                        items: {
                                          type: "object",
                                          properties: {
                                            id: {
                                              type: "string",
                                              example: "T-AES-ESSAY",
                                            },
                                            name: {
                                              type: "string",
                                              example: "Essay scorer",
                                            },
                                            type: {
                                              type: "string",
                                              enum: ["grader", "conf", "ard", "feedback"],
                                              example: "grader",
                                            },
                                            overall: {
                                              type: "number",
                                              minimum: 0,
                                              maximum: 9,
                                              example: 6.5,
                                            },
                                            label: {
                                              type: "string",
                                              enum: ["A2", "B1", "B2", "C1", "C2"],
                                              example: "B2",
                                            },
                                            dimensions: {
                                              type: "object",
                                              properties: {
                                                TA: {
                                                  type: "number",
                                                  example: 6.0,
                                                },
                                                CC: {
                                                  type: "number",
                                                  example: 6.5,
                                                },
                                                Vocab: {
                                                  type: "number",
                                                  example: 6.5,
                                                },
                                                Grammar: {
                                                  type: "number",
                                                  example: 6.0,
                                                },
                                                Overall: {
                                                  type: "number",
                                                  example: 6.5,
                                                },
                                              },
                                              example: {
                                                TA: 6.0,
                                                CC: 6.5,
                                                Vocab: 6.5,
                                                Grammar: 6.0,
                                                Overall: 6.5,
                                              },
                                            },
                                          },
                                        },
                                      },
                                    },
                                  },
                                },
                              },
                            },
                          },
                        },
                        template: {
                          type: "object" as const,
                          required: ["name", "version"],
                          properties: {
                            name: { type: "string" as const, example: "generic" },
                            version: { type: "integer" as const, example: 1 },
                          },
                          example: { name: "generic", version: 1 },
                        },
                        error_message: {
                          type: "string" as const,
                          description: "Error message if status is 'error'",
                          example: "Assessment failed",
                        },
                        meta: {
                          type: "object" as const,
                          description:
                            "Additional metadata (wordCount, errorCount, overallScore, timestamp, etc.)",
                          properties: {
                            wordCount: { type: "integer" as const, example: 150 },
                            errorCount: { type: "integer" as const, example: 3 },
                            overallScore: { type: "number" as const, example: 6.5 },
                            timestamp: {
                              type: "string" as const,
                              format: "date-time" as const,
                              example: "2025-01-18T16:00:00Z",
                            },
                          },
                          additionalProperties: true,
                        },
                      },
                    },
                  },
                ],
              },
              examples: {
                pending: {
                  value: {
                    status: "pending",
                  },
                },
                success: {
                  value: {
                    status: "success",
                    results: {
                      parts: [
                        {
                          part: "1",
                          status: "success",
                          answers: [
                            {
                              id: "550e8400-e29b-41d4-a716-446655440000",
                              "assessor-results": [
                                {
                                  id: "T-AES-ESSAY",
                                  name: "Essay scorer",
                                  type: "grader",
                                  overall: 6.5,
                                  label: "B2",
                                  dimensions: {
                                    TA: 6.0,
                                    CC: 6.5,
                                    Vocab: 6.5,
                                    Grammar: 6.0,
                                    Overall: 6.5,
                                  },
                                },
                              ],
                            },
                          ],
                        },
                      ],
                    },
                    template: { name: "generic", version: 1 },
                    meta: {
                      wordCount: 150,
                      errorCount: 3,
                      overallScore: 6.5,
                      timestamp: "2025-01-18T16:00:00Z",
                    },
                  },
                },
              },
            },
          },
        },
        "400": badRequestResponse(
          "invalid format or missing required fields",
          "Invalid submission_id format",
        ),
        "404": notFoundResponse("Submission"),
        "500": internalServerErrorResponse,
      },
    },
  },
} as const;
