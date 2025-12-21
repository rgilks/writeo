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
const POST_DESCRIPTION = `Creates a new submission and processes it synchronously for assessment. Results are returned immediately in the response body.

**Processing Mode:**
- **Synchronous**: API processes the submission immediately and returns results in the POST response
- **Typical processing time**: 3-10 seconds (estimate, actual time may vary)
- **Maximum processing time**: <20 seconds (estimate, not a guarantee)

**Submission Format:**

- **Answers**: Must always be sent inline with the submission (using the \`text\` field). Answers cannot be referenced by ID.
- **Questions**: Can be sent inline (with \`questionText\` as a string), referenced by ID (omit \`questionText\`, question must exist), or free writing (set \`questionText\` to \`null\`).

**Request Body:**
The request body must include \`submissionId\` (UUID) along with the submission data. The \`submissionId\` must be unique - if a submission with this ID already exists, the request will fail with 409 Conflict.

**Response:**
Returns \`200 OK\` with assessment results in the response body. Results include band scores, CEFR levels, grammar errors, and AI feedback.`;

export const submissionsPath = {
  "/v1/text/submissions": {
    post: {
      tags: ["Submissions"],
      summary: "Create a new submission for assessment",
      description: POST_DESCRIPTION,
      operationId: "createSubmission",
      security: [{ ApiKeyAuth: [] }],
      requestBody: {
        required: true,
        content: {
          "application/json": {
            schema: {
              type: "object" as const,
              required: ["submissionId", "submission"],
              properties: {
                submissionId: {
                  type: "string" as const,
                  format: "uuid" as const,
                  description:
                    "Unique identifier for the submission (UUID format). Must be unique - if a submission with this ID already exists, the request will fail.",
                  example: "660e8400-e29b-41d4-a716-446655440000",
                },
                submission: {
                  type: "array",
                  description:
                    "Array of submission parts. Answers must always be sent inline with the submission (using the text field). Questions can be sent inline (with questionText), referenced by ID (omit questionText), or free writing (set questionText to null).",
                  items: {
                    type: "object" as const,
                    required: ["part", "answers"],
                    properties: {
                      part: {
                        type: "integer" as const,
                        description:
                          "Part identifier (typically 1 or 2). Must be a positive integer.",
                        example: 1,
                        minimum: 1,
                      },
                      answers: {
                        type: "array",
                        description:
                          "Array of answers. Each answer must include: id, questionId, and text. Optionally include questionText to create/update the question inline, or set to null for free writing.",
                        items: {
                          type: "object" as const,
                          required: ["id", "questionId", "text"],
                          properties: {
                            id: {
                              type: "string" as const,
                              format: "uuid" as const,
                              description: "Answer ID (UUID format)",
                              example: "660e8400-e29b-41d4-a716-446655440000",
                            },
                            questionId: {
                              type: "string" as const,
                              format: "uuid" as const,
                              description:
                                "Question ID (required - will auto-create question if questionText is provided as a string, otherwise question must exist)",
                              example: "550e8400-e29b-41d4-a716-446655440000",
                            },
                            questionText: {
                              oneOf: [
                                { type: "string" as const, maxLength: 10000 },
                                { type: "null" as const },
                              ],
                              description:
                                "Question text: if provided as a string, will create/update question; if null, free writing (no question); if omitted, question must exist. Maximum length: 10,000 characters.",
                              example: "Describe your weekend. What did you do?",
                            },
                            text: {
                              type: "string" as const,
                              maxLength: 50000,
                              description:
                                "Answer text (required - answers must always be sent inline). Maximum length: 50,000 characters.",
                              example:
                                "I went to the park yesterday and played football with my friends.",
                            },
                          },
                        },
                      },
                    },
                  },
                },
                assessors: {
                  type: "array" as const,
                  items: { type: "string" as const },
                  description:
                    "Optional list of specific assessors to run. If omitted, all assessors enabled for the default configuration will run. Example: ['AES-DEBERTA', 'GEC-LT']",
                  example: ["AES-DEBERTA", "GEC-LT"],
                },
                storeResults: {
                  type: "boolean" as const,
                  description: "Whether to store results on the server (default: false)",
                  example: false,
                },
              },
            },
            examples: {
              inlineQuestion: {
                summary: "Submission with inline question",
                value: {
                  submissionId: "660e8400-e29b-41d4-a716-446655440000",
                  submission: [
                    {
                      part: 1,
                      answers: [
                        {
                          id: "660e8400-e29b-41d4-a716-446655440000",
                          questionId: "550e8400-e29b-41d4-a716-446655440000",
                          questionText: "Describe your weekend. What did you do?",
                          text: "I went to the park yesterday and played football with my friends.",
                        },
                      ],
                    },
                  ],

                  storeResults: false,
                },
              },
              referencedQuestion: {
                summary: "Submission with referenced question",
                value: {
                  submissionId: "660e8400-e29b-41d4-a716-446655440000",
                  submission: [
                    {
                      part: 1,
                      answers: [
                        {
                          id: "660e8400-e29b-41d4-a716-446655440000",
                          questionId: "550e8400-e29b-41d4-a716-446655440000",
                          text: "I went to the park yesterday and played football with my friends.",
                        },
                      ],
                    },
                  ],
                },
              },
              freeWriting: {
                summary: "Free writing (no question)",
                value: {
                  submissionId: "660e8400-e29b-41d4-a716-446655440000",
                  submission: [
                    {
                      part: 1,
                      answers: [
                        {
                          id: "660e8400-e29b-41d4-a716-446655440000",
                          questionId: "550e8400-e29b-41d4-a716-446655440000",
                          questionText: null,
                          text: "I went to the park yesterday and played football with my friends.",
                        },
                      ],
                    },
                  ],

                  storeResults: true,
                },
              },
            },
          },
        },
      },
      responses: {
        "201": {
          description:
            "Submission created and processed successfully - results returned immediately in response body",
          headers: {
            Location: {
              description: "URL of the created submission resource",
              schema: {
                type: "string",
                example: "/v1/text/submissions/660e8400-e29b-41d4-a716-446655440000",
              },
            },
            "X-Request-Id": {
              description: "Unique request identifier for debugging",
              schema: { type: "string", example: "req-1234567890" },
            },
            "X-RateLimit-Limit": {
              description: "Maximum number of requests allowed per time window",
              schema: { type: "integer", example: 10 },
            },
            "X-RateLimit-Remaining": {
              description: "Number of requests remaining in the current time window",
              schema: { type: "integer", example: 7 },
            },
            "X-RateLimit-Reset": {
              description: "Unix timestamp (seconds) when the rate limit window resets",
              schema: { type: "integer", example: 1705689600 },
            },
            "X-Timing-Total": {
              description: "Total request processing time in milliseconds",
              schema: { type: "string", example: "3.45" },
            },
          },
          content: {
            "application/json": {
              schema: {
                $ref: "#/components/schemas/AssessmentResults",
              },
            },
          },
        },
        "400": badRequestResponse(
          "invalid format or missing required fields",
          "Answer text is required. Answers must be sent inline with the submission.",
          "INVALID_SUBMISSION_FORMAT",
        ),
        "409": conflictResponse(
          "submission already exists",
          "Submission already exists. Each submission must have a unique submissionId.",
          "SUBMISSION_EXISTS_DIFFERENT_CONTENT",
        ),
        "413": payloadTooLargeResponse,
        "500": internalServerErrorResponse,
      },
    },
  },
  "/v1/text/submissions/{submission_id}": {
    get: {
      tags: ["Submissions"],
      summary: "Get submission results",
      description:
        "Retrieves stored assessment results for a submission. **Note:** This endpoint only works for submissions where `storeResults: true` was set during submission. By default, results are stored only in the user's browser (localStorage) and are not available via this endpoint. If results are not found (404), it may mean the submission was created with `storeResults: false` (default), the results have expired (90-day retention for opt-in storage), or the submission ID is incorrect.",
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
                                      assessorResults: {
                                        type: "array",
                                        items: {
                                          type: "object",
                                          properties: {
                                            id: {
                                              type: "string",
                                              example: "AES-DEBERTA",
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

                        error_message: {
                          type: "string" as const,
                          description: "Error message if status is 'error'",
                          example: "Assessment failed",
                        },
                        meta: {
                          type: "object" as const,
                          description:
                            "Additional metadata including wordCount, errorCount, overallScore, timestamp, answerTexts, and optional draft tracking fields (draftNumber, parentSubmissionId, draftHistory). Draft tracking fields are populated by Server Actions when retrieving results, not by the API itself.",
                          properties: {
                            wordCount: { type: "integer" as const, example: 150 },
                            errorCount: { type: "integer" as const, example: 3 },
                            overallScore: { type: "number" as const, example: 6.5 },
                            timestamp: {
                              type: "string" as const,
                              format: "date-time" as const,
                              example: "2025-01-18T16:00:00Z",
                            },
                            answerTexts: {
                              type: "object" as const,
                              description: "Map of answer IDs to answer texts",
                              additionalProperties: { type: "string" as const },
                            },
                            draftNumber: {
                              type: "integer" as const,
                              description: "Draft number (populated by Server Actions)",
                              example: 1,
                            },
                            parentSubmissionId: {
                              type: "string" as const,
                              format: "uuid" as const,
                              description:
                                "Root submission ID for draft chain (populated by Server Actions)",
                              example: "770e8400-e29b-41d4-a716-446655440000",
                            },
                            draftHistory: {
                              type: "array" as const,
                              description: "Draft history array (populated by Server Actions)",
                              items: {
                                type: "object" as const,
                                properties: {
                                  draftNumber: { type: "integer" as const },
                                  timestamp: {
                                    type: "string" as const,
                                    format: "date-time" as const,
                                  },
                                  wordCount: { type: "integer" as const },
                                  errorCount: { type: "integer" as const },
                                  overallScore: { type: "number" as const },
                                },
                              },
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
                          part: 1,
                          status: "success",
                          answers: [
                            {
                              id: "550e8400-e29b-41d4-a716-446655440000",
                              assessorResults: [
                                {
                                  id: "AES-DEBERTA",
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
          "INVALID_UUID_FORMAT",
        ),
        "404": notFoundResponse("Submission", "SUBMISSION_NOT_FOUND"),
        "500": internalServerErrorResponse,
      },
    },
  },
} as const;
