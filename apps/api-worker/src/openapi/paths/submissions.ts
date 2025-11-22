/**
 * Submissions endpoint paths
 */

export const submissionsPath = {
  "/text/submissions/{submission_id}": {
    put: {
      tags: ["Submissions"],
      summary: "Create a submission for assessment",
      description: `Creates a submission and processes it synchronously for assessment. Results are returned immediately in the response body.

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
    "part": 1,
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
    "part": 1,
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
Returns \`200 OK\` with assessment results in the response body. Results include band scores, CEFR levels, grammar errors, and AI feedback.`,
      operationId: "createSubmission",
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
              required: ["submission", "template"],
              properties: {
                submission: {
                  type: "array",
                  description:
                    "Array of submission parts. Answers must always be sent inline with the submission (using the text field). Questions can be sent inline (with question-text) or referenced by ID (question must already exist).",
                  items: {
                    type: "object",
                    required: ["part", "answers"],
                    properties: {
                      part: {
                        type: "integer",
                        description: "Part number (typically 1 or 2)",
                        example: 1,
                        minimum: 1,
                      },
                      answers: {
                        type: "array",
                        description:
                          "Array of answers. Each answer must include: id, question-number, question-id, and text. Optionally include question-text to create/update the question inline.",
                        items: {
                          type: "object",
                          required: ["id", "question-number", "question-id", "text"],
                          properties: {
                            id: {
                              type: "string",
                              format: "uuid",
                              description: "Answer ID (UUID format)",
                              example: "660e8400-e29b-41d4-a716-446655440000",
                            },
                            "question-number": {
                              type: "integer",
                              description: "Question number within the part",
                              example: 1,
                            },
                            "question-id": {
                              type: "string",
                              format: "uuid",
                              description:
                                "Question ID (required - will auto-create question if question-text is provided, otherwise question must exist)",
                              example: "550e8400-e29b-41d4-a716-446655440000",
                            },
                            "question-text": {
                              type: "string",
                              description:
                                "Question text (optional - if provided, will create/update question; if omitted, question must exist)",
                              example: "Describe your weekend. What did you do?",
                            },
                            text: {
                              type: "string",
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
                  type: "object",
                  required: ["name", "version"],
                  properties: {
                    name: {
                      type: "string",
                      description: "Template name",
                      example: "essay-task-2",
                    },
                    version: {
                      type: "integer",
                      description: "Template version",
                      example: 1,
                    },
                  },
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
                type: "object",
                required: ["status", "template"],
                properties: {
                  status: {
                    type: "string",
                    enum: ["success", "error", "pending", "bypassed"],
                    example: "success",
                  },
                  results: {
                    type: "object",
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
                      type: "object",
                      required: ["name", "version"],
                      properties: {
                        name: { type: "string", example: "essay-task-2" },
                        version: { type: "integer", example: 1 },
                      },
                    },
                    error_message: {
                      type: "string",
                      description: "Error message if status is 'error'",
                      example: "Assessment failed",
                    },
                    meta: {
                      type: "object",
                      description:
                        "Additional metadata (wordCount, errorCount, overallScore, timestamp, etc.)",
                      properties: {
                        wordCount: { type: "integer", example: 150 },
                        errorCount: { type: "integer", example: 3 },
                        overallScore: { type: "number", example: 6.5 },
                        timestamp: {
                          type: "string",
                          format: "date-time",
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
                        part: 1,
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
        "400": {
          description: "Bad request - invalid format or missing required fields",
          content: {
            "application/json": {
              schema: {
                type: "object",
                properties: {
                  error: {
                    type: "string",
                    example:
                      "Answer text is required. Answers must be sent inline with the submission.",
                  },
                },
              },
            },
          },
        },
        "409": {
          description: "Conflict - submission exists with different content",
          content: {
            "application/json": {
              schema: {
                type: "object",
                properties: {
                  error: {
                    type: "string",
                    example: "Submission already exists with different content",
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
          description: "Internal server error or assessment failed",
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
    get: {
      tags: ["Submissions"],
      summary: "Get submission results",
      description:
        "Retrieves stored assessment results for a submission. Note: PUT endpoint returns results immediately, so GET is primarily useful for retrieving previously stored results or checking status.",
      operationId: "getSubmissionResults",
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
      responses: {
        "200": {
          description: "Submission results (success, error, or pending)",
          content: {
            "application/json": {
              schema: {
                oneOf: [
                  {
                    type: "object",
                    properties: {
                      status: {
                        type: "string",
                        enum: ["pending"],
                        example: "pending",
                      },
                    },
                  },
                  {
                    type: "object",
                    properties: {
                      status: {
                        type: "string",
                        enum: ["success", "error", "bypassed"],
                        example: "success",
                      },
                      results: {
                        type: "object",
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
                          type: "object",
                          required: ["name", "version"],
                          properties: {
                            name: { type: "string", example: "generic" },
                            version: { type: "integer", example: 1 },
                          },
                          example: { name: "generic", version: 1 },
                        },
                        error_message: {
                          type: "string",
                          description: "Error message if status is 'error'",
                          example: "Assessment failed",
                        },
                        meta: {
                          type: "object",
                          description:
                            "Additional metadata (wordCount, errorCount, overallScore, timestamp, etc.)",
                          properties: {
                            wordCount: { type: "integer", example: 150 },
                            errorCount: { type: "integer", example: 3 },
                            overallScore: { type: "number", example: 6.5 },
                            timestamp: {
                              type: "string",
                              format: "date-time",
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
                          part: 1,
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
        "400": {
          description: "Bad request - invalid submission_id format",
          content: {
            "application/json": {
              schema: {
                type: "object",
                properties: {
                  error: {
                    type: "string",
                    example: "Invalid submission_id format",
                  },
                },
              },
            },
          },
        },
        "404": {
          description: "Submission not found",
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
