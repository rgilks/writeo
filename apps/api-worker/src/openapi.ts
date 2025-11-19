// OpenAPI spec extracted from main file
// This is a large object, keeping it separate for clarity
export const openApiSpec = {
  openapi: "3.0.0",
  info: {
    title: "Writeo Text API",
    version: "1.0.0",
    description: `## Writeo Text API

A modern, scalable API for essay assessment and scoring.

### Features
- **Question Management**: Create and store essay questions
- **Answer Management**: Create and store student answers
- **Submission Processing**: Submit essays for automated scoring
- **Assessment Results**: Retrieve detailed scoring results with band scores and CEFR levels
- **Synchronous Processing**: Results returned immediately in PUT response body

### Architecture
- **Edge API**: Cloudflare Workers for low-latency global access
- **Storage**: R2 for questions/answers/submissions, KV for results
- **ML Scoring**: Modal service for transformer-based essay scoring

### Authentication
All endpoints (except /health, /docs, /openapi.json) require API key authentication via the Authorization header:

Authorization: Token <your_api_key>

Contact support to obtain an API key.

### Rate Limits (per IP)
- Submissions: 10 requests per minute (expensive operations)
- Results: 60 requests per minute (read-only)
- Questions/Answers: 30 requests per minute (data writes)
- Other endpoints: 30 requests per minute

### Support
- Documentation: Available at /docs (Swagger UI)
- Health Check: GET /health`,
    contact: {
      name: "Writeo API Support",
      url: "https://writeo.tre.systems",
    },
    license: {
      name: "Apache-2.0",
      url: "https://www.apache.org/licenses/LICENSE-2.0",
    },
  },
  servers: [
    {
      url: "https://your-api-worker.workers.dev",
      description: "Production server (configured dynamically)",
    },
    {
      url: "http://localhost:8787",
      description: "Local development server",
    },
  ],
  components: {
    securitySchemes: {
      ApiKeyAuth: {
        type: "apiKey",
        in: "header",
        name: "Authorization",
        description: "API key authentication. Format: 'Token <your_api_key>'",
      },
    },
  },
  security: [{ ApiKeyAuth: [] }],
  tags: [
    { name: "Questions", description: "Question management endpoints" },
    { name: "Submissions", description: "Submission processing and results" },
    {
      name: "Feedback",
      description: "AI-powered feedback and learning assistance endpoints",
    },
    {
      name: "Health",
      description: "Health check endpoints (no authentication required)",
    },
  ],
  paths: {
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
    "/health": {
      get: {
        tags: ["Health"],
        summary: "Health check",
        description: "Returns the health status of the API. No authentication required.",
        operationId: "healthCheck",
        security: [],
        responses: {
          "200": {
            description: "Service is healthy",
            content: {
              "application/json": {
                schema: {
                  type: "object",
                  properties: {
                    status: {
                      type: "string",
                      example: "ok",
                    },
                  },
                },
              },
            },
          },
        },
      },
    },
    "/docs": {
      get: {
        tags: ["Health"],
        summary: "API Documentation",
        description: "Swagger UI documentation for the API. No authentication required.",
        operationId: "getDocs",
        security: [],
      },
    },
  },
};
