/**
 * Shared OpenAPI components (schemas) for reuse across endpoints
 * These schemas are referenced using $ref in path definitions
 */

// Common response headers
export const commonResponseHeaders = {
  "X-Request-Id": {
    description: "Unique request identifier for debugging",
    schema: {
      type: "string",
      example: "req-1234567890",
    },
  },
  "X-RateLimit-Limit": {
    description: "Maximum number of requests allowed per time window",
    schema: {
      type: "integer",
      example: 10,
    },
  },
  "X-RateLimit-Remaining": {
    description: "Number of requests remaining in the current time window",
    schema: {
      type: "integer",
      example: 7,
    },
  },
  "X-RateLimit-Reset": {
    description: "Unix timestamp (seconds) when the rate limit window resets",
    schema: {
      type: "integer",
      example: 1705689600,
    },
  },
  "X-Timing-Total": {
    description: "Total request processing time in milliseconds",
    schema: {
      type: "string",
      example: "3.45",
    },
  },
  "X-Timing-Slowest": {
    description: "Top 5 slowest processing steps (format: 'step:time; step:time')",
    schema: {
      type: "string",
      example: "6_process_essay:2.10; 7_process_languagetool:0.85",
    },
  },
};

export const sharedComponents = {
  schemas: {
    AssessorResult: {
      type: "object",
      required: ["id", "name", "type"],
      properties: {
        id: {
          type: "string",
          description: "Assessor identifier",
          example: "T-AES-ESSAY",
        },
        name: {
          type: "string",
          description: "Human-readable assessor name",
          example: "Essay scorer",
        },
        type: {
          type: "string",
          enum: ["grader", "conf", "ard", "feedback"],
          description: "Type of assessor",
          example: "grader",
        },
        overall: {
          type: "number",
          minimum: 0,
          maximum: 9,
          description: "Overall band score (0-9, 0.5 increments)",
          example: 6.5,
        },
        label: {
          type: "string",
          enum: ["A2", "B1", "B2", "C1", "C2"],
          description: "CEFR level label",
          example: "B2",
        },
        dimensions: {
          type: "object",
          description: "Detailed scores by dimension",
          properties: {
            TA: {
              type: "number",
              minimum: 0,
              maximum: 9,
              description: "Task Achievement score",
              example: 6.0,
            },
            CC: {
              type: "number",
              minimum: 0,
              maximum: 9,
              description: "Coherence & Cohesion score",
              example: 6.5,
            },
            Vocab: {
              type: "number",
              minimum: 0,
              maximum: 9,
              description: "Vocabulary score",
              example: 6.5,
            },
            Grammar: {
              type: "number",
              minimum: 0,
              maximum: 9,
              description: "Grammar score",
              example: 6.0,
            },
            Overall: {
              type: "number",
              minimum: 0,
              maximum: 9,
              description: "Overall score",
              example: 6.5,
            },
          },
        },
        errors: {
          type: "array",
          items: {
            type: "object",
            description: "Grammar and language errors (for T-GEC-LT and T-GEC-LLM assessors)",
            additionalProperties: true,
          },
          description: "Array of grammar/language errors",
        },
        meta: {
          type: "object",
          description: "Assessor-specific metadata",
          additionalProperties: true,
        },
      },
    },
    AnswerResult: {
      type: "object",
      required: ["id", "assessorResults"],
      description: "Answer result with assessor results",
      properties: {
        id: {
          type: "string",
          format: "uuid",
          description: "Answer ID (UUID)",
          example: "550e8400-e29b-41d4-a716-446655440000",
        },
        assessorResults: {
          type: "array",
          description: "List of assessor results for this answer",
          items: {
            $ref: "#/components/schemas/AssessorResult",
          },
          minItems: 1,
        },
      },
    },
    AssessmentPart: {
      type: "object",
      required: ["part", "status", "answers"],
      description: "Assessment results for a single part of the submission",
      properties: {
        part: {
          type: "integer",
          description: "Part number",
          example: 1,
          minimum: 1,
        },
        status: {
          type: "string",
          enum: ["success", "error"],
          description: "Processing status for this part",
          example: "success",
        },
        answers: {
          type: "array",
          description: "List of answer results with assessor results",
          items: {
            $ref: "#/components/schemas/AnswerResult",
          },
          minItems: 1,
        },
      },
    },
    AssessmentResults: {
      type: "object",
      required: ["status", "template"],
      properties: {
        status: {
          type: "string",
          enum: ["success", "error", "pending", "bypassed"],
          description: "Overall processing status",
          example: "success",
        },
        results: {
          type: "object",
          description: "Assessment results organized by parts",
          properties: {
            parts: {
              type: "array",
              items: {
                $ref: "#/components/schemas/AssessmentPart",
              },
            },
          },
        },
        requestedAssessors: {
          type: "array",
          items: {
            type: "string",
          },
          description: "List of assessors requested by the client",
          example: ["T-AES-ESSAY", "T-GEC-LT"],
        },
        activeAssessors: {
          type: "array",
          items: {
            type: "string",
          },
          description: "List of assessors actually run",
          example: ["T-AES-ESSAY", "T-GEC-LT"],
        },
        template: {
          type: "object",
          description: "Template metadata (echoed from request)",
          required: ["name", "version"],
          properties: {
            name: {
              type: "string",
              example: "essay-task-2",
            },
            version: {
              type: "integer",
              example: 1,
            },
          },
        },
        error_message: {
          type: "string",
          description: "Error message if status is 'error'",
          example: "RuntimeError: Failed to load model engessay",
        },
        meta: {
          type: "object",
          description:
            "Additional metadata including wordCount, errorCount, overallScore, timestamp, answerTexts, and optional draft tracking fields (draftNumber, parentSubmissionId, draftHistory). Draft tracking fields are populated by Server Actions when retrieving results, not by the API itself.",
          properties: {
            wordCount: {
              type: "integer",
              description: "Total word count of the submission",
              example: 150,
            },
            errorCount: {
              type: "integer",
              description: "Total number of grammar/language errors detected",
              example: 3,
            },
            overallScore: {
              type: "number",
              description: "Overall band score (0-9)",
              example: 6.5,
            },
            timestamp: {
              type: "string",
              format: "date-time",
              description: "Processing timestamp",
              example: "2025-01-18T16:00:00Z",
            },
            answerTexts: {
              type: "object",
              description: "Map of answer IDs to answer texts",
              additionalProperties: {
                type: "string",
              },
            },
            draftNumber: {
              type: "integer",
              description: "Draft number (populated by Server Actions)",
              example: 1,
            },
            parentSubmissionId: {
              type: "string",
              format: "uuid",
              description: "Root submission ID for draft chain (populated by Server Actions)",
              example: "770e8400-e29b-41d4-a716-446655440000",
            },
            draftHistory: {
              type: "array",
              description: "Draft history array (populated by Server Actions)",
              items: {
                type: "object",
                properties: {
                  draftNumber: {
                    type: "integer",
                  },
                  timestamp: {
                    type: "string",
                    format: "date-time",
                  },
                  wordCount: {
                    type: "integer",
                  },
                  errorCount: {
                    type: "integer",
                  },
                  overallScore: {
                    type: "number",
                  },
                },
              },
            },
          },
          additionalProperties: true,
        },
      },
    },
  },
} as const;
