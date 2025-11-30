# Writeo API Specification

> **Interactive Docs:** Available at `/docs` endpoint on your API server | [OpenAPI Spec](openapi.yaml)  
> **Architecture:** See [ARCHITECTURE.md](ARCHITECTURE.md) | **Quick Start:** See [README.md](../README.md)

## Overview

RESTful API for essay assessment. All endpoints (except `/health`, `/docs`, `/openapi.json`) require authentication:

```
Authorization: Token <api_key>
```

**Base URL**: `https://your-api-worker.workers.dev` (configure via `API_BASE_URL` environment variable)

## Endpoints

### Health Check

**GET** `/health`

Returns service health status. No authentication required.

**Response:**

```json
{
  "status": "ok"
}
```

### Create Question

**PUT** `/text/questions/{question_id}`

Creates or updates a question. The `question_id` must be a valid UUID.

**Request Body:**

```json
{
  "text": "Describe your weekend. What did you do?"
}
```

**Response Codes:**

- `201 CREATED` - Question created
- `204 NO CONTENT` - Question already exists with identical text
- `400 BAD REQUEST` - Invalid UUID format or missing text
- `409 CONFLICT` - Question exists with different text
- `500 SERVER ERROR` - Internal error

### Create Submission

**PUT** `/text/submissions/{submission_id}`

Creates a submission and triggers assessment. The `submission_id` must be a valid UUID.

**Request Body:**

**Note**: Answers must always be sent inline with the submission. The old reference format (answer ID only) is no longer supported. See examples below.

**Inline Format** (answers must always be sent inline, questions can be inline, referenced, or empty for free writing):

```json
{
  "submission": [
    {
      "part": "1",
      "answers": [
        {
          "id": "answer-uuid",
          "question-number": 1,
          "question-id": "question-uuid",
          "question-text": "Describe your weekend",
          "text": "I went to the park..."
        }
      ]
    }
  ],
  "template": { "name": "generic", "version": 1 },
  "storeResults": false
}
```

**Note:** The `storeResults` parameter is optional and defaults to `false`. When `false` (default), results are returned immediately but not stored on the server. Results are stored only in the user's browser (localStorage). Set `storeResults: true` to enable server storage for 90 days.

**Free Writing** (no question - use empty string for `question-text`):

```json
{
  "submission": [
    {
      "part": "1",
      "answers": [
        {
          "id": "answer-uuid",
          "question-number": 1,
          "question-id": "question-uuid",
          "question-text": "",
          "text": "I went to the park..."
        }
      ]
    }
  ],
  "template": { "name": "generic", "version": 1 },
  "storeResults": false
}
```

**With Referenced Question** (question must exist, omit question-text):

```json
{
  "submission": [
    {
      "part": "1",
      "answers": [
        {
          "id": "answer-uuid",
          "question-number": 1,
          "question-id": "question-uuid",
          "text": "I went to the park..."
        }
      ]
    }
  ],
  "template": { "name": "generic", "version": 1 }
}
```

**Response Codes:**

- `200 OK` - Submission processed successfully - results returned immediately in response body
- `204 NO CONTENT` - Submission already exists with identical content
- `400 BAD REQUEST` - Invalid structure, missing fields, invalid UUIDs, missing answer text (answers must be sent inline), or referenced question does not exist
- `409 CONFLICT` - Submission exists with different content
- `413 PAYLOAD TOO LARGE` - Payload exceeds 1MB limit
- `500 SERVER ERROR` - Internal error

### Get Submission Results

**GET** `/text/submissions/{submission_id}`

Retrieves assessment results for a submission. **Note:** This endpoint only works for submissions where `storeResults: true` was set. By default, results are stored only in the user's browser (localStorage) and are not available via this endpoint.

**Response (Pending):**

```json
{
  "status": "pending"
}
```

**Response (Success):**

```json
{
  "status": "success",
  "results": {
    "parts": [
      {
        "part": "1",
        "status": "success",
        "assessor-results": [
          {
            "id": "T-AES-ESSAY",
            "name": "Essay scorer",
            "type": "grader",
            "overall": 6.5,
            "label": "B2",
            "dimensions": {
              "TA": 6.0,
              "CC": 6.5,
              "Vocab": 6.5,
              "Grammar": 6.0,
              "Overall": 6.5
            }
          },
          {
            "id": "T-GEC-LT",
            "name": "LanguageTool (OSS)",
            "type": "feedback",
            "errors": [
              {
                "start": 2,
                "end": 6,
                "message": "Possible subject–verb agreement error.",
                "suggestions": ["go", "went"],
                "severity": "error",
                "category": "GRAMMAR",
                "rule_id": "SUBJECT_VERB_AGREEMENT",
                "confidenceScore": 0.85,
                "highConfidence": true,
                "errorType": "Subject-verb agreement",
                "explanation": "The verb doesn't match the subject in number (singular/plural).",
                "example": "you goes → you go"
              }
            ]
          },
          {
            "id": "T-AI-FEEDBACK",
            "name": "AI Tutor Feedback",
            "type": "feedback",
            "meta": {
              "relevance": {
                "addressesQuestion": true,
                "score": 0.85,
                "explanation": "The answer addresses most aspects of the question..."
              },
              "feedback": {
                "strengths": ["Good use of vocabulary", "Clear structure"],
                "improvements": ["Focus on grammar accuracy", "Expand on examples"],
                "overall": "Well-written essay at B2 level. Focus on grammar to reach B2+."
              }
            }
          },
          {
            "id": "T-RELEVANCE-CHECK",
            "name": "Answer Relevance Check",
            "type": "feedback",
            "meta": {
              "addressesQuestion": true,
              "similarityScore": 0.72,
              "threshold": 0.5
            }
          }
        ]
      }
    ]
  },
  "template": { "name": "generic", "version": 1 },
  "meta": {
    "answerTexts": {
      "answer-uuid": "I went to the park..."
    },
    "wordCount": 150,
    "errorCount": 5,
    "overallScore": 6.0,
    "timestamp": "2025-01-16T10:30:00Z",
    "draftNumber": 1,
    "parentSubmissionId": "uuid-of-root-submission",
    "draftHistory": [
      {
        "draftNumber": 1,
        "timestamp": "2025-01-16T10:30:00Z",
        "wordCount": 150,
        "errorCount": 5,
        "overallScore": 6.0
      }
    ]
  }
}
```

**Note**: Draft tracking metadata (`draftNumber`, `parentSubmissionId`, `draftHistory`) is stored in the `meta` field. These fields are populated by Server Actions when retrieving results, not by the API itself.

**Important**: `parentSubmissionId` is stored **only** in `results.meta.parentSubmissionId`. It is not stored as a separate field in the results store, and URLs do not include `?parent=` query parameters. This simplifies the system and ensures a single source of truth.

**Draft History UI**: The frontend displays draft history with improved styling and deduplication logic. Each draft number appears only once, with scores displayed below the draft number. The UI uses flexbox layout for better responsiveness and visual hierarchy.

**Response Codes:**

- `200 OK` - Results returned (pending or success)
- `404 NOT FOUND` - Submission not found
- `500 SERVER ERROR` - Internal error

## Data Types

### UUID Format

All resource IDs must be valid UUIDs (version 4 format):

```
xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
```

### CEFR Levels

The system maps band scores to CEFR levels:

- **A2**: < 4.0
- **B1**: 4.0-4.99
- **B2**: 5.5-6.99
- **C1**: 7.0-8.49
- **C2**: 8.5-9.0

### Error Response Format

All error responses follow this format:

```json
{
  "error": "Error message description",
  "code": "ERROR_CODE"
}
```

## Synchronous Processing

The API processes submissions synchronously and returns results immediately in the PUT response body. Results are available as soon as processing completes (typically 3-10 seconds, maximum <20 seconds).

The PUT response format matches the GET response format - see "Get Submission Results" section above for the complete response structure.

## Rate Limits

Rate limits are applied per IP address:

- **Submissions**: 10 requests per minute (burst limit) AND 100 requests per day (daily limit) per IP
- **Results (GET)**: 60 requests per minute (read-only)
- **Questions/Answers**: 30 requests per minute (data writes)
- **Other endpoints**: 30 requests per minute

For higher limits, please contact the project maintainer via [GitHub](https://github.com/rgilks/writeo) or [Discord](https://discord.gg/YxuFAXWuzw).

## References

- [Interactive API Documentation](https://writeo-api-worker.rob-gilks.workers.dev/docs) - Swagger UI (available at `/docs` endpoint)
- [OpenAPI Specification](openapi.yaml) - Machine-readable API spec
- [Architecture Documentation](ARCHITECTURE.md) - System design and architecture
- [Deployment Guide](DEPLOYMENT.md) - Deployment instructions
