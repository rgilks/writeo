/**
 * OpenAPI metadata and configuration
 *
 * Note: Server URLs are dynamically set in routes/health.ts based on the request
 */

const API_DESCRIPTION = `## Writeo Text API

A modern, scalable API for essay assessment and scoring.

### Features
- **Question Management**: Create and store essay questions
- **Submission Processing**: Submit essays for automated scoring (answers are sent inline with submissions)
- **Assessment Results**: Retrieve detailed scoring results with band scores and CEFR levels
- **Synchronous Processing**: Results returned immediately in POST response body
- **AI Feedback**: Get AI-powered feedback and teacher guidance for essay improvement

### Architecture
- **Edge API**: Cloudflare Workers for low-latency global access
- **Storage**: R2 for questions and submissions, KV for results (when opt-in storage is enabled)
- **ML Scoring**: Modal service for transformer-based essay scoring

### Authentication
All endpoints (except /health, /docs, /openapi.json) require API key authentication via the Authorization header:

\`Authorization: Token <your_api_key>\`

For API access, please contact the project maintainer via GitHub (https://github.com/rgilks/writeo) or Discord (https://discord.gg/YxuFAXWuzw).

### Rate Limits (per IP)
- **Submissions**: 10 requests per minute (burst limit) AND 100 requests per day (daily limit)
- **Results (GET)**: 60 requests per minute (read-only)
- **Questions**: 30 requests per minute (data writes)
- **Other endpoints**: 30 requests per minute

For higher limits, please contact the project maintainer via GitHub (https://github.com/rgilks/writeo) or Discord (https://discord.gg/YxuFAXWuzw).

### CEFR Level Mapping
Band scores are automatically mapped to CEFR (Common European Framework of Reference) levels:
- **A2**: < 4.0
- **B1**: ≥ 4.0, < 5.5
- **B2**: ≥ 5.5, < 7.0
- **C1**: ≥ 7.0, < 8.5
- **C2**: ≥ 8.5

### Support
- Documentation: Available at /docs (Swagger UI)
- Health Check: GET /health`;

export const openApiMetadata = {
  openapi: "3.0.0",
  info: {
    title: "Writeo Text API",
    version: "1.0.0",
    description: API_DESCRIPTION,
    contact: {
      name: "Robert Gilks",
      url: "https://tre.systems",
    },
    license: {
      name: "Apache-2.0",
      url: "https://www.apache.org/licenses/LICENSE-2.0",
    },
  },
  // Servers are dynamically set in routes/health.ts based on the request
  components: {
    securitySchemes: {
      ApiKeyAuth: {
        type: "apiKey",
        in: "header",
        name: "Authorization",
        description: "API key authentication. Format: 'Token <your_api_key>'",
      },
    },
    // Schemas are defined in components.ts and merged at runtime
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
} as const;
