/**
 * OpenAPI metadata and configuration
 *
 * Note: Server URLs are dynamically set in routes/health.ts based on the request
 */

const API_DESCRIPTION = `## Writeo Text API

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

\`Authorization: Token <your_api_key>\`

For API access, please contact the project maintainer via GitHub (https://github.com/rgilks/writeo) or Discord (https://discord.gg/9rtwCKp2).

### Rate Limits (per IP)
- Submissions: 10 requests per minute (expensive operations)
- Results: 60 requests per minute (read-only)
- Questions/Answers: 30 requests per minute (data writes)
- Other endpoints: 30 requests per minute

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
