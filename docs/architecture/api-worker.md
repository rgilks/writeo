# API Worker Architecture

This document describes the architecture of the Writeo API Worker, a Cloudflare Workers-based API for essay assessment and feedback.

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Request Flow](#request-flow)
3. [Middleware Chain](#middleware-chain)
4. [Submission Processing Flow](#submission-processing-flow)
5. [Service Architecture](#service-architecture)
6. [Storage Architecture](#storage-architecture)
7. [External Services](#external-services)
8. [Error Handling](#error-handling)
9. [Security](#security)

---

## High-Level Overview

The API Worker is a serverless application running on Cloudflare Workers that provides:

- **Essay Assessment**: Automated scoring and grammar checking
- **AI Feedback**: Streaming and teacher-style feedback
- **Question Management**: CRUD operations for questions and answers
- **Result Storage**: Persistent storage of assessment results

```
┌─────────────────────────────────────────────────────────────┐
│                    Cloudflare Workers                       │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              API Worker (Hono Framework)             │   │
│  │                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │   │
│  │  │  Middleware  │  │   Routes     │  │  Services  │  │   │
│  │  │   Chain      │→ │   Handlers   │→ │  Layer     │  │   │
│  │  └──────────────┘  └──────────────┘  └────────────┘  │   │
│  │                                                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  R2 Bucket   │  │  KV Store    │  │  AI Binding  │       │
│  │  (Storage)   │  │  (Results)   │  │  (Cloudflare)│       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
6└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         │                    │                    │
         ▼                    ▼                    ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │  Modal       │  │  OpenAI/     │  │  LanguageTool│
   │  Services    │  │  Groq API    │  │  (via Modal) │
   │  (Essay/LT)  │  │              │  │              │
   └──────────────┘  └──────────────┘  └──────────────┘
```

---

## Request Flow

### Complete Request Journey

```
Client Request
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. CORS Middleware                                          │
│    - Validates origin (if ALLOWED_ORIGINS configured)       │
│    - Sets CORS headers                                      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Security Headers Middleware                              │
│    - Adds X-Content-Type-Options, X-Frame-Options, etc.     │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Request ID Middleware                                    │
│    - Generates unique request ID (first 8 chars of UUID)    │
│    - Stores in context for logging                          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Route Matching                                           │
│    - /health → Public (no auth)                             │
│    - /docs, /openapi.json → Public                          │
│    - All others → Require authentication                    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Authentication Middleware                                │
│    - Extracts API key from Authorization header             │
│    - Validates against:                                     │
│      1. Admin key (API_KEY env var)                         │
│      2. Test key (TEST_API_KEY env var)                     │
│      3. User keys (KV store lookup)                         │
│    - Sets apiKeyOwner and isTestKey in context              │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Rate Limiting Middleware                                 │
│    - Checks rate limits based on endpoint type:             │
│      • Submissions: 10/min (prod), 500/min (test)           │
│      • Results: 60/min (prod), 2000/min (test)              │
│      • Questions: 30/min (prod), 1000/min (test)            │
│    - Checks daily submission limit (100/day)                │
│    - Uses IP for shared keys, owner ID for user keys        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. Route Handler                                            │
│    - Questions: PUT /v1/text/questions/:question_id         │
│    - Submissions: POST /v1/text/submissions (create)        │
│    - Results: GET /v1/text/submissions/:submission_id       │
│    - Feedback: POST /v1/text/submissions/:id/ai-feedback/stream│
│    - Feedback: POST /v1/text/submissions/:id/teacher-feedback  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. Error Handling (if error occurs)                         │
│    - Catches exceptions                                     │
│    - Logs with request ID                                   │
│    - Returns sanitized error response (production)          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Response to Client
```

---

## Middleware Chain

The middleware executes in this order:

```
Request
  │
  ├─→ CORS
  │     │
  │     ├─→ Security Headers
  │     │     │
  │     │     ├─→ Request ID (generates req-abc123)
  │     │     │     │
  │     │     │     ├─→ Route Check
  │     │     │     │     │
  │     │     │     │     ├─→ Public? → Handler
  │     │     │     │     │
  │     │     │     │     └─→ Protected? → Auth
  │     │     │     │           │
  │     │     │     │           ├─→ Invalid? → 401
  │     │     │     │           │
  │     │     │     │           └─→ Valid? → Rate Limit
  │     │     │     │                 │
  │     │     │     │                 ├─→ Exceeded? → 429
  │     │     │     │                 │
  │     │     │     │                 └─→ OK? → Handler
  │     │     │     │
  │     │     │     └─→ Response (with security headers)
  │     │     │
  │     │     └─→ Response (with security headers)
  │     │
  │     └─→ Response
  │
  └─→ Response
```

### Middleware Details

| Middleware       | Order | Purpose                                 | Can Skip?                      |
| ---------------- | ----- | --------------------------------------- | ------------------------------ |
| CORS             | 1     | Handles cross-origin requests           | No                             |
| Security Headers | 2     | Adds security headers to responses      | No                             |
| Request ID       | 3     | Generates unique ID for request tracing | No                             |
| Route Matching   | 4     | Determines if route is public/protected | No                             |
| Authentication   | 5     | Validates API key                       | Yes (public routes)            |
| Rate Limiting    | 6     | Enforces rate limits                    | Yes (public routes, test keys) |
| Route Handler    | 7     | Business logic                          | No                             |

---

## Submission Processing Flow

The submission processing is the most complex operation. Here's the detailed flow:

```
POST /v1/text/submissions (create) or PUT /v1/text/submissions/:submission_id (update)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Validate & Parse                                   │
│  • Validate request body size (max 1MB)                     │
│  • Parse JSON body                                          │
│  • Validate submission structure (Zod schema)               │
│  • Extract questions/answers to create                      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: Load Data                                          │
│  • Auto-create questions/answers if needed                  │
│  • Load existing data from R2 (if storeResults=true)        │
│  • Build Modal request format                               │
│  • Prepare service requests using Assessor Registry         │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Phase 3: Execute Services (PARALLEL)                         │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Legacy Services                                        │  │
│  │  • LanguageTool (Modal)                                │  │
│  │  • LLM Assessment (OpenAI/Groq)                        │  │
│  │  • Relevance Check (Modal)                             │  │
│  └────────────────────────────────────────────────────────┘  │
│                           │                                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Generic Registry Services (via service-registry.ts)    │  │
│  │  • Corpus Scorer (Modal)                               │  │
│  │  • Feedback Scorer (Modal)                             │  │
│  │  • GEC (Seq2Seq / GECToR) (Modal)                      │  │
│  │  • Deberta Scorer (Modal)                              │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  All services execute in parallel using Promise.allSettled() │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 4: Process Results                                    │
│  • Process generic registry results (Corpus, GEC, etc.)     │
│  • Process legacy results (LanguageTool, LLM, Relevance)    │
│  • Extract essay scores                                     │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 5: Generate AI Feedback (if requested)                │
│  • For each answer:                                         │
│    - Generate combined feedback (LLM + Teacher)             │
│    - Retry on failure (up to 3 attempts)                    │
│  • Collect all feedback by answer ID                        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 6: Merge & Store                                      │
│  • Merge all assessment results                             │
│  • Build metadata (word count, error count, scores)         │
│  • Store results in KV (if storeResults=true)               │
│  • Build response headers (timing data)                     │
│  • Log performance metrics                                  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Response (200 OK with assessment results)
```

### Timing Breakdown

Each phase is timed and logged:

```
Timings tracked:
├─ 1_parse_request
├─ 1b_validate_submission
├─ 1c_auto_create_entities
├─ 4_load_data_from_r2
├─ 5_parallel_services_total

│  ├─ 5b_languagetool_fetch
│  ├─ 5c_relevance_fetch
│  ├─ 5d_ai_assessment_fetch
│  ├─ 5e_corpus_fetch
│  ├─ 5f_feedback_fetch
│  ├─ 5g_gec_fetch
│  └─ 5i_deberta_fetch
├─ 5_generic_services_total
├─ 7_process_languagetool
├─ 7b_process_ai_assessment
├─ 8_ai_feedback
├─ 9_process_relevance
├─ 10_merge_results
├─ 10a_metadata
├─ 11_store_results
└─ 0_total (overall)
```

---

## Service Architecture

### File Organization

```
src/
├── index.ts                     # Application entry point
├── config/                      # Configuration management
├── middleware/                  # Request middleware
├── routes/                      # Route handlers
│   ├── feedback/                # Feedback sub-routes (streaming/teacher)
│   ├── health.ts                # Health checks
│   ├── questions.ts             # Question CRUD
│   └── submissions.ts           # Main submission handler
├── services/                    # Business logic
│   ├── ai-assessment/           # LLM-based assessment logic
│   ├── clients/                 # HTTP clients
│   ├── feedback/                # Feedback generation logic
│   ├── modal/                   # Modal service integration
│   ├── submission/              # Submission processing core
│   │   ├── service-registry.ts  # Registry of all assessment services
│   │   ├── services.ts          # Service execution orchestration
│   │   ├── submission-processor.ts # Main orchestration logic
│   │   └── ...                  # Result processing helpers
│   └── storage.ts               # Storage abstraction
├── utils/                       # Shared utilities
└── types/                       # TypeScript definitions
```

### Service Registry Pattern

The API uses a **Service Registry** pattern (`src/services/submission/service-registry.ts`) to manage the diverse set of assessment services. This allows new services (scorers, grammar checkers) to be added with minimal code changes.

Each service in the registry defines:

- **ID**: Unique identifier (e.g., `AES-DEBERTA`, `GEC-SEQ2SEQ`).
- **Config Path**: Where to find its enabling flag in the app config.
- **Request Factory**: How to build the request for the Modal service.
- **Response Parser**: How to interpret the service's output.

### Service Dependencies

```
processSubmission (submission-processor.ts)
    │
    ├─→ validateSubmissionBody
    │
    ├─→ loadSubmissionData
    │       │
    │       └─→ prepareServiceRequests (services.ts)
    │               │
    │               └─→ createServiceRequests (service-registry.ts)
    │                       Creates Generic Requests (Corpus, GEC, Deberta, etc.)
    │
    ├─→ executeServiceRequests (services.ts)
    │       │
    │       ├─→ Generic Registry Services (Parallel Batch)
    │       │
    │       └─→ Legacy/Specialized Services
    │             ├─→ LanguageTool
    │             ├─→ LLM Assessment (OpenAI/Groq)
    │             └─→ Relevance Check
    │
    ├─→ processServiceResults
    │       │
    │       ├─→ processLanguageToolResults
    │       ├─→ processLLMResults
    │       └─→ processRelevanceResults
    │
    ├─→ generateCombinedFeedback (feedback.ts)
    │       │
    │       └─→ LLM/Teacher Feedback (OpenAI/Groq)
    │
    └─→ mergeAssessmentResults (merge-results.ts)
            Merges all registry results + legacy results into final response
```

---

## Storage Architecture

### Storage Services

The API uses two Cloudflare storage services:

```
┌────────────────────────────────────────────────────────────┐
│                    Storage Service                         │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              R2 Bucket (WRITEO_DATA)                 │  │
│  │                                                      │  │
│  │  • Questions: question:{id} → { text: string }       │  │
│  │  • Answers: answer:{id} → { question-id, text }      │  │
│  │  • Submissions: submission:{id} → full submission    │  │
│  │                                                      │  │
│  │  Operations:                                         │  │
│  │  - getQuestion(id)                                   │  │
│  │  - putQuestion(id, data)                             │  │
│  │  - getAnswer(id)                                     │  │
│  │  - putAnswer(id, data)                               │  │
│  │  - getSubmission(id)                                 │  │
│  │  - putSubmission(id, data)                           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          KV Namespace (WRITEO_RESULTS)               │  │
│  │                                                      │  │
│  │  • Results: submission:{id} → AssessmentResults      │  │
│  │  • Rate Limits: rate_limit:{type}:{id} → count data  │  │
│  │  • API Keys: apikey:{key} → { owner: string }        │  │
│  │                                                      │  │
│  │  Operations:                                         │  │
│  │  - getResults(id)                                    │  │
│  │  - putResults(id, data, ttl)                         │  │
│  │  - Rate limit state management                       │  │
│  │  - API key lookups                                   │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

---

## External Services

### Service Integration Diagram

```
API Worker
    │
    ├─────────────────────────────────────────────────────┐
    │                                                     │
    ▼                                                     ▼
┌───────────────┐                                  ┌──────────────┐
│ Modal Service │                                  │ OpenAI/Groq  │
│               │                                  │              │
│ • Corpus      │                                  │ • LLM        │
│   Scorer      │                                  │   Assessment │
│               │                                  │              │
│ • GEC Services│                                  │ • Streaming  │
│   (Seq2Seq)   │                                  │   Feedback   │
│               │                                  │              │
│ • Deberta     │                                  │ • Teacher    │
│   Scorer      │                                  │   Feedback   │
│               │                                  │              │
│ • LanguageTool│                                  │              │
└───────────────┘                                  └──────────────┘
```

### Service Details

| Service                | Purpose                            | Endpoint            | Authentication    |
| ---------------------- | ---------------------------------- | ------------------- | ----------------- |
| **Modal Services**     | General assessment (Scorers, GEC)  | `MODAL_DEBERTA_URL` | API key in header |
| **Modal LanguageTool** | Grammar and style checking         | `MODAL_LT_URL`      | API key in header |
| **OpenAI API**         | LLM assessment & feedback          | `api.openai.com`    | `OPENAI_API_KEY`  |
| **Groq API**           | LLM assessment & feedback (faster) | `api.groq.com`      | `GROQ_API_KEY`    |

---

## Error Handling

### Error Flow

```
Request Handler
    │
    ├─→ Try Block
    │     │
    │     ├─→ Validation Error → 400 Bad Request
    │     ├─→ Auth Error → 401 Unauthorized
    │     ├─→ Rate Limit → 429 Too Many Requests
    │     ├─→ Not Found → 404 Not Found
    │     ├─→ Conflict → 409 Conflict
    │     └─→ Service Error → Continue (graceful degradation)
    │
    └─→ Catch Block
          │
          ├─→ Log Error (with request ID)
          │     │
          │     └─→ Sanitize sensitive data
          │
          └─→ Return 500 Internal Server Error
                │
                └─→ Sanitize message in production
```

### Error Response Format

All errors follow this format:

```json
{
  "error": "Error message here"
}
```

**Production Behavior:**

- 4xx errors: Full error message returned
- 5xx errors: Sanitized message ("An internal error occurred. Please try again later.")
- Error details logged with request ID for debugging

---

## Security

### Security Layers

1.  **Network Security**: Cloudflare DDoS protection, HTTPS only, Security headers.
2.  **Authentication**: API key required (Admin/Test/User keys). keys stored in Cloudflare secrets.
3.  **Rate Limiting**: Per-endpoint and daily limits. IP-based fallback.
4.  **Input Validation**: Zod schemas, size limits (1MB), text validation.
5.  **Data Sanitization**: Production-safe error messages, sensitive data redacted from logs.

### Security Features

| Feature                | Implementation                        | Notes                                           |
| ---------------------- | ------------------------------------- | ----------------------------------------------- |
| **API Key Auth**       | Required for all protected routes     | Keys validated in order: admin → test → user    |
| **Rate Limiting**      | Per-endpoint and daily limits         | Different limits for test vs production keys    |
| **Input Validation**   | Zod schemas + custom validation       | Prevents injection, XSS, and oversized requests |
| **Error Sanitization** | Production-safe error messages        | Prevents information leakage                    |
| **Log Sanitization**   | Automatic redaction of sensitive data | API keys, tokens, etc. removed from logs        |

---

## Deployment

### Cloudflare Workers

- **Platform**: Cloudflare Workers
- **Runtime**: V8 Isolate
- **Regions**: Global edge network
- **Scaling**: Automatic, per-request

### Environment Variables

Required:

- `API_KEY`: Admin API key
- `MODAL_DEBERTA_URL`: Essay assessment service URL
- `OPENAI_API_KEY` or `GROQ_API_KEY`: LLM provider key

Optional:

- `TEST_API_KEY`: Test key with higher rate limits
- `LLM_PROVIDER`: "openai" or "groq" (default: "openai")
- `ALLOWED_ORIGINS`: CORS whitelist
- `ENVIRONMENT`: "development" or "production"

### Storage Bindings

- `WRITEO_DATA`: R2 bucket for questions/answers/submissions
- `WRITEO_RESULTS`: KV namespace for results and rate limits
- `AI`: Cloudflare AI binding (optional)
