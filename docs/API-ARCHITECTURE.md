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
└─────────────────────────────────────────────────────────────┘
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
│    - Questions: PUT /text/questions/:question_id            │
│    - Submissions: PUT /text/submissions/:submission_id      │
│    - Results: GET /text/submissions/:submission_id          │
│    - Feedback: POST /text/submissions/:id/ai-feedback/stream│
│    - Feedback: POST /text/submissions/:id/teacher-feedback  │
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
PUT /text/submissions/:submission_id
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
│  • Prepare service requests (essay, LT, LLM, relevance)     │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Phase 3: Execute Services (PARALLEL)                         │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ Essay        │  │ LanguageTool │  │ LLM          │        │
│  │ Assessment   │  │ Grammar      │  │ Assessment   │        │
│  │ (Modal)      │  │ Check        │  │ (OpenAI/     │        │
│  │              │  │ (Modal)      │  │  Groq)       │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│         │                  │                  │              │
│         └──────────────────┼──────────────────┘              │
│                            │                                 │
│  ┌──────────────────────────────────────────────┐            │
│  │ Relevance Check (Modal)                      │            │
│  └──────────────────────────────────────────────┘            │
│                                                              │
│  All services execute in parallel using Promise.allSettled() │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 4: Process Results                                    │
│  • Process essay assessment results                         │
│  • Process LanguageTool errors                              │
│  • Process LLM assessment errors                            │
│  • Process relevance check results                          │
│  • Extract essay scores                                     │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 5: Generate AI Feedback                               │
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
│  ├─ 5a_essay_fetch
│  ├─ 5b_languagetool_fetch
│  ├─ 5c_relevance_fetch
│  └─ 5d_ai_assessment_fetch
├─ 6_process_essay
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

### Service Layer Organization

```
services/
├── submission-processor.ts      # Main orchestration
├── submission/
│   ├── validator.ts            # Request validation
│   ├── data-loader.ts          # Data loading & Modal request building
│   ├── storage.ts              # Entity storage operations
│   ├── services.ts             # Service request preparation & execution
│   ├── results-essay.ts        # Essay result processing
│   ├── results-languagetool.ts # LanguageTool result processing
│   ├── results-llm.ts          # LLM result processing
│   ├── results-relevance.ts    # Relevance check processing
│   ├── results-scores.ts       # Score extraction
│   └── metadata.ts             # Metadata building
├── feedback.ts                 # Feedback generation orchestration
├── feedback/
│   ├── combined.ts            # Combined feedback generation
│   ├── teacher.ts             # Teacher feedback generation
│   ├── retry.ts               # Retry logic
│   └── ...
├── ai-assessment.ts           # LLM assessment logic
├── merge-results.ts           # Result merging
├── storage.ts                 # Storage service abstraction
└── clients/
    ├── base-client.ts         # Base HTTP client
    ├── essay-client.ts        # Essay service client
    └── languagetool-client.ts # LanguageTool client
```

### Service Dependencies

```
processSubmission
    │
    ├─→ validateAndParseSubmission
    │       │
    │       └─→ validateSubmissionBody (Zod)
    │
    ├─→ loadSubmissionData
    │       │
    │       ├─→ storeSubmissionEntities
    │       │       │
    │       │       └─→ StorageService (R2)
    │       │
    │       └─→ buildModalRequest
    │               │
    │               └─→ StorageService (R2)
    │
    ├─→ executeServiceRequests
    │       │
    │       ├─→ EssayClient → Modal Service
    │       ├─→ LanguageToolClient → Modal Service
    │       ├─→ LLM Assessment → OpenAI/Groq
    │       └─→ Relevance Check → Modal Service
    │
    ├─→ processServiceResults
    │       │
    │       ├─→ processEssayResult
    │       ├─→ processLanguageToolResults
    │       ├─→ processLLMResults
    │       ├─→ processRelevanceResults
    │       └─→ extractEssayScores
    │
    ├─→ generateCombinedFeedback
    │       │
    │       └─→ getCombinedFeedbackWithRetry
    │               │
    │               ├─→ generateCombinedFeedback
    │               │       │
    │               │       ├─→ generateTeacherFeedback
    │               │       └─→ generateLLMFeedback
    │               │
    │               └─→ retry logic (up to 3 attempts)
    │
    └─→ mergeAssessmentResults
            │
            └─→ StorageService (KV) - if storeResults=true
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

### Data Flow: Storage Operations

```
Submission Request (storeResults=true)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Store Entities (if new)                                  │
│    R2.putQuestion(questionId, { text })                     │
│    R2.putAnswer(answerId, { question-id, text })            │
│    R2.putSubmission(submissionId, fullSubmission)           │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Load Existing Data (if needed)                           │
│    R2.getQuestion(questionId) → question text               │
│    R2.getAnswer(answerId) → answer data                     │
│    KV.getResults(submissionId) → existing results           │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Store Results (after processing)                         │
│    KV.putResults(submissionId, results, ttl=90 days)        │
└─────────────────────────────────────────────────────────────┘
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
│ • Essay       │                                  │ • LLM        │
│   Assessment  │                                  │   Assessment │
│               │                                  │              │
│ • LanguageTool│                                  │ • Streaming  │
│   Grammar     │                                  │   Feedback   │
│   Check       │                                  │              │
│               │                                  │ • Teacher    │
│ • Relevance   │                                  │   Feedback   │
│   Check       │                                  │              │
└───────────────┘                                  └──────────────┘
```

### Service Details

| Service                 | Purpose                            | Endpoint          | Authentication    |
| ----------------------- | ---------------------------------- | ----------------- | ----------------- |
| **Modal Essay Service** | Essay scoring with band scores     | `MODAL_GRADE_URL` | API key in header |
| **Modal LanguageTool**  | Grammar and style checking         | `MODAL_LT_URL`    | API key in header |
| **Modal Relevance**     | Answer relevance checking          | `MODAL_GRADE_URL` | API key in header |
| **OpenAI API**          | LLM assessment & feedback          | `api.openai.com`  | `OPENAI_API_KEY`  |
| **Groq API**            | LLM assessment & feedback (faster) | `api.groq.com`    | `GROQ_API_KEY`    |

### Service Request Flow

```
Parallel Service Execution
    │
    ├─→ Essay Service
    │     │
    │     └─→ POST MODAL_GRADE_URL
    │           Body: { submission_id, template, parts }
    │           Response: AssessmentResults
    │
    ├─→ LanguageTool Service
    │     │
    │     └─→ POST MODAL_LT_URL
    │           Body: { text, language }
    │           Response: LanguageToolError[]
    │
    ├─→ LLM Assessment
    │     │
    │     └─→ POST OpenAI/Groq API
    │           Body: { model, messages, ... }
    │           Response: LLM errors & suggestions
    │
    └─→ Relevance Check
          │
          └─→ POST MODAL_GRADE_URL
                Body: { question, answer }
                Response: RelevanceCheck
```

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

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Network Security                                   │
│  • Cloudflare DDoS protection                               │
│  • HTTPS only                                               │
│  • Security headers (X-Content-Type-Options, etc.)          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: Authentication                                     │
│  • API key required (except /health, /docs)                 │
│  • Key validation: Admin → Test → User (KV lookup)          │
│  • Keys stored securely (Cloudflare secrets)                │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Rate Limiting                                      │
│  • Per-endpoint limits (submissions: 10/min)                │
│  • Daily submission limits (100/day)                        │
│  • IP-based or user-based tracking                          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: Input Validation                                   │
│  • Request body size limits (1MB max)                       │
│  • Text validation (length, dangerous patterns)             │
│  • Schema validation (Zod)                                  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 5: Data Sanitization                                  │
│  • Error messages sanitized in production                   │
│  • Logs sanitized (API keys, tokens removed)                │
│  • Sensitive data redacted                                  │
└─────────────────────────────────────────────────────────────┘
```

### Security Features

| Feature                | Implementation                                | Notes                                           |
| ---------------------- | --------------------------------------------- | ----------------------------------------------- |
| **API Key Auth**       | Required for all protected routes             | Keys validated in order: admin → test → user    |
| **Rate Limiting**      | Per-endpoint and daily limits                 | Different limits for test vs production keys    |
| **Input Validation**   | Zod schemas + custom validation               | Prevents injection, XSS, and oversized requests |
| **Error Sanitization** | Production-safe error messages                | Prevents information leakage                    |
| **Log Sanitization**   | Automatic redaction of sensitive data         | API keys, tokens, etc. removed from logs        |
| **CORS**               | Configurable origin whitelist                 | Optional (security via API key)                 |
| **Security Headers**   | X-Content-Type-Options, X-Frame-Options, etc. | Applied to all responses                        |

---

## Request Tracing & Observability

### Request ID Flow

Every request gets a unique ID that flows through the system:

```
Request → requestId middleware
    │
    ├─→ Generate ID: crypto.randomUUID().split("-")[0]
    │     Example: "a1b2c3d4"
    │
    ├─→ Store in context: c.set("requestId", id)
    │
    └─→ Include in all logs: [req-a1b2c3d4] Message here
```

### Logging Structure

All logs include the request ID prefix:

```
[req-a1b2c3d4] Request completed {
  submissionId: "123",
  endpoint: "/text/submissions/123",
  method: "PUT",
  timings: { "0_total": 2345.67, ... },
  totalMs: "2345.67"
}
```

### Performance Metrics

Performance metrics are logged at the end of each submission request:

- **Timing breakdown**: Each phase is timed separately
- **Total duration**: Overall request time
- **Service timings**: Individual service call durations
- **Response headers**: Timing data included in `X-Timing-Data` header

---

## Key Design Decisions

### 1. Synchronous Processing

**Decision**: Results returned immediately in PUT response body

**Rationale**:

- Simpler client implementation (no polling needed)
- Better user experience (immediate feedback)
- Processing time is acceptable (3-10 seconds typical)

### 2. Parallel Service Execution

**Decision**: All assessment services run in parallel

**Rationale**:

- Faster overall processing time
- Services are independent
- Uses `Promise.allSettled()` for graceful degradation

### 3. Fail-Safe Production Detection

**Decision**: Defaults to production when environment is unclear

**Rationale**:

- Security: Better to sanitize errors than expose them
- Prevents accidental information leakage
- Explicit `ENVIRONMENT=development` required for dev mode

### 4. Rate Limiting Fail-Open

**Decision**: Rate limiting errors don't block requests

**Rationale**:

- Prevents rate limiting bugs from blocking legitimate traffic
- Errors logged for monitoring
- Better availability

### 5. Minimal Documentation

**Decision**: Only document non-obvious behavior

**Rationale**:

- Code should be self-documenting
- Reduces maintenance burden
- Focuses on what's actually helpful

---

## File Organization

```
src/
├── index.ts                    # Application entry point
├── middleware/                 # Request middleware
│   ├── auth.ts                # API key authentication
│   ├── rate-limit.ts          # Rate limiting
│   ├── request-id.ts          # Request ID generation
│   └── security.ts            # Security headers
├── routes/                    # Route handlers
│   ├── health.ts              # Health check & docs
│   ├── questions.ts           # Question management
│   ├── submissions.ts         # Submission processing
│   └── feedback.ts           # Feedback endpoints
├── services/                  # Business logic
│   ├── submission-processor.ts # Main orchestration
│   ├── submission/            # Submission processing
│   ├── feedback/              # Feedback generation
│   ├── storage.ts             # Storage abstraction
│   └── clients/               # External service clients
├── utils/                     # Utility functions
│   ├── errors.ts              # Error handling
│   ├── logging.ts             # Logging utilities
│   ├── validation.ts          # Input validation
│   └── context.ts             # Context helpers
└── types/                     # TypeScript types
    └── env.ts                 # Environment bindings
```

---

## Performance Characteristics

### Typical Request Times

| Endpoint              | Typical Time | Notes                           |
| --------------------- | ------------ | ------------------------------- |
| Health Check          | <10ms        | No processing                   |
| Question Creation     | <100ms       | Simple R2 write                 |
| Submission Processing | 3-10s        | Parallel services + AI feedback |
| Results Retrieval     | <50ms        | KV read                         |
| Streaming Feedback    | 2-5s         | Depends on LLM response time    |

### Optimization Strategies

1. **Parallel Service Execution**: All assessment services run concurrently
2. **Caching**: Results stored in KV for fast retrieval
3. **Early Validation**: Invalid requests rejected before expensive operations
4. **Graceful Degradation**: Service failures don't block entire request
5. **Request Timeouts**: External service calls have timeouts

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
- `MODAL_GRADE_URL`: Essay assessment service URL
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

---

## Future Considerations

### Potential Improvements

1. **Caching Layer**: Add caching for frequently accessed questions
2. **Batch Processing**: Support batch submission processing
3. **Webhooks**: Notify clients when processing completes
4. **Metrics Aggregation**: Collect and aggregate performance metrics
5. **Request Queuing**: Queue long-running requests for async processing

### Scalability

The current architecture scales well because:

- Stateless workers (no shared state)
- Cloudflare's global edge network
- Parallel service execution
- Efficient storage (R2 + KV)

Limitations:

- 30-second execution time limit (Cloudflare Workers)
- KV write limits (1000 writes/second per namespace)
- R2 operation limits (based on plan)

---

## Conclusion

The API Worker is designed for:

- **Simplicity**: Clear separation of concerns, minimal abstractions
- **Performance**: Parallel execution, efficient storage
- **Reliability**: Graceful degradation, comprehensive error handling
- **Security**: Multiple layers of protection
- **Observability**: Request tracing and performance metrics

The architecture prioritizes maintainability and developer experience while ensuring high performance and security.

---

## References

- [ARCHITECTURE.md](ARCHITECTURE.md) - System-wide architecture and design
- [SPEC.md](SPEC.md) - Complete API specification
- [OPERATIONS.md](OPERATIONS.md) - Operations guide and environment configuration
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guide
- [SERVICES.md](SERVICES.md) - Service documentation
