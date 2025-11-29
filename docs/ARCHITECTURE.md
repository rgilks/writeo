# Writeo System Architecture

**Version:** 2.0  
**Architecture Pattern:** Serverless Edge Computing with ML-as-a-Service

**Live Services:**

- [Web Frontend](https://writeo.tre.systems) - Interactive essay submission interface
- [API Documentation](https://writeo-api.workers.dev/docs) - Interactive Swagger UI (available at `/docs` endpoint)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Components & Technology](#3-components--technology)
4. [Data Flow](#4-data-flow)
5. [Storage Architecture](#5-storage-architecture)
6. [Performance](#6-performance)

---

## 1. System Overview

Writeo provides comprehensive essay assessment:

- **Essay Scoring** - Multi-dimensional analysis (TA, CC, Vocab, Grammar, Overall) using ML models
- **AI Feedback** - Context-aware feedback using OpenAI (GPT-4o-mini) or Groq (Llama 3.3 70B) - switchable providers
- **Grammar Checking** - LanguageTool integration with inline error annotations
- **Relevance Validation** - Fast embeddings-based answer relevance checking
- **CEFR Mapping** - Automatic conversion to A2-C2 proficiency levels
- **Scale-to-Zero** - Cost-effective serverless architecture

### Operational Modes

Writeo supports two operational modes optimized for different use cases:

#### 🪙 Cheap Mode (Cost-Optimized)

**Configuration:**

- **LLM:** OpenAI GPT-4o-mini
- **Modal Services:** Scale-to-zero after 30 seconds

**Processing Flow:**

```
Client → API Worker → [Essay Scoring + LanguageTool + Relevance Check (parallel)] → OpenAI Feedback → KV Storage → Client
```

**Performance:**

- **Cold Start:** 8-15s (first request after inactivity - Modal cold start)
- **Warm:** 3-10s (subsequent requests)
- **Best For:** Cost-conscious deployments, variable traffic

#### ⚡ Turbo Mode (Performance-Optimized)

**Configuration:**

- **LLM:** Groq Llama 3.3 70B Versatile
- **Modal Services:** Keep warm (reduced scaledown window)

**Processing Flow:**

```
Client → API Worker → [Essay Scoring + LanguageTool + Relevance Check (parallel, warm)] → Groq Feedback (ultra-fast) → KV Storage → Client
```

**Performance:**

- **First Request:** 2-5s (Modal warm, Groq ultra-fast)
- **Warm:** 1-3s (all services warm)
- **Best For:** Production deployments requiring low latency

### Current Implementation

The system uses **synchronous processing** - all assessment is completed before returning results:

**Key Features:**

- Synchronous processing: Results returned immediately in PUT response body
- Parallel processing: All services run concurrently
- Streaming: Real-time AI feedback generation via Server-Sent Events (separate endpoint)
  - Teacher feedback uses streaming by default in the UI for better user experience
  - Feedback appears incrementally as it's generated, providing immediate visual feedback
- Optimized: Combined LLM calls, parallelized R2 operations, model caching
- Flexible scaling: Choose scale-to-zero (Cheap Mode) or keep-warm (Turbo Mode)

---

## 2. Architecture Diagram

```mermaid
graph TB
    External[🔌 External System<br/>API Client] -->|HTTPS REST API| API[⚡ API Worker<br/>Cloudflare Workers]
    User[👤 End User] -->|Uses| Web[🌐 Web Frontend<br/>Next.js]

    Web -->|Server Actions| API

    API -->|Store/Retrieve| R2[(💾 R2 Storage<br/>Questions, Answers, Submissions)]
    API -->|Store/Retrieve| KV[(🗄️ KV Store<br/>Assessment Results)]
    API -->|POST /grade<br/>Parallel| Essay[🤖 Essay Scoring Service<br/>FastAPI + ML Models]
    API -->|POST /check<br/>Parallel| LT[📝 LanguageTool Service<br/>FastAPI + LanguageTool]
    API -->|Embeddings| RELEVANCE[🔍 Relevance Check<br/>Cloudflare AI Embeddings]

    Essay -->|Essay Scores| API
    LT -->|Grammar Errors| API
    RELEVANCE -->|Similarity Score| API
    API -->|With Context| AI[💬 AI Feedback<br/>OpenAI - GPT-4o-mini]
    AI -->|Contextual Feedback| API

    style User fill:#e1f5ff
    style Web fill:#e1f5ff
    style External fill:#ffe1e1
    style API fill:#fff4e1
    style R2 fill:#e1ffe1
    style KV fill:#ffe1e1
    style Essay fill:#ffe1f5
    style LT fill:#e1f5ff
```

**Processing Flow:**

1. **End User** uses Web Frontend UI, OR **External System** calls API directly
2. **If `storeResults: true` (opt-in):** API Worker stores questions, answers, and submission in R2 (parallelized)
3. API Worker builds request from inline data (answers are always sent inline)
4. API Worker calls **services in parallel**:
   - Essay Scoring Service (`modal-essay`) for essay scoring
   - LanguageTool Service (`modal-lt`) for grammar checking
   - Relevance Check (Cloudflare AI embeddings) for fast relevance validation
5. API Worker merges results from parallel services
6. API Worker calls **AI Feedback** (OpenAI) with full context from Essay Scoring and LanguageTool results
7. API Worker merges results from all services (including AI feedback)
8. **If `storeResults: true` (opt-in):** Results stored in KV (90-day TTL)
9. Results returned to client immediately in PUT response body (typically 3-10s, max <20s)
10. **Default behavior:** Results stored only in browser localStorage (no server storage)

---

## 3. Components & Technology

### 3.1 Cloudflare Components

| Component           | Technology                  | Responsibility                                                       | Scale-to-Zero        |
| ------------------- | --------------------------- | -------------------------------------------------------------------- | -------------------- |
| **Web Frontend**    | Next.js 15+ (App Router)    | User interface, form handling, result display, history page          | ✅ Yes               |
| **API Worker**      | Cloudflare Workers          | REST API, request validation, data orchestration                     | ✅ Yes               |
| **R2 Storage**      | Cloudflare R2               | Persistent storage for questions, answers, submissions (opt-in only) | ❌ No (storage only) |
| **KV Store**        | Cloudflare KV               | Assessment results cache (90-day TTL, opt-in only)                   | ❌ No (storage only) |
| **Browser Storage** | localStorage/sessionStorage | Default storage location (client-side only)                          | ✅ Yes (client-side) |

### 3.1.1 Frontend State Management

**Technology Stack:**

- **Zustand** - Global state management
- **Immer** - Immutable state updates
- **useState** - Component-specific UI state

**Stores:**

- **Draft Store** - Draft content, assessment results, draft history, progress tracking, achievements, streaks
- **Preferences Store** - User preferences (view mode, storage settings)

**Key Features:**

- Selective subscriptions prevent unnecessary re-renders
- Computed selectors for expensive calculations
- DevTools integration for debugging
- Automatic localStorage persistence

See [STATE_MANAGEMENT.md](STATE_MANAGEMENT.md) for detailed documentation.

### 3.2 Modal Services

| Service                   | Technology             | Responsibility                    | Scale-to-Zero |
| ------------------------- | ---------------------- | --------------------------------- | ------------- |
| **Essay Scoring Service** | FastAPI + PyTorch      | ML model inference, essay scoring | ✅ Yes        |
| **LanguageTool**          | FastAPI + LanguageTool | Grammar, spelling, style checking | ✅ Yes        |

### 3.3 Cloudflare Workers AI

| Service             | Technology            | Responsibility                             | Scale-to-Zero |
| ------------------- | --------------------- | ------------------------------------------ | ------------- |
| **AI Feedback**     | OpenAI API            | Context-aware essay feedback (GPT-4o-mini) | ✅ Yes        |
| **Relevance Check** | Cloudflare Workers AI | Fast embeddings-based relevance validation | ✅ Yes        |

**AI Feedback:**

- **Multi-Provider Support**: Choose between OpenAI (GPT-4o-mini) or Groq (Llama 3.3 70B Versatile)
- **OpenAI (GPT-4o-mini)**: Cost-effective, excellent quality, ~1-3s inference
- **Groq (Llama 3.3 70B Versatile)**: Ultra-fast (~100-500ms inference), excellent quality
- Receives full context from essay scores and LanguageTool errors
- Provides contextual, actionable feedback tailored to student's level
- Switch providers via `LLM_PROVIDER` environment variable

**Relevance Check:**

- Uses embeddings model (`@cf/baai/bge-base-en-v1.5`)
- Fast cosine similarity calculation (~100-200ms)

**Essay Scoring Service (`modal-essay`):**

- Uses `KevSun/Engessay_grading_ML` model (default)
  - Citation: Sun, K., & Wang, R. (2024). Automatic Essay Multi-dimensional Scoring with Fine-tuning and Multiple Regression. _ArXiv_. https://arxiv.org/abs/2406.01198
  - Well-suited for academic argumentative writing practice
  - Provides strong coverage of Coherence & Cohesion, Lexical Resource, and Grammatical Range & Accuracy
  - Task Achievement assessed separately via LLM feedback for comprehensive evaluation
- Multi-dimensional scoring: TA, CC, Vocab, Grammar, Overall
- CEFR level mapping (A2-C2)
- GPU/CPU inference with model caching

**LanguageTool Service (`modal-lt`):**

- Open-source grammar checker
- Detects grammar, spelling, and style errors
- Provides suggestions and corrections
- CPU-only (no GPU needed)
- Fast warm checks (~100-500ms)

### 3.3 API Worker Components

```mermaid
graph TB
    subgraph "API Worker"
        Router[Hono Router]
        Auth[Auth Middleware]
        Validator[Request Validator]
        SubmissionHandler[Submission Handler]
        ResultsHandler[Results Handler]
        R2Client[R2 Client]
        KVClient[KV Client]
        ModalClient[Modal Client]
        LTClient[LanguageTool Client]
        Merger[Result Merger]
    end

    Router --> Auth
    Auth --> Validator
    Validator --> SubmissionHandler
    SubmissionHandler --> R2Client
    SubmissionHandler --> ModalClient
    SubmissionHandler --> LTClient
    SubmissionHandler --> Merger
    SubmissionHandler --> KVClient
    ResultsHandler --> KVClient
```

**Key Components:**

- **Submission Handler**: Orchestrates parallel calls to Essay Scoring and LanguageTool services
- **Result Merger**: Combines essay scores and grammar errors into unified response
- **R2/KV Clients**: Handle storage operations

---

## 4. Data Flow

### 4.1 Submission Processing Flow

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant API as API Worker
    participant R2 as R2 Storage
    participant Essay as Essay Scoring Service
    participant LT as LanguageTool Service
    participant KV as KV Storage

    Client->>API: PUT /text/submissions/:id
    API->>R2: Store submission
    API->>R2: Read answers & questions

    par Parallel Processing
        API->>Essay: POST /grade
        Essay-->>API: Essay scores
    and
        API->>LT: POST /check (for each answer)
        LT-->>API: Grammar errors
    end

    API->>API: Merge results
    API->>KV: Store merged results
    API-->>Client: 200 OK (with results in body)

    Client->>API: GET /text/submissions/:id
    API->>KV: Read results
    API-->>Client: Assessment results
```

### 4.2 Assessment Results Structure

**Merged Results Format:**

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
                "length": 4,
                "category": "GRAMMAR",
                "rule_id": "SVA",
                "message": "Possible subject–verb agreement error.",
                "suggestions": ["go", "went"],
                "source": "LT",
                "severity": "error"
              }
            ],
            "meta": {
              "language": "en-GB",
              "engine": "LT-OSS",
              "errorCount": 1
            }
          }
        ]
      }
    ]
  },
  "template": { "name": "generic", "version": 1 },
  "meta": {
    "answerTexts": {
      "answer-id": "Original essay text..."
    }
  }
}
```

---

## 5. Storage Architecture

**Important:** Writeo uses an **opt-in server storage model**. By default (`storeResults: false`), no data is stored on servers. Results are stored only in the user's browser (localStorage). Server storage (R2/KV) is only used when `storeResults: true` is explicitly set.

### 5.1 Browser Storage (Default)

**Location:** Client-side localStorage/sessionStorage

| Storage Type     | Purpose                             | Retention                      |
| ---------------- | ----------------------------------- | ------------------------------ |
| `localStorage`   | Persistent results storage          | Until user clears browser data |
| `sessionStorage` | Temporary results during navigation | Until browser tab closes       |

**Access Patterns:**

- **Write**: Automatic after submission processing (including draft creation)
  - Results are stored immediately in both `localStorage` and `sessionStorage` after creation
  - This ensures immediate availability even when server storage is enabled (`storeResults: true`)
  - Critical for draft tracking: draft results are available immediately after creation
- **Read**: Immediate access from browser
  - Results page checks `localStorage`/`sessionStorage` first before attempting server fetch
  - This prevents "Results Not Available" errors when creating drafts with server storage enabled
- **Privacy**: Data never leaves user's device (when `storeResults: false`)

**Benefits:**

- ✅ Maximum privacy (data never leaves user's device)
- ✅ No legal compliance requirements (no server data collection)
- ✅ User controls data lifecycle
- ✅ No data retention policies needed
- ✅ No deletion/export APIs needed
- ✅ No data breach notification needed

### 5.2 R2 Object Storage (Opt-in Only)

**Bucket:** `writeo-data`  
**Usage:** Only when `storeResults: true`

| Path Pattern                       | Content Type       | Structure                                    |
| ---------------------------------- | ------------------ | -------------------------------------------- |
| `questions/{question_id}.json`     | `application/json` | `{text: string}`                             |
| `answers/{answer_id}.json`         | `application/json` | `{question-id: string, answer-text: string}` |
| `submissions/{submission_id}.json` | `application/json` | `{submission: Part[], template: {}}`         |

**Access Patterns:**

- **Write**: Single PUT per resource creation (opt-in only)
- **Read**: Batch reads during submission processing (opt-in only)
- **TTL**: No automatic expiration (consider lifecycle policies)

### 5.3 KV Storage (Opt-in Only)

**Namespace:** `WRITEO_RESULTS`  
**Usage:** Only when `storeResults: true`

| Key Pattern                  | Value Type  | TTL                         |
| ---------------------------- | ----------- | --------------------------- |
| `submission:{submission_id}` | JSON string | 90 days (7,776,000 seconds) |

**Access Patterns:**

- **Write**: Single PUT after processing completes (opt-in only)
- **Read**: Single GET per client poll request (opt-in only)
- **Consistency**: Eventual consistency (read-after-write may have delay)

### 5.4 API Behavior

**PUT `/text/submissions/{id}`:**

- **Default (`storeResults: false`):**
  - Processes submission
  - Returns results immediately in response body
  - Stores results in browser localStorage
  - Does NOT store on server (R2/KV)

- **Opt-in (`storeResults: true`):**
  - Processes submission
  - Returns results immediately in response body
  - Stores results in browser localStorage and sessionStorage immediately (critical for draft tracking)
  - Also stores on server (R2/KV) for 90 days (may have slight delay)
  - **Important**: Results are available immediately from localStorage even if server storage isn't ready yet

**GET `/text/submissions/{id}`:**

- Only works if `storeResults: true` was set during submission
- Returns 404 if submission was not stored on server
- Returns results if found (within 90-day retention period)

### 5.5 Privacy & Legal Implications

**Default (No Server Storage):**

- ✅ No GDPR/CCPA requirements (no server data collection)
- ✅ No COPPA requirements (no data collection from children)
- ✅ No data deletion API needed
- ✅ No data export API needed
- ✅ No data breach notification needed
- ✅ No age verification needed

**Opt-in (Server Storage):**

- ⚠️ GDPR/CCPA requirements apply (server data collection)
- ⚠️ Data deletion API recommended
- ⚠️ Data export API recommended
- ⚠️ Data breach notification procedures needed
- ⚠️ Age verification may be needed (if serving children)

See [LEGAL_COMPLIANCE.md](LEGAL_COMPLIANCE.md) for detailed compliance information.

### 5.6 Best Practices

1. **Default to no storage**: Use `storeResults: false` by default
2. **Make opt-in clear**: Explain benefits of server storage to users
3. **Respect user choice**: Don't force server storage
4. **Provide alternatives**: Browser storage is sufficient for most use cases
5. **Document retention**: Clearly state 90-day retention for opt-in storage

### 5.7 Data Size Estimates

| Resource Type      | Average Size | Max Size | Storage Location (Default) | Storage Location (Opt-in) |
| ------------------ | ------------ | -------- | -------------------------- | ------------------------- |
| Question           | ~100 bytes   | 1 KB     | Not stored                 | R2                        |
| Answer (essay)     | ~2 KB        | 50 KB    | Not stored                 | R2                        |
| Submission         | ~500 bytes   | 5 KB     | Not stored                 | R2                        |
| Assessment Results | ~1-5 KB      | 10 KB    | localStorage               | KV (90-day TTL)           |

---

## 6. Performance

### 6.1 Latency (Warm vs Cold)

| Step                                  | Warm (P50) | Cold      | Notes                                |
| ------------------------------------- | ---------- | --------- | ------------------------------------ |
| **PUT /text/submissions/{id}**        | ~0.94s     | 11.3s     | Includes parallel Modal calls        |
| **Modal POST /grade (Essay Scoring)** | ~0.2s      | 9.5-11.5s | GPU inference + model loading        |
| **Modal POST /check (LanguageTool)**  | ~0.2-0.4s  | 7.5-8.5s  | CPU-only, JAR download on cold start |
| **GET /text/submissions/{id}**        | 6ms        | n/a       | KV read                              |
| **End-to-end (user experience)**      | ~1.8-2.5s  | ~12s      | Full submission → results visible    |

**Bottleneck Analysis:**

- **Primary bottleneck**: AI Feedback generation (13-18 seconds per answer)
- **Secondary bottleneck**: Modal cold starts (8-15s for Essay Scoring, 2-5s for LanguageTool)
- **R2 Operations**: Already parallelized using `Promise.all()` for optimal performance

### 6.2 Throughput

| Metric                   | Value                                   |
| ------------------------ | --------------------------------------- |
| **Max Requests per Day** | 100,000 (free tier limit)               |
| **Concurrent Requests**  | Unlimited (auto-scales per request)     |
| **Average Requests/sec** | ~1.2 requests/second (100k/day average) |
| **Burst Capacity**       | Handles traffic spikes automatically    |

### 6.3 Cost Information

See [COST_REVIEW.md](COST_REVIEW.md) for detailed cost analysis, including per-submission costs, monthly estimates, and cost controls.

### 6.4 Performance Optimizations Implemented

**Current Optimizations:**

1. ✅ **Parallelized R2 Operations** - All R2 reads/writes use `Promise.all()`
2. ✅ **Parallel Service Calls** - Essay Scoring, LanguageTool, and Relevance checks run concurrently
3. ✅ **Combined AI Feedback Calls** - Single LLM call per answer (50% fewer API calls)
4. ✅ **Model Caching** - Modal Volume caches model weights for faster cold starts
5. ✅ **Synchronous Processing** - Results returned immediately in PUT response body (typically 3-10s)

---

## Appendix: Technology Stack Details

### Frontend Stack

- **Framework**: Next.js 15+ (App Router)
- **Language**: TypeScript 5+
- **Styling**: Global CSS with CSS Variables
- **API Client**: Server Actions (server-side only)

### API Worker Stack

- **Runtime**: Cloudflare Workers
- **Framework**: Hono 4+
- **Language**: TypeScript 5+
- **Storage**: R2 API, KV API

### Modal Services Stack

- **Platform**: Modal (serverless ML)
- **Framework**: FastAPI 0.104+
- **Language**: Python 3.11+
- **ML Framework**: PyTorch 2.1.0+
- **Models**: HuggingFace Transformers 4.40+
- **Grammar**: LanguageTool 6.4+

---
