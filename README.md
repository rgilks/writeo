# Writeo

<div align="center">

**High-Precision Revision-First Writing Practice Tool**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Node.js](https://img.shields.io/badge/node-%3E%3D18-brightgreen.svg)](https://nodejs.org/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

[Live Demo](https://writeo.tre.systems) • [API Docs](https://your-api-worker.workers.dev/docs) • [Documentation](#-documentation)

<a href='https://ko-fi.com/N4N31DPNUS' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi2.png?v=6' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>

</div>

---

## 📖 Table of Contents

- [About](#-about)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [API](#-api)
- [Development](#-development)
- [Documentation](#-documentation)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [Support](#-support)
- [Cost Optimization](#-cost-optimization)
- [Troubleshooting](#-troubleshooting)
- [Status](#-status)
- [Roadmap](#-roadmap)
- [License](#-license)

---

## 📖 About

Writeo is a modern, scalable writing practice system with **AI-powered feedback** built for educational use. It provides comprehensive essay assessment including scoring, grammar checking, and context-aware AI feedback.

**Philosophy**: High-precision, low-stakes, revision-first writing practice. Writeo gives cautious, explainable feedback, encourages multiple drafts, visualizes progress, and is designed to support teachers—not replace them.

**Key Principles:**

- 🎯 **Precision over Coverage** - Only highlights errors with high confidence (>80%)
- 🔄 **Revision-First** - Encourages multiple drafts and tracks progress
- 🌱 **Formative Assessment** - Growth-focused language, no pass/fail framing
- 🔐 **Privacy-First** - No cookies, local storage only, transparent data handling

**Built With:**

- ⚡ **Cloudflare Workers** - Edge API with global low-latency
- 🤖 **Modal** - ML inference for essay scoring
- 🧠 **Multi-LLM Support** - OpenAI (GPT-4o-mini) or Groq (Llama 3.3 70B) - switch between providers
- 📝 **LanguageTool** - Grammar and style checking
- ⚛️ **Next.js** - Modern React frontend
- 💾 **R2 & KV** - Serverless storage

---

## ✨ Features

### Learning Features

- 🔄 **Draft Tracking** - Link multiple drafts, track revision history, compare progress, navigate between drafts
- 🎯 **Precision-Focused Feedback** - Only highlights errors with high confidence (>80%), with toggles for medium/low confidence
- 📝 **Structured Error Feedback** - Error type, explanation, and examples with collapsible detail levels
- 🗺️ **Heat Map Visualization** - Visual problem areas without revealing exact errors (encourages discovery)
- 💭 **Metacognition Tools** - Reflection prompts, self-evaluation checklists, pattern insights
- 📊 **Humble Score Presentation** - Confidence ranges, CEFR descriptors, "Estimated Level" with disclaimers
- 🌱 **Formative Framing** - Growth-focused language throughout (no pass/fail terminology)
- 📈 **Progress Visualization** - Charts showing score, error count, and CEFR level improvements
- 🎉 **Celebratory Feedback** - Positive reinforcement when learners improve between drafts
- 👩‍🏫 **Interactive Teacher Feedback** - Short encouragement notes with optional detailed analysis

### Assessment Features

- ⚡ **Fast Processing** - Results returned synchronously in 3-10 seconds (typically), max <20s
- 🎯 **AI-Powered Feedback** - Context-aware feedback using OpenAI (GPT-4o-mini) or Groq (Llama 3.3 70B) - choose your provider
- 📊 **Essay Scoring** - Multi-dimensional analysis (TA, CC, Vocab, Grammar, Overall)
- ✅ **Relevance Checking** - Fast embeddings-based validation using Cloudflare Workers AI
- 📝 **Grammar Checking** - LanguageTool integration with inline annotations and confidence tiers
- 🌍 **CEFR Mapping** - Automatic conversion to A2-C2 with confidence indicators and progress tracking
- 🔍 **Error Confidence Tiers** - High (>80%), Medium (60-80%), Low (<60%) with user controls

### Technical Features

- ⚡ **Parallel Processing** - All services run concurrently for optimal performance
- 📡 **Streaming Support** - Server-Sent Events for real-time AI feedback (optional)
- 🔐 **Secure** - Token authentication, rate limiting, CORS, CSP headers
- 💾 **Optimized** - Model caching, parallelized operations, scale-to-zero architecture
- 🔌 **RESTful API** - Standard request/response formats with OpenAPI specification
- 📱 **Responsive Design** - Works seamlessly on desktop and mobile devices
- 🌐 **Serverless** - Built on Cloudflare Workers and Modal for global scale

---

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.11+ (for Modal service)
- **Cloudflare account** (free tier works)
- **Modal account** (free tier works)
- **LLM API key** - Choose your provider:
  - **OpenAI API key** (GPT-4o-mini - [get one here](https://platform.openai.com/api-keys)) - Recommended for cost efficiency
  - **Groq API key** (Llama 3.3 70B - [get one here](https://console.groq.com/)) - Recommended for speed

**Note**: All services offer free tiers suitable for development and testing.

### Environment Setup

Copy the example files and fill in your values:

```bash
# API Worker
cp apps/api-worker/.dev.vars.example apps/api-worker/.dev.vars
# Edit apps/api-worker/.dev.vars with your values

# Web App
cp apps/web/.env.example apps/web/.env.local
# Edit apps/web/.env.local with your values

# Tests
cp .env.example .env.local
# Edit .env.local with your values
```

See [docs/OPERATIONS.md](docs/OPERATIONS.md) for detailed environment variable documentation.  
See [docs/MODES.md](docs/MODES.md) for quick mode switching guide.

### One-Command Deployment

```bash
# Install dependencies
npm install

# Authenticate with Cloudflare and Modal
wrangler login
modal token new

# Setup Cloudflare resources (first time only)
./scripts/setup.sh

# Deploy everything
./scripts/deploy-all.sh
```

The `deploy-all.sh` script automatically:

1. Deploys Modal service
2. Extracts and configures the Modal URL as a secret
3. Builds shared packages
4. Deploys API worker
5. Deploys frontend
6. Optionally runs smoke tests

### Manual Deployment

For step-by-step instructions, see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

```bash
# Install dependencies
npm install

# Authenticate
wrangler login
modal token new

# Setup Cloudflare resources
./scripts/setup.sh

# Deploy Modal service
./scripts/deploy-modal.sh

# Configure secrets
cd apps/api-worker
wrangler secret put MODAL_GRADE_URL  # Paste Modal endpoint URL
wrangler secret put API_KEY          # Generate secure API key
# Choose your LLM provider (set one or both):
wrangler secret put LLM_PROVIDER     # Set to "openai" (default) or "groq"
wrangler secret put OPENAI_API_KEY   # Required if LLM_PROVIDER=openai
wrangler secret put GROQ_API_KEY      # Required if LLM_PROVIDER=groq

# Deploy API Worker
wrangler deploy

# Deploy Frontend
cd ../web
npm run build:cf
wrangler deploy

# Test
npm test
```

---

## 🏗️ Architecture

### Synchronous Processing Architecture

The system uses **synchronous processing** - all assessment is completed before returning results:

**Processing Flow:**

```
Client → API Worker → [Essay Scoring + LanguageTool + Relevance Check (parallel)] → AI Feedback (with context) → KV Storage → Client (3-20s)
```

**Key Features:**

- Synchronous processing: Results returned immediately in PUT response body (typically 3-10s, max <20s)
- Parallel processing: All services run concurrently
- Streaming: Real-time AI feedback generation via Server-Sent Events (separate endpoint)
- Optimized: Combined LLM calls, parallelized R2 operations, model caching

### Components

- **API Worker** (`apps/api-worker`) - Cloudflare Worker handling public API endpoints
- **Essay Scoring Service** (`services/modal-essay`) - FastAPI service using `KevSun/Engessay_grading_ML`
- **LanguageTool Service** (`services/modal-lt`) - FastAPI service for grammar checking
- **AI Feedback** (Multi-Provider) - Context-aware feedback using OpenAI GPT-4o-mini or Groq Llama 3.3 70B
- **Relevance Check** (Cloudflare Workers AI) - Fast embeddings-based validation
- **Storage** - R2 bucket for questions/answers/submissions, KV namespace for results
- **Frontend** (`apps/web`) - Next.js web app with inline grammar error annotations

### Performance

- **Typical Response Time**: 3-10 seconds for complete results
- **Maximum Response Time**: <20 seconds
- **Streaming**: Real-time AI feedback via Server-Sent Events
- **Scale-to-Zero**: No idle costs when not in use

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

### Project Structure

```
writeo/
├── apps/
│   ├── api-worker/          # Public API endpoints (Cloudflare Workers)
│   └── web/                  # Next.js frontend
├── services/
│   ├── modal-essay/          # Essay scoring service (FastAPI)
│   └── modal-lt/             # LanguageTool service (FastAPI)
├── packages/
│   └── shared/               # Shared TypeScript/Python types
├── docs/                     # Documentation
├── scripts/                  # Deployment and test scripts
└── tests/                    # Automated tests
```

---

## 🔌 API

**Base URL**: `https://your-api-worker.workers.dev` (configure via `API_BASE_URL` environment variable)  
**Authentication**: `Authorization: Token <api_key>` (required for all endpoints except `/health`, `/docs`, `/openapi.json`)

### Quick Example

```bash
# Set environment variables
export API_BASE="https://your-api-worker.workers.dev"  # Or use API_BASE_URL
export API_KEY="your-api-key"

# Submit for assessment (answers must be sent inline)
# Questions can be sent inline or referenced by ID
curl -X PUT "$API_BASE/text/submissions/$(uuidgen)" \
  -H "Authorization: Token $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "submission": [{
      "part": 1,
      "answers": [{
        "id": "answer-uuid",
        "question-number": 1,
        "question-id": "question-uuid",
        "question-text": "Describe your weekend. What did you do?",
        "text": "Last weekend I went to the park."
      }]
    }],
    "template": {"name": "generic", "version": 1}
  }'

# Or reference an existing question (create question first):
curl -X PUT "$API_BASE/text/questions/$(uuidgen)" \
  -H "Authorization: Token $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "Describe your weekend. What did you do?"}'

# Then submit with question reference:
curl -X PUT "$API_BASE/text/submissions/$(uuidgen)" \
  -H "Authorization: Token $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "submission": [{
      "part": 1,
      "answers": [{
        "id": "answer-uuid",
        "question-number": 1,
        "question-id": "question-uuid",
        "text": "Last weekend I went to the park."
      }]
    }],
    "template": {"name": "generic", "version": 1}
  }'
```

### Endpoints

- `PUT /text/questions/{id}` - Create or update question (optional - questions can also be sent inline with submissions)
- `PUT /text/submissions/{id}` - Submit for assessment (answers must be sent inline, returns results immediately)
- `GET /text/submissions/{id}` - Get stored results
- `GET /health` - Health check (no auth required)
- `GET /docs` - Interactive Swagger UI (no auth required)

**Interactive Documentation**: Available at `/docs` endpoint on your API server  
**Complete Specification**: [docs/SPEC.md](docs/SPEC.md) | [OpenAPI Spec](docs/openapi.yaml)

---

## 🛠️ Development

### Local Development

```bash
# API Worker
cd apps/api-worker
wrangler dev

# Frontend (in another terminal)
cd apps/web
npm run dev
```

### Code Quality

The project uses git hooks to ensure code quality:

- **Pre-commit hook**: Automatically formats code, runs linting, and type checking
- **Pre-push hook**: Runs all tests against local servers before pushing

**Git hooks must be installed manually** by running:

```bash
npm run install-hooks
# or
./scripts/install-hooks.sh
```

This is a one-time setup step that each developer should run after cloning the repository.

**Available commands:**

```bash
npm run format        # Format all code with Prettier
npm run format:check  # Check formatting without changing files
npm run lint          # Run linting across all workspaces
npm run type-check    # Run TypeScript type checking
npm test              # Run API integration tests
npm run test:e2e      # Run E2E browser tests
npm run test:all      # Run all tests (API + E2E)
```

**Bypassing hooks** (if needed):

```bash
git commit --no-verify  # Skip pre-commit hook
git push --no-verify    # Skip pre-push hook
```

### Environment Variables

**Required:**

- `MODAL_GRADE_URL` - Essay scoring Modal service endpoint
- `API_KEY` - API authentication key
- **LLM Provider** - Choose one:
  - `OPENAI_API_KEY` + `LLM_PROVIDER=openai` (default) - Cost-effective, uses GPT-4o-mini
  - `GROQ_API_KEY` + `LLM_PROVIDER=groq` - Ultra-fast, uses Llama 3.3 70B Versatile

**Optional:**

- `LLM_PROVIDER` - LLM provider: "openai" (default), "groq", "anthropic", "google"
- `AI_MODEL` - Model name (default: "gpt-4o-mini" for OpenAI, "llama-3.3-70b-versatile" for Groq)
- `MODAL_LT_URL` - LanguageTool Modal service endpoint
- `LT_LANGUAGE` - Default language code (default: `"en-GB"`)

Set via `wrangler secret put <KEY>` for Cloudflare Workers.  
See [docs/OPERATIONS.md](docs/OPERATIONS.md) for complete environment variable reference.

---

## 📚 Documentation

**Quick Links:**

- 📖 [Documentation Index](docs/README.md) - Complete documentation index
- 🚀 [Deployment Guide](docs/DEPLOYMENT.md) - Step-by-step deployment instructions
- 🏗️ [Architecture](docs/ARCHITECTURE.md) - System architecture and design
- 🔌 [API Specification](docs/SPEC.md) - Complete API reference
- 💰 [Cost Review](docs/COST_REVIEW.md) - Cost analysis and optimization
- 🧪 [Testing Guide](docs/TESTING.md) - Testing quick reference
- 📜 [Scripts Reference](docs/SCRIPTS.md) - Utility scripts documentation
- ⚖️ [Legal Compliance](docs/LEGAL_COMPLIANCE.md) - Compliance checklist
- ✅ [Status](docs/STATUS.md) - Current status and roadmap

---

## 🧪 Testing

### Automated Tests

The project includes comprehensive automated testing:

**API Tests** (`tests/api.test.ts` - Vitest):

- Full E2E workflow
- AI feedback integration
- Grammar error detection
- Confidence scores & tiers
- Context-aware tense detection
- LLM assessment integration
- Performance timing
- Input validation
- API compatibility

**E2E Tests** (`tests/e2e/*.spec.ts` - Playwright):

- Homepage and navigation
- Writing interface and submission
- Results display and feedback
- Interactive learning flow
- Draft tracking
- Error handling
- Visual design and responsive layout

**Run Tests:**

```bash
npm test              # Run API tests (Vitest)
npm run test:e2e      # Run E2E tests (Playwright)
npm run test:all      # Run all tests
npm run test:watch    # Watch mode (API tests)
npm run test:e2e:ui   # Playwright UI mode
```

**Git Hooks:**

- Pre-commit: Formats code, runs linting and type checking
- Pre-push: Runs all tests against local servers

**CI/CD:**

- GitHub Actions automatically deploys and tests on push to `main`
- See [.github/README.md](.github/README.md) for workflow details

### Test Coverage

- ✅ **API Tests** - 28 tests covering all API endpoints and workflows
- ✅ **E2E Tests** - 60 tests covering user-facing flows
- ✅ **Browser Verification** - Critical features verified
- ✅ **Manual Testing** - All critical features tested

See [docs/TESTING.md](docs/TESTING.md) for testing guide and [docs/TEST_PLAN.md](docs/TEST_PLAN.md) for test plan overview.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow existing code style and patterns
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting

---

## 💬 Support

**Getting Help:**

- 📖 **Documentation**: See [docs/README.md](docs/README.md) for complete documentation index
- 🐛 **Issues**: Check [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for troubleshooting
- 📊 **Status**: See [docs/STATUS.md](docs/STATUS.md) for current status and known issues
- 🔍 **API Reference**: Available at `/docs` endpoint on your API server

---

## 💰 Cost Optimization

The system supports two operational modes optimized for different use cases:

### 🪙 Cheap Mode (Cost-Optimized)

**Configuration:** OpenAI GPT-4o-mini + Modal scale-to-zero (30s)

- **Cost:** ~$7.60-8.50/month (100 submissions/day)
- **Performance:** 8-15s cold start, 3-10s warm
- **Best For:** Cost-conscious deployments, variable traffic

### ⚡ Turbo Mode (Performance-Optimized)

**Configuration:** Groq Llama 3.3 70B + Modal keep-warm

- **Cost:** ~$65-80/month (100 submissions/day)
- **Performance:** 2-5s first request, 1-3s warm
- **Best For:** Production deployments requiring low latency

**Cost Efficiency Features:**

- **Scale-to-Zero**: No idle costs - Workers and Modal scale to zero when not in use (Cheap Mode)
- **Free Tier Friendly**: Works on Cloudflare free tier (100k requests/day)
- **Model Caching**: Modal Volume caches model weights to reduce cold starts
- **Pay-Per-Use**: Only pay for what you use
- **Rate Limiting**: 10 submissions/minute prevents runaway costs

**Cost Breakdown Per Submission:**

**OpenAI (GPT-4o-mini)** - Recommended for cost efficiency:

- **Base submission:** ~$0.002-0.003 (2 required API calls)
- **With teacher feedback (optional):** ~$0.003-0.004
- **Average:** ~$0.0025 per submission

**Groq (Llama 3.3 70B)** - Recommended for speed:

- **Base submission:** ~$0.015-0.02 (2 required API calls)
- **With teacher feedback (optional):** ~$0.02-0.03
- **Average:** ~$0.016-0.022 per submission

**Estimated Monthly Costs**:

| Service                   | Cost                | Notes                        |
| ------------------------- | ------------------- | ---------------------------- |
| Cloudflare Workers        | $0                  | Free tier: 100k requests/day |
| Cloudflare Workers AI     | $0                  | Free tier: 10k requests/day  |
| **LLM API** (choose one): |                     |                              |
| - OpenAI (GPT-4o-mini)    | ~$0.0025/submission | Cost-effective option        |
| - Groq (Llama 3.3 70B)    | ~$0.02/submission   | Ultra-fast option            |
| R2 Storage                | ~$0.01-0.10/month   | <10GB storage                |
| KV Storage                | ~$0.01-0.05/month   | <100MB storage               |
| Modal                     | ~$0.10-1.00/month   | Pay-per-use inference        |

**Monthly Cost Examples (OpenAI):**

- **Low usage** (10 submissions/day): ~$0.75/month
- **Moderate usage** (100 submissions/day): ~$7.50/month
- **High usage** (1,000 submissions/day): ~$75/month
- **Maximum** (14,400/day, rate limited): ~$1,080/month

**Monthly Cost Examples (Groq):**

- **Low usage** (10 submissions/day): ~$6/month
- **Moderate usage** (100 submissions/day): ~$60/month
- **High usage** (1,000 submissions/day): ~$600/month
- **Maximum** (14,400/day, rate limited): ~$8,640/month

**Total Infrastructure**: ~$0.12-1.15/month on free tier (excluding OpenAI API)

**Cost Without LLM API:**

- Infrastructure only: ~$0.11-1.10/month (essentially free tier)
- No variable costs based on submission volume
- Features still available: Essay scoring, LanguageTool grammar checking, relevance checking
- Features unavailable: AI-powered feedback, teacher feedback, context-aware suggestions

**Provider Comparison:**

| Feature                 | OpenAI (GPT-4o-mini)                                   | Groq (Llama 3.3 70B)                           |
| ----------------------- | ------------------------------------------------------ | ---------------------------------------------- |
| **Cost per submission** | ~$0.0025                                               | ~$0.02 (8x more expensive)                     |
| **Speed**               | ~1-3s                                                  | ~100-500ms (ultra-fast, 3-10x faster)          |
| **Quality**             | Excellent                                              | Excellent                                      |
| **Best for**            | Cost-conscious deployments, high volume                | Speed-critical applications, low latency needs |
| **API Key**             | [Get OpenAI key](https://platform.openai.com/api-keys) | [Get Groq key](https://console.groq.com/)      |
| **Model**               | gpt-4o-mini                                            | llama-3.3-70b-versatile                        |

**Recommendation:**

- **Choose OpenAI** if cost is a primary concern or you have high submission volume
- **Choose Groq** if you need ultra-low latency and cost is less of a concern
- Both provide excellent quality feedback - the choice is primarily about cost vs speed trade-offs

**Cost Controls:**

- Rate limiting: 10 submissions/minute per IP
- Word limits: 250-500 words per essay
- Text truncation: Essays truncated to 15,000 chars for AI processing
- Token limits: Reduced max tokens to minimize costs

See [docs/COST_REVIEW.md](docs/COST_REVIEW.md) for detailed cost analysis, including costs without Groq API.

---

## 🐛 Troubleshooting

**Common Issues:**

- **Wrangler not found**: `cd apps/api-worker && npx wrangler <command>`
- **Modal fails**: Check auth (`modal token show`), ensure Python 3.11+, try `uv sync`
- **Deployment fails**: Verify KV/R2 IDs in `wrangler.toml`, check logs with `./scripts/check-logs.sh api-worker`
- **Results not appearing**: Check logs, verify secrets are set, test Modal endpoints
- **Cold starts slow**: First request after inactivity takes 8-15s (Modal warm-up), subsequent requests are fast

**Getting Help:**

- 📖 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) - Detailed troubleshooting guide
- 📖 [docs/OPERATIONS.md](docs/OPERATIONS.md) - Logging and monitoring
- 📖 [docs/STATUS.md](docs/STATUS.md) - Known limitations

---

## ✅ Status

**Production Ready** - All core features deployed and operational.

- ✅ All critical features working and verified
- ✅ Comprehensive test coverage (automated + browser verification)
- ✅ Privacy and security measures in place

**Known Limitations:**

- Modal cold starts: 8-15s (Essay Scoring), 2-5s (LanguageTool) - only affects first request after inactivity
- LLM API: Pay-per-use - OpenAI (~$0.0025/submission) or Groq (~$0.02/submission) - rate limited to 10/min

See [docs/STATUS.md](docs/STATUS.md) for detailed status information.

---

## 🗺️ Roadmap

Currently focused on stability and performance optimization. Future enhancements will be added based on user feedback.

---

## 📝 License

Licensed under the Apache License, Version 2.0 (the "License");
http://www.apache.org/licenses/LICENSE-2.0
