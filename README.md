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
- 🧠 **Groq** - Ultra-fast AI feedback (Llama 3.3 70B)
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
- 🎯 **AI-Powered Feedback** - Context-aware feedback using Groq (Llama 3.3 70B Versatile)
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
- **Groq API key** (for AI feedback - [get one here](https://console.groq.com/))

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
wrangler secret put GROQ_API_KEY     # Your Groq API key

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
- **AI Feedback** (Groq) - Context-aware feedback using Llama 3.3 70B Versatile
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
- `GROQ_API_KEY` - Groq API key for AI feedback

**Optional:**

- `MODAL_LT_URL` - LanguageTool Modal service endpoint
- `LT_LANGUAGE` - Default language code (default: `"en-GB"`)
- `AI_MODEL` - Groq model (default: `"llama-3.3-70b-versatile"`)

Set via `wrangler secret put <KEY>` for Cloudflare Workers.  
See [docs/OPERATIONS.md](docs/OPERATIONS.md) for complete environment variable reference.

---

## 📚 Documentation

### Getting Started

- **[docs/README.md](docs/README.md)** - Documentation index and quick links
- **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Step-by-step deployment guide
- **[docs/STATUS.md](docs/STATUS.md)** - Current status and roadmap

### Core Documentation

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture, components, and data flow
- **[docs/SPEC.md](docs/SPEC.md)** - Complete API specification with request/response examples
- **[docs/OPERATIONS.md](docs/OPERATIONS.md)** - Operations guide: environment variables, logging, performance
- **[docs/TEST_PLAN.md](docs/TEST_PLAN.md)** - Test plan with automated tests and manual verification
- **[tests/README.md](tests/README.md)** - Test suite documentation and quick reference

### API Documentation

- **[docs/openapi.yaml](docs/openapi.yaml)** - OpenAPI 3.0 specification

### Legal & Compliance

- **[docs/LEGAL_COMPLIANCE.md](docs/LEGAL_COMPLIANCE.md)** - Legal compliance checklist and requirements

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

See [docs/TEST_PLAN.md](docs/TEST_PLAN.md) for test plan and [tests/README.md](tests/README.md) for test suite documentation.

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

The system is designed for cost efficiency with scale-to-zero architecture:

- **Scale-to-Zero**: No idle costs - Workers and Modal scale to zero when not in use
- **Free Tier Friendly**: Works on Cloudflare free tier (100k requests/day)
- **Model Caching**: Modal Volume caches model weights to reduce cold starts
- **Pay-Per-Use**: Only pay for what you use

**Estimated Monthly Costs (Free Tier)**:

| Service               | Cost              | Notes                        |
| --------------------- | ----------------- | ---------------------------- |
| Cloudflare Workers    | $0                | Free tier: 100k requests/day |
| Cloudflare Workers AI | $0                | Free tier: 10k requests/day  |
| Groq API              | ~$0.01/request    | LLM feedback (pay-per-use)   |
| R2 Storage            | ~$0.01-0.10/month | <10GB storage                |
| KV Storage            | ~$0.01-0.05/month | <100MB storage               |
| Modal                 | ~$0.10-1.00/month | Pay-per-use inference        |

**Total**: ~$0.12-1.15/month on free tier (excluding Groq API usage)

---

## 🐛 Troubleshooting

**Common Issues:**

- **Wrangler not found**: `cd apps/api-worker && npx wrangler <command>`
- **Modal fails**: Check auth (`modal token show`), ensure Python 3.11+, try `uv sync`
- **Deployment fails**: Verify KV/R2 IDs in `wrangler.toml`, check logs with `./scripts/check-logs.sh api-worker`
- **Results not appearing**: Check logs, verify secrets are set, test Modal endpoints
- **Cold starts slow**: First request after inactivity takes 8-15s (Modal warm-up), subsequent requests are fast

**Getting Help:**

- See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed troubleshooting
- See [docs/OPERATIONS.md](docs/OPERATIONS.md) for logging and monitoring
- Check [docs/STATUS.md](docs/STATUS.md) for known limitations

---

## ✅ Status

**Production Ready** - All core features deployed and operational.

**Current Status:**

- ✅ All critical features working and verified
- ✅ Draft tracking and navigation implemented
- ✅ Comprehensive test coverage (automated + browser verification)
- ✅ Privacy and security measures in place

**Known Limitations:**

- Modal cold starts: 8-15s (Essay Scoring), 2-5s (LanguageTool) - only affects first request after inactivity
- Groq API: Pay-per-use (~$0.01 per request) - no free tier

For detailed status information, see [docs/STATUS.md](docs/STATUS.md).

---

## 🗺️ Roadmap

**Future Enhancements:**

_No planned enhancements at this time._

**Not Currently Planned:**

- Translation features - Not implemented (documented but not planned)

---

## 📝 License

Licensed under the Apache License, Version 2.0 (the "License");
http://www.apache.org/licenses/LICENSE-2.0
