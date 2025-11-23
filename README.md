# Writeo

**The AI-Powered Essay Assessment Platform**

Writeo is a comprehensive, open-source automated essay scoring and feedback system designed for educational use. It provides multi-dimensional assessment, grammar checking, and detailed AI feedback to help students improve their writing skills.

![License](https://img.shields.io/badge/license-MIT-blue)
![Status](https://img.shields.io/badge/status-production--ready-green)
![LLM](https://img.shields.io/badge/LLM-Groq%20Llama%203.3%2070B%20%7C%20OpenAI%20GPT--4o--mini-orange)

[Live Demo](https://writeo.tre.systems) • [API Docs](https://your-api-worker.workers.dev/docs) • [Documentation](#-documentation)

<a href='https://ko-fi.com/N4N31DPNUS' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi2.png?v=6' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>

---

## 🚀 Key Features

- **🤖 AI Essay Scoring**: Multi-dimensional scoring (Task Achievement, Coherence & Cohesion, Vocabulary, Grammar) using fine-tuned RoBERTa models.
- **📝 Grammar & Style Checking**: Advanced grammar, spelling, and style analysis using LanguageTool.
- **💬 Detailed Feedback**: Context-aware, actionable feedback powered by **Groq Llama 3.3 70B** (Turbo) or **OpenAI GPT-4o-mini** (Cheap).
- **📊 CEFR Mapping**: Automatic mapping of scores to Common European Framework of Reference for Languages (A2-C2).
- **🔒 Privacy First**: "Opt-in" server storage model. By default, data never leaves the user's browser storage.
- **⚡ High Performance**: Serverless architecture with Cloudflare Workers and Modal, supporting parallel processing and scale-to-zero.
- **💰 Cost Effective**: Operational modes to balance cost and performance (from ~$8/month to ~$25/month for moderate usage).

## 📚 Documentation

| Topic                                       | Description                                                         |
| ------------------------------------------- | ------------------------------------------------------------------- |
| **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** | System design, components, data flow, and technology stack.         |
| **[DEPLOYMENT.md](docs/DEPLOYMENT.md)**     | Step-by-step guide for deploying to production.                     |
| **[COST_REVIEW.md](docs/COST_REVIEW.md)**   | Detailed cost analysis, guardrails, and pricing for OpenAI vs Groq. |
| **[MODES.md](docs/MODES.md)**               | Guide to switching between Cheap Mode and Turbo Mode.               |
| **[SERVICES.md](docs/SERVICES.md)**         | Documentation for Modal services (Essay Scoring, LanguageTool).     |
| **[SPEC.md](docs/SPEC.md)**                 | Complete API specification.                                         |
| **[TESTING.md](docs/TESTING.md)**           | Testing guide and strategies.                                       |
| **[OPERATIONS.md](docs/OPERATIONS.md)**     | Operational guide for logging, monitoring, and maintenance.         |

## 🛠️ Architecture Overview

Writeo uses a serverless edge architecture:

- **Frontend**: Next.js 15+ (App Router) on Cloudflare Pages.
- **API**: Cloudflare Workers (Hono) for orchestration.
- **ML Services**: Modal for GPU-accelerated essay scoring and LanguageTool.
- **AI Feedback**: Groq (Llama 3.3 70B) or OpenAI (GPT-4o-mini).
- **Storage**: Cloudflare R2 (Object Storage) and KV (Key-Value) - _Opt-in only_.

## ⚡ Quick Start

### Prerequisites

- Node.js 18+
- Python 3.11+
- [Modal](https://modal.com) account and CLI
- [Cloudflare](https://cloudflare.com) account and Wrangler CLI

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/writeo.git
cd writeo

# Install dependencies
npm install

# Install Python dependencies for Modal services
cd services/modal-essay && pip install -e .
cd ../modal-lt && pip install -e .
```

### 2. Configuration

Create `.dev.vars` in `apps/api-worker/`:

```bash
# apps/api-worker/.dev.vars
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_key
# OR
# LLM_PROVIDER=openai
# OPENAI_API_KEY=your_openai_key

# Optional: Cloudflare credentials if needed locally
```

### 3. Deploy Modal Services

**Option A: One-Command Deployment (Recommended)**

```bash
# Setup Cloudflare resources and deploy everything
./scripts/setup.sh
./scripts/deploy-all.sh
```

**Option B: Manual Deployment**

Deploy the ML backend services to Modal:

```bash
# Deploy Essay Scoring Service
cd services/modal-essay
modal deploy app.py

# Deploy LanguageTool Service
cd ../services/modal-lt
modal deploy app.py
```

### 4. Run Locally

Start the development server:

```bash
# From root
npm run dev
```

This starts:

- Frontend at `http://localhost:3000`
- API Worker at `http://localhost:8787`

## 🎛️ Operational Modes

Writeo can run in two modes to optimize for cost or speed:

1.  **Turbo Mode (Recommended)**: Uses **Groq Llama 3.3 70B Versatile** for ultra-fast feedback (~$0.006/submission).
2.  **Cheap Mode**: Uses **OpenAI GPT-4o-mini** for lowest cost (~$0.0025/submission).

Switch modes easily:

```bash
./scripts/set-mode.sh turbo
# OR
./scripts/set-mode.sh cheap
```

See **[MODES.md](docs/MODES.md)** and **[COST_REVIEW.md](docs/COST_REVIEW.md)** for details.

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
npm test

# Run specific test file
npx vitest tests/api.test.ts
```

See **[TESTING.md](docs/TESTING.md)** for more info.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
