# Writeo

**The AI-Powered Essay Assessment Platform**

Writeo is a comprehensive, open-source automated essay scoring and feedback system designed for educational use. It provides multi-dimensional assessment, grammar checking, and detailed AI feedback to help students improve their writing skills.

![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Status](https://img.shields.io/badge/status-production--ready-green)
![LLM](https://img.shields.io/badge/LLM-Groq%20or%20OpenAI-orange)

[Live Demo](https://writeo.tre.systems)

<a href='https://ko-fi.com/N4N31DPNUS' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi2.png?v=6' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>

---

## 🚀 Key Features

- **🤖 AI Essay Scoring**: Multi-dimensional scoring (Task Achievement, Coherence & Cohesion, Vocabulary, Grammar) using fine-tuned RoBERTa models.
- **✨ Smart Grammar Correction**: Dual GEC services running in parallel:
  - **Seq2Seq (GEC-SEQ2SEQ)**: High-quality corrections using Flan-T5 (~12-16s)
  - **GECToR (GEC-GECTOR)**: Fast token-tagging approach (~1-2s, 10x faster)
- **📝 Grammar & Style Checking**: Advanced grammar, spelling, and style analysis using LanguageTool.
- **💬 Detailed Feedback**: Context-aware, actionable feedback powered by **Groq Llama 3.3 70B** (Turbo) or **OpenAI GPT-4o-mini** (Cheap).
- **📊 CEFR Mapping**: Automatic mapping of scores to Common European Framework of Reference for Languages (A2-C2).
- **🔒 Privacy First**: "Opt-in" server storage model. By default, data never leaves the user's browser storage.
- **⚡ High Performance**: Serverless architecture with Cloudflare Workers and Modal, supporting parallel processing and scale-to-zero.
- **💰 Cost Effective**: Operational modes to balance cost and performance. See [Cost Analysis](docs/operations/cost.md).
- **📱 Progressive Web App**: Installable PWA with offline support.

## 📚 Documentation

**📖 [Full Documentation Index](docs/README.md)** - Complete guide to Architecture, Operations, and Models.

### Key Resources

| Topic            | Description                   | Link                                                              |
| ---------------- | ----------------------------- | ----------------------------------------------------------------- |
| **Architecture** | System design and API details | [docs/architecture/overview.md](docs/architecture/overview.md)    |
| **Deployment**   | Production deployment guide   | [docs/operations/deployment.md](docs/operations/deployment.md)    |
| **Operations**   | Monitoring, Logging, Cost     | [docs/operations/monitoring.md](docs/operations/monitoring.md)    |
| **API Docs**     | Interactive Swagger UI        | [Live Docs](https://writeo-api-worker.rob-gilks.workers.dev/docs) |

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
- [uv](https://github.com/astral-sh/uv) (for Python package management)
- [Modal](https://modal.com) account and CLI
- [Cloudflare](https://cloudflare.com) account and Wrangler CLI

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/rgilks/writeo.git
cd writeo

# Install dependencies
npm install

# Install Python dependencies for Modal services
cd services/modal-deberta && uv pip install -e .
cd ../modal-feedback && uv pip install -e .
cd ../modal-gec && uv pip install -e .
cd ../modal-lt && uv pip install -e .
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
# Deploy Essay Scoring & Feedback Services
cd services/modal-deberta
modal deploy app.py

cd ../modal-feedback
modal deploy app.py

# Deploy GEC Services
cd ../modal-gec
modal deploy main.py

cd ../modal-gector
modal deploy main.py

# Deploy LanguageTool
cd ../modal-lt
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

### 5. Development Workflow

**Install git hooks** (recommended):

```bash
npm run install-hooks
```

Hooks provide:

- Pre-commit: Auto-formatting, linting, type checking
- Pre-push: Full test suite (use `QUICK_PUSH=true git push` to skip E2E tests)

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

See **[Operational Modes](docs/operations/modes.md)** and **[Cost Analysis](docs/operations/cost.md)** for details.

## 🧪 Testing

Run the test suite:

```bash
# Run all tests (unit + E2E)
npm run test:all

# Run unit tests only
npm run test:unit

# Run E2E tests (automatically starts test server)
npm run test:e2e

# Run specific test file
npx vitest tests/api.test.ts
```

See **[Testing Guide](docs/operations/testing.md)** for more info.

## 🗺️ Roadmap & Status

See **[Status](docs/reference/status.md)** for current production status, completed features, known limitations, and roadmap.

## 💬 Support

- **Discord**: [Join our Discord server](https://discord.gg/YxuFAXWuzw)
- **GitHub Issues**: [Report bugs or request features](https://github.com/rgilks/writeo/issues)
- **Documentation**: [Full documentation index](docs/README.md)

## ⚖️ Licensing & Third-Party Dependencies

### Project License

This project is open-source and **non-commercial**. Any donations received are solely for hosting and infrastructure costs. We do not sell access or services.

### Third-Party Dependencies

| Component        | License         | Notes                                                                      |
| ---------------- | --------------- | -------------------------------------------------------------------------- |
| **Next.js**      | MIT             | ✅ Fully permissive                                                        |
| **React**        | MIT             | ✅ Fully permissive                                                        |
| **Hono**         | MIT             | ✅ Fully permissive                                                        |
| **LanguageTool** | LGPL 2.1        | ✅ Permissive for SaaS - we use it as a service, not distributing binaries |
| **Transformers** | Apache 2.0      | ✅ Fully permissive                                                        |
| **Modal**        | Commercial SaaS | ✅ Standard usage terms                                                    |

### ML Models

| Model              | Base License | Notes                       |
| ------------------ | ------------ | --------------------------- |
| **DeBERTa-v3**     | MIT          | ✅ Microsoft open-source    |
| **Flan-T5** (GEC)  | Apache 2.0   | ✅ Google open-source       |
| **GECToR-RoBERTa** | Apache 2.0   | ✅ Publicly available model |
| **RoBERTa**        | MIT          | ✅ Meta open-source         |

### Training Datasets

| Dataset                 | License Status                | Our Usage                                            |
| ----------------------- | ----------------------------- | ---------------------------------------------------- |
| **IELTS-WT2-LLaMa3-1k** | Unknown (HuggingFace)         | Primary dimensional training                         |
| **DREsS**               | Academic (consent form)       | Primary dimensional training                         |
| **Write & Improve**     | Non-commercial, research only | Calibration & validation only (not primary training) |

> **Note**: W&I Corpus is used **only for CEFR score calibration and validation**, not as primary training data. This aligns with research/educational use terms. We are happy to source alternative CEFR-labeled datasets if any licensing concerns arise.

For detailed dataset licensing information, see [docs/models/datasets.md](docs/models/datasets.md).

### LLM Providers

- **Groq**: Used via their API under standard terms of service
- **OpenAI**: Used via their API under standard terms of service

No LLM model weights are distributed with this project.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
