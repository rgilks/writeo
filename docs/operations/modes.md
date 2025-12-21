# Operational Modes Guide

Writeo allows you to balance cost, performance, and feedback quality by adjusting three key levers:

1.  **LLM Provider**: Controls the feedback text generation speed and cost.
2.  **Assessors**: Controls which AI models analyze the text.
3.  **Service Scaling**: Controls the latency of cold starts.

## Modes Overview

### 🪙 Economy Mode (Development)

Best for development, testing, and low-budget deployments.

- **LLM**: OpenAI GPT-4o-mini (Cost-efficient)
- **Assessors**: Any configuration (typically minimal)
- **Scaling**: Scale-to-zero (30s keep-warm)
- **Cost**: ~$7-9/month (100 submissions/day)
- **Latency**: 10-15s cold start, 3-10s warm

### 🚀 Production Mode (Turbo)

Best for live user traffic where low latency is critical.

- **LLM**: Groq Llama 3.3 70B (Extremely fast)
- **Assessors**: Full Suite (DeBERTa, GEC-SEQ2SEQ, GEC-GECTOR)
- **Scaling**: Keep-warm (2s scaledown window)
- **Cost**: ~$25-40/month
- **Latency**: 2-5s first request, 1-3s warm

## Configuration Levers

### 1. LLM Provider (Text Feedback)

Controls which model generates the written feedback.

| Provider   | Setting (`LLM_PROVIDER`) | Pros                    | Cons                        |
| :--------- | :----------------------- | :---------------------- | :-------------------------- |
| **OpenAI** | `openai`                 | Cheapest, very reliable | Slower (2-5s generation)    |
| **Groq**   | `groq`                   | Instant (sub-second)    | More expensive/rate-limited |

**To Switch:**

```bash
# Local
./scripts/set-mode.sh cheap   # Sets openai
./scripts/set-mode.sh turbo   # Sets groq

# Production
./scripts/set-mode-production.sh cheap
./scripts/set-mode-production.sh turbo
```

### 2. Active Assessors (Scoring & Grammar)

Controls which models run to analyze the text. Configured in `apps/api-worker/src/config/assessors.json`.

**Recommended Production Config:**

```json
{
  "scoring": {
    "essay": false,
    "feedback": false,
    "deberta": true // Primary Dimensional Scoring
  },
  "grammar": {
    "languageTool": true,
    "gecSeq2seq": true, // High precision
    "gecGector": true // Low latency
  }
}
```

**Minimal / Local Config:**
Disable `deberta` and `gecSeq2seq` to save on GPU usage during testing, unless specifically testing those components.

### 3. Service Scaling (Latency)

Controls how long Modal services stay "warm" (active on GPU) after a request.

**Scale-to-Zero (Economy Default):**

- **Setting**: `SCALEDOWN_WINDOW_SECONDS = 30` (in `app.py` or `main.py`)
- **Pros**: You only pay for seconds of usage.
- **Cons**: "Cold starts" take 5-15s while models load.

**Keep-Warm (Production):**

- **Setting**: `SCALEDOWN_WINDOW_SECONDS = 2` (Logic: Aggressively keeps warm if requests are frequent, but technically this value is the _idle_ timeout. For true keep-warm, you'd use `keep_warm=1` in Modal, but we use a short timeout + frequent traffic strategy for cost balance, or manual `keep_warm` during events).
- _Correction_: To _keep_ it warm, you typically increase the window or use `min_containers`. However, for "Turbo" mode in our scripts, we often toggle settings to ensure readiness.
- **Current Script Logic**: The scripts suggest setting `scaledown_window=2` to _reduce_ broken billing overlaps or quickly free resources, **BUT** for true production speed, you typically want a _longer_ window or a `min_containers=1`. _Note: The previous documentation suggested 2s, which effectively makes it scale down instantly. For production, you usually want 60-300s._

**Update:** The current best practice for "Turbo" in this project is to ensure services are warm. If you are experiencing cold starts, consider increasing the keep-warm time in `app.py`.

## Quick Switch Scripts

Use the helper scripts to switch `LLM_PROVIDER` and view instructions for Modal services.

```bash
# Local Development
./scripts/set-mode.sh cheap
./scripts/set-mode.sh turbo

# Production
./scripts/set-mode-production.sh cheap
./scripts/set-mode-production.sh turbo
```

See [Cost Analysis](cost.md) for detailed pricing breakdown.
See [Deployment Guide](deployment.md) for deployment instructions.
