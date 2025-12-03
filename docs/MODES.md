# Operational Modes Guide

Quick guide for switching between Cheap Mode and Turbo Mode.

## Quick Reference

| Mode         | LLM Provider                 | Modal Scaling                       | Cost (100/day)    | Latency                 |
| ------------ | ---------------------------- | ----------------------------------- | ----------------- | ----------------------- |
| **🪙 Cheap** | OpenAI GPT-4o-mini           | Scale-to-zero (Essay: 30s, LT: 60s) | ~$7.60-8.50/month | 10-15s cold, 3-10s warm |
| **⚡ Turbo** | Groq Llama 3.3 70B Versatile | Keep warm (2s)                      | ~$25-40/month     | 2-5s first, 1-3s warm   |

## Local Development

### Quick Switch

```bash
# Switch to Cheap Mode
./scripts/set-mode.sh cheap

# Switch to Turbo Mode
./scripts/set-mode.sh turbo
```

The script updates `apps/api-worker/.dev.vars` with the correct `LLM_PROVIDER`.

**Cheap Mode:**

- Sets `LLM_PROVIDER=openai`
- Modal services use default scaledown windows (Essay: 30s, LanguageTool: 60s - scale-to-zero)
- No Modal redeployment needed

**Turbo Mode:**

- Sets `LLM_PROVIDER=groq`
- You need to manually update Modal services to `scaledown_window=2` and redeploy

### Manual Configuration

**Cheap Mode:**

```bash
# In apps/api-worker/.dev.vars
LLM_PROVIDER=openai
OPENAI_API_KEY=your-openai-key
```

**Turbo Mode:**

```bash
# In apps/api-worker/.dev.vars
LLM_PROVIDER=groq
GROQ_API_KEY=your-groq-key

# Update Modal services (Essay: scaledown_window=30 → 2, LanguageTool: scaledown_window=60 → 2)
# Then redeploy: cd services/modal-essay && modal deploy app.py
# And: cd services/modal-lt && modal deploy app.py
```

## Production

```bash
# Switch to Cheap Mode
./scripts/set-mode-production.sh cheap

# Switch to Turbo Mode
./scripts/set-mode-production.sh turbo
```

Or manually:

```bash
cd apps/api-worker
echo "openai" | wrangler secret put LLM_PROVIDER  # Cheap Mode
echo "groq" | wrangler secret put LLM_PROVIDER     # Turbo Mode
```

## When to Use Each Mode

**Cheap Mode:** Cost-conscious, variable traffic, development/testing  
**Turbo Mode:** Low latency critical, production with steady traffic

See [OPERATIONS.md](OPERATIONS.md) and [COST_REVIEW.md](COST_REVIEW.md) for detailed cost and performance information.
