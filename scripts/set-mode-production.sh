#!/bin/bash
# Script to toggle between Cheap Mode and Turbo Mode in production (Cloudflare Workers)
# Usage: ./scripts/set-mode-production.sh [cheap|turbo]
#
# This script updates Cloudflare Workers secrets. For Modal service changes,
# you'll need to manually update scaledown_window in the Modal service files
# and redeploy them.

set -e

MODE="${1:-cheap}"

if [ "$MODE" != "cheap" ] && [ "$MODE" != "turbo" ]; then
  echo "Error: Mode must be 'cheap' or 'turbo'"
  echo "Usage: ./scripts/set-mode-production.sh [cheap|turbo]"
  exit 1
fi

echo "🔄 Switching production to $MODE mode..."

cd apps/api-worker

if [ "$MODE" == "cheap" ]; then
  echo "🪙 Configuring Cheap Mode (GPT-4o-mini + Modal scale-to-zero)"
  
  echo "Setting LLM_PROVIDER=openai..."
  echo "openai" | wrangler secret put LLM_PROVIDER
  
  echo ""
  echo "✅ Production configured for Cheap Mode"
  echo ""
  echo "📝 Ensure OPENAI_API_KEY is set:"
  echo "   wrangler secret put OPENAI_API_KEY"
  echo ""
  echo "📝 Modal services will automatically scale-to-zero (no redeployment needed)"
  
elif [ "$MODE" == "turbo" ]; then
  echo "⚡ Configuring Turbo Mode (Llama 3.3 70B + Modal keep-warm)"
  
  echo "Setting LLM_PROVIDER=groq..."
  echo "groq" | wrangler secret put LLM_PROVIDER
  
  echo ""
  echo "✅ Production configured for Turbo Mode"
  echo ""
  echo "📝 Ensure GROQ_API_KEY is set:"
  echo "   wrangler secret put GROQ_API_KEY"
  echo ""
  echo "📝 Redeploy Modal services with reduced scaledown_window:"
  echo "   cd ../../services/modal-essay && modal deploy app.py"
  echo "   cd ../modal-lt && modal deploy app.py"
  echo ""
  echo "   Or edit scaledown_window in app.py files before deploying:"
  echo "   - services/modal-essay/app.py: scaledown_window=2"
  echo "   - services/modal-lt/app.py: scaledown_window=2"
fi

echo ""
echo "✅ Production mode switch complete!"

