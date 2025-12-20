#!/bin/bash
# Script to toggle between Cheap Mode and Turbo Mode
# Usage: ./scripts/set-mode.sh [cheap|turbo]

set -e

MODE="${1:-cheap}"

if [ "$MODE" != "cheap" ] && [ "$MODE" != "turbo" ]; then
  echo "Error: Mode must be 'cheap' or 'turbo'"
  echo "Usage: ./scripts/set-mode.sh [cheap|turbo]"
  exit 1
fi

echo "🔄 Switching to $MODE mode..."

DEV_VARS_FILE="apps/api-worker/.dev.vars"

if [ ! -f "$DEV_VARS_FILE" ]; then
  echo "Error: $DEV_VARS_FILE not found"
  exit 1
fi

if [ "$MODE" == "cheap" ]; then
  echo "🪙 Configuring Cheap Mode (GPT-4o-mini + Modal scale-to-zero)"
  
  # Update LLM_PROVIDER
  if grep -q "^LLM_PROVIDER=" "$DEV_VARS_FILE"; then
    sed -i.bak 's/^LLM_PROVIDER=.*/LLM_PROVIDER=openai/' "$DEV_VARS_FILE"
  else
    echo "LLM_PROVIDER=openai" >> "$DEV_VARS_FILE"
  fi
  
  echo "✅ Set LLM_PROVIDER=openai"
  echo ""
  echo "📝 Ensure OPENAI_API_KEY is set in $DEV_VARS_FILE"
  echo "📝 Modal services use scaledown_window=30 (scale-to-zero) by default."
  echo "   No redeployment needed if you haven't changed defaults."
  
elif [ "$MODE" == "turbo" ]; then
  echo "⚡ Configuring Turbo Mode (Llama 3.3 70B + Modal keep-warm)"
  
  # Update LLM_PROVIDER
  if grep -q "^LLM_PROVIDER=" "$DEV_VARS_FILE"; then
    sed -i.bak 's/^LLM_PROVIDER=.*/LLM_PROVIDER=groq/' "$DEV_VARS_FILE"
  else
    echo "LLM_PROVIDER=groq" >> "$DEV_VARS_FILE"
  fi
  
  echo "✅ Set LLM_PROVIDER=groq"
  echo ""
  echo "📝 Ensure GROQ_API_KEY is set in $DEV_VARS_FILE"
  echo "📝 For Turbo Mode (Low Latency), update Modal services:"
  echo "   1. Edit 'app.py' or 'main.py' in these directories:"
  echo "      - services/modal-deberta (Primary)"
  echo "      - services/modal-gec (Grammar)"
  echo "      - services/modal-gector (Fast Grammar)"
  echo "      - services/modal-corpus (Secondary)"
  echo "   2. Change: scaledown_window=30  ->  scaledown_window=2"
  echo "   3. Redeploy each modified service:"
  echo "      cd services/modal-deberta && modal deploy app.py"
  echo "      cd services/modal-gec && modal deploy main.py"
  echo "      (repeat for others)"
fi

rm -f "$DEV_VARS_FILE.bak"

echo ""
echo "✅ Mode switch complete!"
grep "^LLM_PROVIDER=" "$DEV_VARS_FILE" || echo "LLM_PROVIDER not set"
