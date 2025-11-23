#!/bin/bash
# Helper script to manage API keys in Cloudflare KV

# Configuration
# This ID matches the WRITEO_RESULTS KV namespace in apps/api-worker/wrangler.toml
KV_ID="7a0dc51ef6884c81829ac2ff8e9261a9"

if [ -z "$1" ]; then
  echo "Usage: ./manage-api-keys.sh <action> [args]"
  echo "Actions:"
  echo "  create <owner_name>  - Create a new API key for a user"
  echo "  revoke <api_key>     - Revoke an API key"
  echo "  get <api_key>        - Check who owns a key"
  exit 1
fi

ACTION=$1

if [ "$ACTION" == "create" ]; then
  OWNER=$2
  if [ -z "$OWNER" ]; then
    echo "Error: Owner name required"
    exit 1
  fi
  
  # Generate a random key (32 chars hex)
  # Using openssl if available, otherwise fallback
  if command -v openssl &> /dev/null; then
    API_KEY=$(openssl rand -hex 16)
  else
    API_KEY=$(date +%s | sha256sum | base64 | head -c 32)
  fi
  
  # Store in KV: apikey:<token> -> {"owner": "name", "created": "date"}
  CREATED_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  VALUE="{\"owner\": \"$OWNER\", \"created\": \"$CREATED_DATE\"}"
  
  echo "Creating API key for '$OWNER'..."
  # Use the wrangler from the root node_modules or api-worker
  npx wrangler kv:key put "apikey:$API_KEY" "$VALUE" --namespace-id "$KV_ID"
  
  echo ""
  echo "✅ API Key created successfully!"
  echo "Owner: $OWNER"
  echo "Token: $API_KEY"
  echo "Header: Authorization: Token $API_KEY"
  
elif [ "$ACTION" == "revoke" ]; then
  API_KEY=$2
  if [ -z "$API_KEY" ]; then
    echo "Error: API key required"
    exit 1
  fi
  
  echo "Revoking API key..."
  npx wrangler kv:key delete "apikey:$API_KEY" --namespace-id "$KV_ID"
  echo "✅ API Key revoked."

elif [ "$ACTION" == "get" ]; then
  API_KEY=$2
  if [ -z "$API_KEY" ]; then
    echo "Error: API key required"
    exit 1
  fi
  
  echo "Fetching key info..."
  npx wrangler kv:key get "apikey:$API_KEY" --namespace-id "$KV_ID"

else
  echo "Unknown action: $ACTION"
  exit 1
fi

