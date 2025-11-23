#!/bin/bash
# Reset rate limits in Cloudflare KV
# Only deletes rate limit keys, preserves all other data

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration - get from wrangler.toml
KV_NAMESPACE_ID="7a0dc51ef6884c81829ac2ff8e9261a9"  # Production namespace ID from wrangler.toml
ACCOUNT_ID="571f130502618993d848f58d27ae288d"

echo "=== Reset Rate Limits ==="
echo ""

# Check for required tools
if ! command -v wrangler &> /dev/null; then
    echo -e "${RED}✘${NC} Wrangler not found. Please install it:"
    echo "  npm install -g wrangler"
    exit 1
fi

if ! command -v curl &> /dev/null; then
    echo -e "${RED}✘${NC} curl not found. Please install it."
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo -e "${RED}✘${NC} jq not found. Please install it:"
    echo "  brew install jq  # macOS"
    echo "  apt-get install jq  # Linux"
    exit 1
fi

# Get Cloudflare API token from Wrangler config
WRANGLER_CONFIG="${HOME}/.wrangler/config/default.toml"
if [ -f "$WRANGLER_CONFIG" ]; then
    CLOUDFLARE_API_TOKEN=$(grep 'oauth_token' "$WRANGLER_CONFIG" | sed -E 's/.*"([^"]+)".*/\1/' | head -n 1)
fi

# Check if user is logged in to Wrangler
if [ -z "$CLOUDFLARE_API_TOKEN" ] || ! wrangler whoami > /dev/null 2>&1; then
    echo -e "${RED}✘${NC} Not logged in to Wrangler or token not found"
    echo ""
    echo "Please run: wrangler login"
    exit 1
fi

echo -e "${GREEN}✓${NC} Wrangler authenticated"
echo ""

# Function to list all keys with a prefix
list_keys_with_prefix() {
    local prefix=$1
    local cursor=""
    local all_keys=()
    
    while true; do
        local url="https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/storage/kv/namespaces/${KV_NAMESPACE_ID}/keys"
        local params="?prefix=${prefix}&limit=1000"
        
        if [ -n "$cursor" ]; then
            params="${params}&cursor=${cursor}"
        fi
        
        local response=$(curl -s -X GET \
            "${url}${params}" \
            -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" \
            -H "Content-Type: application/json")
        
        if echo "$response" | jq -e '.success == false' > /dev/null 2>&1; then
            echo -e "${RED}✘${NC} Error listing keys: $(echo "$response" | jq -r '.errors[0].message // "Unknown error"')" >&2
            return 1
        fi
        
        local keys=$(echo "$response" | jq -r '.result[]?.name // empty')
        if [ -n "$keys" ]; then
            while IFS= read -r key; do
                if [ -n "$key" ]; then
                    all_keys+=("$key")
                fi
            done <<< "$keys"
        fi
        
        cursor=$(echo "$response" | jq -r '.result_info.cursor // empty')
        if [ -z "$cursor" ] || [ "$cursor" = "null" ]; then
            break
        fi
    done
    
    printf '%s\n' "${all_keys[@]}"
}

# Function to delete a key
delete_key() {
    local key=$1
    local response=$(curl -s -X DELETE \
        "https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/storage/kv/namespaces/${KV_NAMESPACE_ID}/values/${key}" \
        -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" \
        -H "Content-Type: application/json")
    
    if echo "$response" | jq -e '.success == false' > /dev/null 2>&1; then
        return 1
    fi
    return 0
}

echo "Finding rate limit keys..."
echo ""

# List all rate limit keys
RATE_LIMIT_KEYS=()
while IFS= read -r key; do
    if [ -n "$key" ]; then
        RATE_LIMIT_KEYS+=("$key")
    fi
done < <(list_keys_with_prefix "rate_limit:")

if [ ${#RATE_LIMIT_KEYS[@]} -eq 0 ]; then
    echo -e "${GREEN}✓${NC} No rate limit keys found. Nothing to reset."
    exit 0
fi

echo -e "${YELLOW}Found ${#RATE_LIMIT_KEYS[@]} rate limit key(s):${NC}"
for key in "${RATE_LIMIT_KEYS[@]}"; do
    echo "  - $key"
done
echo ""

# Confirm before proceeding
read -p "Delete these rate limit keys? (type 'yes' to confirm): " confirmation

if [ "$confirmation" != "yes" ]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Deleting rate limit keys..."
echo ""

# Delete all rate limit keys
deleted=0
failed=0

for key in "${RATE_LIMIT_KEYS[@]}"; do
    if delete_key "$key"; then
        deleted=$((deleted + 1))
        echo -e "  ${GREEN}✓${NC} Deleted: $key"
    else
        failed=$((failed + 1))
        echo -e "  ${RED}✘${NC} Failed: $key"
    fi
done

echo ""
echo -e "${GREEN}✓${NC} Rate limits reset complete!"
echo "  Deleted: $deleted"
if [ $failed -gt 0 ]; then
    echo -e "  ${YELLOW}Failed: $failed${NC}"
fi
echo ""
echo "Rate limits will reset on the next request."

