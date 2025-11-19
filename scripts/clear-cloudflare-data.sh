#!/bin/bash
# Clear all data from Cloudflare R2 and KV storage
# This script deletes all objects from R2 buckets and all keys from KV namespaces

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
# Note: Account ID is project-specific. If using a different Cloudflare account,
# update this value (you can find it with: wrangler whoami)
ACCOUNT_ID="571f130502618993d848f58d27ae288d"
R2_BUCKET="writeo-data"
KV_NAMESPACE_ID="17677e5237b248b3b94ae3b7c9468933"

echo "=== Clear Cloudflare Data ==="
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

# Get Cloudflare API token (from env var or Wrangler's stored OAuth token)
if [ -z "$CLOUDFLARE_API_TOKEN" ]; then
    # Try to read OAuth token from Wrangler config
    WRANGLER_CONFIG="${HOME}/.wrangler/config/default.toml"
    if [ -f "$WRANGLER_CONFIG" ]; then
        # Extract OAuth token from TOML file (extract value between quotes)
        OAUTH_TOKEN=$(grep '^oauth_token' "$WRANGLER_CONFIG" | sed -E 's/.*"([^"]+)".*/\1/' | head -n 1)
        
        if [ -n "$OAUTH_TOKEN" ] && [ "$OAUTH_TOKEN" != "oauth_token" ]; then
            CLOUDFLARE_API_TOKEN="$OAUTH_TOKEN"
            echo -e "${GREEN}✓${NC} Using Wrangler OAuth token (from wrangler login)"
        fi
    fi
fi

# If still no token, check if user is logged in to Wrangler
if [ -z "$CLOUDFLARE_API_TOKEN" ]; then
    echo -e "${YELLOW}⚠${NC}  No API token found in environment or Wrangler config."
    echo ""
    echo "Checking Wrangler authentication..."
    if wrangler whoami > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} You're logged in to Wrangler"
        echo "Attempting to extract token from Wrangler config..."
        
        # Try to extract token again with alternative method
        WRANGLER_CONFIG="${HOME}/.wrangler/config/default.toml"
        if [ -f "$WRANGLER_CONFIG" ]; then
            OAUTH_TOKEN=$(grep 'oauth_token' "$WRANGLER_CONFIG" | sed -E 's/.*"([^"]+)".*/\1/' | head -n 1)
            if [ -n "$OAUTH_TOKEN" ] && [ "$OAUTH_TOKEN" != "oauth_token" ]; then
                CLOUDFLARE_API_TOKEN="$OAUTH_TOKEN"
                echo -e "${GREEN}✓${NC} Found OAuth token"
            fi
        fi
    else
        echo -e "${RED}✘${NC} Not logged in to Wrangler"
        echo ""
        echo "Please run: wrangler login"
        echo "Or set CLOUDFLARE_API_TOKEN environment variable"
        exit 1
    fi
fi

if [ -z "$CLOUDFLARE_API_TOKEN" ]; then
    echo -e "${RED}✘${NC} Could not find Cloudflare API token"
    echo ""
    echo "Options:"
    echo "1. Run 'wrangler login' to authenticate"
    echo "2. Or set CLOUDFLARE_API_TOKEN environment variable:"
    echo "   export CLOUDFLARE_API_TOKEN='your-token-here'"
    exit 1
fi

echo -e "${GREEN}✓${NC} Tools and credentials ready"
echo ""

# Function to delete all R2 objects
clear_r2_bucket() {
    local bucket=$1
    echo "Clearing R2 bucket: ${bucket}"
    
    # List all objects with pagination
    echo "  Listing objects..."
    local cursor=""
    local total_deleted=0
    local page_count=0
    
    while true; do
        local url="https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/r2/buckets/${bucket}/objects"
        local params=""
        
        if [ -n "$cursor" ]; then
            params="?cursor=${cursor}"
        fi
        
        local response=$(curl -s -X GET "${url}${params}" \
            -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" \
            -H "Content-Type: application/json")
        
        # Check for errors
        if echo "$response" | jq -e '.success == false' > /dev/null 2>&1; then
            local error=$(echo "$response" | jq -r '.errors[0].message // "Unknown error"')
            echo -e "  ${RED}✘${NC} Error listing objects: $error"
            return 1
        fi
        
        # Extract objects array (result is an array directly, not result.objects)
        local objects_json=$(echo "$response" | jq -r '.result // []')
        local truncated=$(echo "$response" | jq -r '.result_info.is_truncated // false')
        
        # Check if we have any objects
        local object_count=$(echo "$objects_json" | jq 'length')
        
        if [ "$object_count" -eq 0 ]; then
            if [ "$page_count" -eq 0 ]; then
                echo -e "  ${GREEN}✓${NC} Bucket is already empty"
            fi
            break
        fi
        
        page_count=$((page_count + 1))
        echo "  Processing page ${page_count} (${object_count} objects)..."
        
        # Delete each object
        local i=0
        while [ $i -lt "$object_count" ]; do
            local key=$(echo "$objects_json" | jq -r ".[$i].key")
            if [ -n "$key" ] && [ "$key" != "null" ]; then
                echo "    Deleting: ${key}"
                local delete_response=$(curl -s -X DELETE \
                    "https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/r2/buckets/${bucket}/objects/${key}" \
                    -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}")
                
                if echo "$delete_response" | jq -e '.success == false' > /dev/null 2>&1; then
                    local error=$(echo "$delete_response" | jq -r '.errors[0].message // "Unknown error"')
                    echo -e "      ${RED}✘${NC} Failed to delete ${key}: $error"
                else
                    total_deleted=$((total_deleted + 1))
                fi
            fi
            i=$((i + 1))
        done
        
        # Check if there are more pages
        if [ "$truncated" != "true" ]; then
            break
        fi
        
        # Get cursor for next page (from result_info)
        cursor=$(echo "$response" | jq -r '.result_info.cursor // empty')
        if [ -z "$cursor" ] || [ "$cursor" = "null" ]; then
            break
        fi
    done
    
    echo -e "  ${GREEN}✓${NC} Deleted ${total_deleted} objects from ${bucket}"
    echo ""
}

# Function to clear KV namespace
clear_kv_namespace() {
    local namespace_id=$1
    echo "Clearing KV namespace: ${namespace_id}"
    
    # List all keys
    echo "  Listing keys..."
    local keys_output=$(cd apps/api-worker && wrangler kv key list --namespace-id="${namespace_id}" --json 2>/dev/null || echo "[]")
    
    if [ "$keys_output" = "[]" ] || [ -z "$keys_output" ]; then
        echo -e "  ${GREEN}✓${NC} No keys found (already empty)"
        echo ""
        return 0
    fi
    
    # Extract key names into array
    local key_count=$(echo "$keys_output" | jq 'length' 2>/dev/null || echo "0")
    
    if [ "$key_count" -eq 0 ]; then
        echo -e "  ${GREEN}✓${NC} No keys found (already empty)"
        echo ""
        return 0
    fi
    
    echo "  Found ${key_count} keys"
    
    # Delete each key
    local deleted=0
    local i=0
    while [ $i -lt "$key_count" ]; do
        local key=$(echo "$keys_output" | jq -r ".[$i].name // empty" 2>/dev/null)
        if [ -n "$key" ] && [ "$key" != "null" ]; then
            echo "    Deleting: ${key}"
            if cd apps/api-worker && wrangler kv key delete "${key}" --namespace-id="${namespace_id}" > /dev/null 2>&1; then
                deleted=$((deleted + 1))
            else
                echo -e "      ${RED}✘${NC} Failed to delete ${key}"
            fi
        fi
        i=$((i + 1))
    done
    
    echo -e "  ${GREEN}✓${NC} Deleted ${deleted} keys from KV namespace"
    echo ""
}

# Confirm before proceeding
echo -e "${YELLOW}⚠${NC}  This will delete ALL data from:"
echo "  - R2 bucket: ${R2_BUCKET}"
echo "  - KV namespace: ${KV_NAMESPACE_ID}"
echo ""
read -p "Are you sure you want to continue? (type 'yes' to confirm): " confirmation

if [ "$confirmation" != "yes" ]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Starting cleanup..."
echo ""

# Clear R2 bucket
clear_r2_bucket "${R2_BUCKET}"

# Clear KV namespace
clear_kv_namespace "${KV_NAMESPACE_ID}"

echo -e "${GREEN}✓${NC} All data cleared successfully!"
echo ""
echo "Summary:"
echo "  - R2 bucket '${R2_BUCKET}': Cleared"
echo "  - KV namespace '${KV_NAMESPACE_ID}': Cleared"

