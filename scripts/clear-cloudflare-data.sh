#!/bin/bash
# Clear all data from Cloudflare R2 and KV storage
# Uses Cloudflare API bulk delete for fast R2 deletion

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
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

# Function to clear R2 bucket using Cloudflare API bulk delete
clear_r2_bucket() {
    local bucket=$1
    echo "Clearing R2 bucket: ${bucket}"
    echo "  Listing objects..."
    
    local cursor=""
    local total_deleted=0
    local batch_size=1000  # R2 bulk delete supports up to 1000 objects per request
    
    while true; do
        # List objects
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
            echo -e "  ${RED}✘${NC} Error: $error"
            return 1
        fi
        
        # Extract objects
        local objects_json=$(echo "$response" | jq -r '.result // []')
        local object_count=$(echo "$objects_json" | jq 'length')
        
        if [ "$object_count" -eq 0 ]; then
            break
        fi
        
        # Build bulk delete payload (up to 1000 objects per request)
        local i=0
        while [ $i -lt "$object_count" ]; do
            local batch_keys=""
            local batch_count=0
            
            # Collect up to batch_size keys
            while [ $i -lt "$object_count" ] && [ $batch_count -lt $batch_size ]; do
                local key=$(echo "$objects_json" | jq -r ".[$i].key")
                if [ -n "$key" ] && [ "$key" != "null" ]; then
                    if [ -z "$batch_keys" ]; then
                        batch_keys="[\"${key}\""
                    else
                        batch_keys="${batch_keys},\"${key}\""
                    fi
                    batch_count=$((batch_count + 1))
                fi
                i=$((i + 1))
            done
            
            if [ -n "$batch_keys" ]; then
                batch_keys="${batch_keys}]"
                echo "  Deleting batch of ${batch_count} objects..."
                
                # Bulk delete
                local delete_response=$(curl -s -X POST \
                    "https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/r2/buckets/${bucket}/objects/bulk-delete" \
                    -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" \
                    -H "Content-Type: application/json" \
                    -d "{\"keys\":${batch_keys}}")
                
                if echo "$delete_response" | jq -e '.success == false' > /dev/null 2>&1; then
                    local error=$(echo "$delete_response" | jq -r '.errors[0].message // "Unknown error"')
                    echo -e "    ${RED}✘${NC} Error deleting batch: $error"
                else
                    total_deleted=$((total_deleted + batch_count))
                fi
            fi
        done
        
        # Check if there are more pages
        local truncated=$(echo "$response" | jq -r '.result_info.is_truncated // false')
        if [ "$truncated" != "true" ]; then
            break
        fi
        
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

# Clear R2 bucket (fast!)
clear_r2_bucket "${R2_BUCKET}"

# Clear KV namespace
clear_kv_namespace "${KV_NAMESPACE_ID}"

echo -e "${GREEN}✓${NC} All data cleared successfully!"
echo ""
echo "Summary:"
echo "  - R2 bucket '${R2_BUCKET}': Cleared"
echo "  - KV namespace '${KV_NAMESPACE_ID}': Cleared"

