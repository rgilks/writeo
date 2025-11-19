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
KV_NAMESPACE_NAME="WRITEO_RESULTS"
WRANGLER_TOML="apps/api-worker/wrangler.toml"

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

# Function to clear R2 bucket by deleting and recreating it
clear_r2_bucket() {
    local bucket=$1
    echo "Clearing R2 bucket: ${bucket}"
    echo "  Deleting bucket (this will remove all objects)..."
    
    # Delete the bucket (this deletes all objects automatically)
    local delete_response=$(curl -s -X DELETE \
        "https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/r2/buckets/${bucket}" \
        -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" \
        -H "Content-Type: application/json")
    
    # Check if deletion was successful or bucket doesn't exist
    if echo "$delete_response" | jq -e '.success == false' > /dev/null 2>&1; then
        local error=$(echo "$delete_response" | jq -r '.errors[0].message // "Unknown error"')
        local error_code=$(echo "$delete_response" | jq -r '.errors[0].code // ""')
        
        # If bucket doesn't exist, that's fine - we'll create it
        if [[ "$error" == *"not found"* ]] || [[ "$error_code" == "10009" ]]; then
            echo "  Bucket doesn't exist, will create it..."
        else
            echo -e "  ${YELLOW}⚠${NC}  Warning: ${error}"
            echo "  Attempting to continue..."
        fi
    else
        echo -e "  ${GREEN}✓${NC} Bucket deleted"
    fi
    
    # Wait a moment for deletion to propagate
    sleep 2
    
    # Recreate the bucket
    echo "  Recreating bucket..."
    local create_response=$(curl -s -X POST \
        "https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/r2/buckets" \
        -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{\"name\":\"${bucket}\"}")
    
    if echo "$create_response" | jq -e '.success == false' > /dev/null 2>&1; then
        local error=$(echo "$create_response" | jq -r '.errors[0].message // "Unknown error"')
        local error_code=$(echo "$create_response" | jq -r '.errors[0].code // ""')
        
        # If bucket already exists, that's fine
        if [[ "$error" == *"already exists"* ]] || [[ "$error_code" == "10013" ]]; then
            echo -e "  ${GREEN}✓${NC} Bucket already exists"
        else
            echo -e "  ${RED}✘${NC} Error creating bucket: $error"
            return 1
        fi
    else
        echo -e "  ${GREEN}✓${NC} Bucket recreated"
    fi
    
    echo ""
}

# Function to clear KV namespace by deleting and recreating it
clear_kv_namespace() {
    local namespace_id=$1
    local namespace_name=$2
    echo "Clearing KV namespace: ${namespace_id}"
    
    # Get namespace title (name) from API if not provided
    if [ -z "$namespace_name" ]; then
        echo "  Getting namespace info..."
        local ns_info=$(curl -s -X GET \
            "https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/storage/kv/namespaces/${namespace_id}" \
            -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" \
            -H "Content-Type: application/json")
        
        if echo "$ns_info" | jq -e '.success == true' > /dev/null 2>&1; then
            namespace_name=$(echo "$ns_info" | jq -r '.result.title // "WRITEO_RESULTS"')
        else
            namespace_name="WRITEO_RESULTS"
        fi
    fi
    
    echo "  Deleting namespace '${namespace_name}'..."
    
    # Delete the namespace
    local delete_response=$(curl -s -X DELETE \
        "https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/storage/kv/namespaces/${namespace_id}" \
        -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" \
        -H "Content-Type: application/json")
    
    # Check if deletion was successful or namespace doesn't exist
    if echo "$delete_response" | jq -e '.success == false' > /dev/null 2>&1; then
        local error=$(echo "$delete_response" | jq -r '.errors[0].message // "Unknown error"')
        local error_code=$(echo "$delete_response" | jq -r '.errors[0].code // ""')
        
        # If namespace doesn't exist, that's fine - we'll create it
        if [[ "$error" == *"not found"* ]] || [[ "$error_code" == "10009" ]]; then
            echo "  Namespace doesn't exist, will create it..."
        else
            echo -e "  ${YELLOW}⚠${NC}  Warning: ${error}"
            echo "  Attempting to continue..."
        fi
    else
        echo -e "  ${GREEN}✓${NC} Namespace deleted"
    fi
    
    # Wait a moment for deletion to propagate
    sleep 2
    
    # Recreate the namespace
    echo "  Recreating namespace '${namespace_name}'..."
    local create_response=$(curl -s -X POST \
        "https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/storage/kv/namespaces" \
        -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{\"title\":\"${namespace_name}\"}")
    
    if echo "$create_response" | jq -e '.success == false' > /dev/null 2>&1; then
        local error=$(echo "$create_response" | jq -r '.errors[0].message // "Unknown error"')
        echo -e "  ${RED}✘${NC} Error creating namespace: $error"
        return 1
    fi
    
    local new_namespace_id=$(echo "$create_response" | jq -r '.result.id // empty')
    
    if [ -z "$new_namespace_id" ] || [ "$new_namespace_id" = "null" ]; then
        echo -e "  ${RED}✘${NC} Failed to get new namespace ID"
        return 1
    fi
    
    echo -e "  ${GREEN}✓${NC} Namespace recreated with ID: ${new_namespace_id}"
    
    # Update wrangler.toml with new namespace ID
    if [ -f "$WRANGLER_TOML" ]; then
        echo "  Updating ${WRANGLER_TOML} with new namespace ID..."
        # Use sed to replace the namespace ID (be careful with the exact format)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS sed
            sed -i '' "s/id = \"${namespace_id}\"/id = \"${new_namespace_id}\"/g" "$WRANGLER_TOML"
        else
            # Linux sed
            sed -i "s/id = \"${namespace_id}\"/id = \"${new_namespace_id}\"/g" "$WRANGLER_TOML"
        fi
        echo -e "  ${GREEN}✓${NC} Updated wrangler.toml"
    else
        echo -e "  ${YELLOW}⚠${NC}  wrangler.toml not found, please update manually:"
        echo "    id = \"${new_namespace_id}\""
    fi
    
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
clear_kv_namespace "${KV_NAMESPACE_ID}" "${KV_NAMESPACE_NAME}"

echo -e "${GREEN}✓${NC} All data cleared successfully!"
echo ""
echo "Summary:"
echo "  - R2 bucket '${R2_BUCKET}': Cleared (recreated)"
echo "  - KV namespace '${KV_NAMESPACE_NAME}': Cleared (recreated)"
echo ""
echo "Note: If the KV namespace ID changed, ${WRANGLER_TOML} has been updated automatically."
echo "      You may need to redeploy your worker for the changes to take effect."

