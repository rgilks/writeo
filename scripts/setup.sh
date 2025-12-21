#!/bin/bash
# Setup script for writeo deployment
# This script helps set up Cloudflare resources and guides through deployment

set -e

echo "=== Writeo Setup Script ==="
echo ""

# Check for wrangler
if ! command -v wrangler &> /dev/null; then
    echo "Wrangler not found. Installing..."
    cd apps/api-worker
    npm install wrangler --save-dev
    cd ../..
    echo "Wrangler installed locally in api-worker"
    echo "Use: cd apps/api-worker && npx wrangler <command>"
    WRANGLER_CMD="cd apps/api-worker && npx wrangler"
else
    echo "✓ Wrangler found"
    WRANGLER_CMD="wrangler"
fi

echo ""
echo "=== Step 1: Cloudflare Authentication ==="
echo "Make sure you're logged in to Cloudflare:"
echo "  $WRANGLER_CMD login"
echo ""
read -p "Press Enter when you're logged in..."

echo ""
echo "=== Step 2: Create R2 Bucket ==="
echo "Creating R2 bucket: writeo-data"
$WRANGLER_CMD r2 bucket create writeo-data || echo "Bucket may already exist"

echo ""
echo "=== Step 3: Create KV Namespace ==="
echo "Creating KV namespace: WRITEO_RESULTS"
KV_OUTPUT=$($WRANGLER_CMD kv:namespace create "WRITEO_RESULTS")
echo "$KV_OUTPUT"
KV_ID=$(echo "$KV_OUTPUT" | grep -oP 'id = "\K[^"]+')

echo ""
echo "Creating preview KV namespace"
KV_PREVIEW_OUTPUT=$($WRANGLER_CMD kv:namespace create "WRITEO_RESULTS" --preview)
echo "$KV_PREVIEW_OUTPUT"
KV_PREVIEW_ID=$(echo "$KV_PREVIEW_OUTPUT" | grep -oP 'id = "\K[^"]+')

if [ -n "$KV_ID" ] && [ -n "$KV_PREVIEW_ID" ]; then
    echo ""
    echo "Updating wrangler.toml files with KV IDs..."
    sed -i.bak "s/id = \"your-kv-namespace-id\"/id = \"$KV_ID\"/g" apps/api-worker/wrangler.toml
    sed -i.bak "s/preview_id = \"your-kv-preview-id\"/preview_id = \"$KV_PREVIEW_ID\"/g" apps/api-worker/wrangler.toml
    rm -f apps/*/wrangler.toml.bak
    echo "✓ KV IDs updated"
else
    echo "⚠ Could not extract KV IDs automatically. Please update manually:"
    echo "  KV ID: $KV_ID"
    echo "  Preview ID: $KV_PREVIEW_ID"
fi

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "1. Deploy Modal service: cd services/modal-deberta && modal deploy app.py"
echo "2. Set Modal URL secret: cd apps/api-worker && $WRANGLER_CMD secret put MODAL_DEBERTA_URL"
echo "3. Deploy API Worker: cd apps/api-worker && $WRANGLER_CMD deploy"
echo "4. Deploy Frontend: cd apps/web && npm run deploy"
echo "5. Test: ./scripts/smoke.sh"

