#!/bin/bash
# Master deployment script for Writeo
# Deploys all services in the correct order

set -e

echo "=== Writeo Full Deployment ==="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check for wrangler (global or local)
if command -v wrangler &> /dev/null; then
    WRANGLER_CMD="wrangler"
elif command -v npx &> /dev/null; then
    # Try to use npx wrangler from one of the workspaces
    WRANGLER_CMD="npx wrangler"
    echo -e "${YELLOW}Using npx wrangler (local installation)${NC}"
else
    echo -e "${RED}Error: wrangler not found. Please install it first.${NC}"
    echo "  npm install -g wrangler"
    exit 1
fi

# Check for modal (will be checked in deploy-modal.sh)
if ! command -v modal &> /dev/null; then
    echo -e "${YELLOW}Warning: modal CLI not found. It will be installed during Modal deployment.${NC}"
fi

echo "=== Step 1: Deploy Modal Service ==="
echo ""
# Deploy Modal and capture output to extract URL
MODAL_OUTPUT=$(./scripts/deploy-modal.sh 2>&1)
echo "$MODAL_OUTPUT"

# Extract Modal URL from output (look for https://...--writeo-essay-fastapi-app.modal.run or similar)
# Use sed for macOS compatibility (grep -P not available on macOS)
MODAL_URL=$(echo "$MODAL_OUTPUT" | grep -o 'https://[^[:space:]]*--writeo-.*-fastapi-app\.modal\.run' | head -1)

if [ -z "$MODAL_URL" ]; then
    echo ""
    echo -e "${YELLOW}Could not automatically extract Modal URL. Please enter it manually.${NC}"
    read -p "Enter the Modal endpoint URL (e.g., https://xxx--writeo-essay-fastapi-app.modal.run): " MODAL_URL
    if [ -z "$MODAL_URL" ]; then
        echo -e "${RED}Error: Modal URL is required${NC}"
        exit 1
    fi
else
    echo ""
    echo -e "${GREEN}✓ Extracted Modal URL: $MODAL_URL${NC}"
fi

echo ""
echo "=== Step 2: Configure Secrets ==="
echo "Setting MODAL_GRADE_URL secret for API worker..."
cd apps/api-worker
echo "$MODAL_URL" | $WRANGLER_CMD secret put MODAL_GRADE_URL
cd ../..

echo ""
echo "Setting API_KEY secrets..."
echo -e "${YELLOW}Note: API_KEY must be set for both Cloudflare Worker and Modal services.${NC}"
read -p "Do you want to set API_KEY now? (y/N): " SET_API_KEY
if [[ $SET_API_KEY =~ ^[Yy]$ ]]; then
    read -sp "Enter API_KEY (input will be hidden): " API_KEY_VALUE
    echo ""
    
    # Set Cloudflare Worker API_KEY
    cd apps/api-worker
    echo "$API_KEY_VALUE" | $WRANGLER_CMD secret put API_KEY
    cd ../..
    echo -e "${GREEN}✓ Cloudflare Worker API_KEY configured${NC}"
    
    # Set Modal API_KEY
    echo "$API_KEY_VALUE" | modal secret create MODAL_API_KEY
    echo -e "${GREEN}✓ Modal API_KEY configured${NC}"
else
    echo -e "${YELLOW}⚠ Skipping API_KEY setup. Make sure to set it manually:${NC}"
    echo "  cd apps/api-worker && wrangler secret put API_KEY"
    echo "  modal secret create MODAL_API_KEY <same-value-as-api-key>"
fi

echo -e "${GREEN}✓ Secrets configured${NC}"

echo ""
echo "=== Step 3: Build Shared Package ==="
echo "Building shared TypeScript package..."
npm run build --workspace=@writeo/shared
echo -e "${GREEN}✓ Shared package built${NC}"

echo ""
echo "=== Step 4: Deploy API Worker ==="
echo "Deploying API worker..."
cd apps/api-worker
$WRANGLER_CMD deploy
cd ../..
echo -e "${GREEN}✓ API worker deployed${NC}"

echo ""
echo "=== Step 5: Deploy Frontend ==="
read -p "Deploy frontend? (Y/n): " DEPLOY_FRONTEND
if [[ ! $DEPLOY_FRONTEND =~ ^[Nn]$ ]]; then
    echo "Deploying frontend (this may take a few minutes)..."
    cd apps/web
    npm run deploy
    cd ../..
    echo -e "${GREEN}✓ Frontend deployed${NC}"
else
    echo "Skipping frontend deployment"
fi

echo ""
echo "=== Deployment Complete! ==="
echo ""
echo -e "${GREEN}All services have been deployed successfully.${NC}"
echo ""
echo "Next steps:"
echo "1. Test the deployment: ./scripts/smoke.sh"
echo "2. Check worker logs: cd apps/api-worker && wrangler tail"
echo "3. View in Cloudflare dashboard: https://dash.cloudflare.com"
echo ""

read -p "Run smoke tests now? (Y/n): " RUN_SMOKE
if [[ ! $RUN_SMOKE =~ ^[Nn]$ ]]; then
    echo ""
    echo "=== Running Smoke Tests ==="
    ./scripts/smoke.sh
fi

