#!/bin/bash
# Deploy Modal service

set -e

echo "=== Deploying Modal Service ==="
echo ""

# Check for uv (preferred) or pip
if command -v uv &> /dev/null; then
    echo "✓ uv found"
    INSTALL_CMD="uv pip install"
elif command -v pip3 &> /dev/null; then
    echo "⚠ Using pip3 (consider installing uv for faster installs: curl -LsSf https://astral.sh/uv/install.sh | sh)"
    INSTALL_CMD="pip3 install"
else
    echo "Error: Neither uv nor pip3 found. Please install Python package manager."
    exit 1
fi

# Check for modal
if ! command -v modal &> /dev/null; then
    echo "Modal CLI not found. Installing..."
    $INSTALL_CMD modal
    echo "✓ Modal installed"
fi

# Check authentication by trying a simple command
echo "Checking Modal authentication..."
if ! modal app list &> /dev/null; then
    echo "Not authenticated. Please run: modal token new"
    exit 1
fi

echo "✓ Authenticated"

cd services/modal-essay

echo ""
echo "Deploying Modal service..."
echo "This may take several minutes on first deployment (model download)..."
echo ""

modal deploy app.py

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Next steps:"
echo ""
echo "1. Set Modal API key secret (use the same API_KEY as your Cloudflare Worker):"
echo "   modal secret create MODAL_API_KEY <your-api-key>"
echo ""
echo "2. Copy the endpoint URL above and set it as a Cloudflare Worker secret:"
echo "   cd apps/api-worker"
echo "   wrangler secret put MODAL_GRADE_URL"
echo "   (paste the endpoint URL when prompted)"
echo ""
echo "Note: The Modal service requires MODAL_API_KEY to be set. Use the same"
echo "      API key value as your Cloudflare Worker API_KEY for consistency."

