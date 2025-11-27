#!/bin/bash
# Install git hooks for Writeo
# This script copies hooks from scripts/hooks/ to .git/hooks/

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the root directory of the git repository
ROOT_DIR=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
HOOKS_SOURCE="$ROOT_DIR/scripts/hooks"
HOOKS_TARGET="$ROOT_DIR/.git/hooks"

if [ ! -d "$HOOKS_SOURCE" ]; then
    echo "${YELLOW}Warning: hooks directory not found at $HOOKS_SOURCE${NC}"
    exit 1
fi

if [ ! -d "$HOOKS_TARGET" ]; then
    echo "${YELLOW}Error: .git/hooks directory not found. Are you in a git repository?${NC}"
    exit 1
fi

echo "Installing git hooks..."

# Copy each hook file and make it executable
for hook in pre-commit pre-push; do
    if [ -f "$HOOKS_SOURCE/$hook" ]; then
        cp "$HOOKS_SOURCE/$hook" "$HOOKS_TARGET/$hook"
        chmod +x "$HOOKS_TARGET/$hook"
        echo "${GREEN}✓ Installed $hook hook${NC}"
    else
        echo "${YELLOW}⚠ Hook $hook not found in $HOOKS_SOURCE${NC}"
    fi
done

echo ""
echo "${GREEN}Git hooks installed successfully!${NC}"
echo ""
echo "The following hooks are now active:"
echo "  - pre-commit: Formats code, runs linting and type checking"
echo "  - pre-push: Runs all tests against local servers"
echo ""
echo "Pre-push optimizations:"
echo "  - Automatically skips E2E tests for docs-only changes"
echo "  - Quick mode: QUICK_PUSH=true git push (skips E2E tests)"
echo ""
echo "To bypass hooks (if needed):"
echo "  git commit --no-verify  # Skip pre-commit hook"
echo "  git push --no-verify    # Skip pre-push hook"

