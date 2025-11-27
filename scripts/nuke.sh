#!/bin/bash
# Nuclear cleanup script - removes all build artifacts, caches, and dependencies
# Use with caution! This will delete everything.

set -e

echo "🧹 Nuclear cleanup - removing all build artifacts and dependencies..."

# Helper function to safely remove directories
remove_dirs() {
  for pattern in "$@"; do
    find . -name "$pattern" -type d -prune -exec rm -rf '{}' + 2>/dev/null || true
  done
}

# Helper function to safely remove files
remove_files() {
  for pattern in "$@"; do
    find . -name "$pattern" -type f -delete 2>/dev/null || true
  done
}

# Node.js artifacts
remove_dirs 'node_modules' 'dist' 'build'
remove_files 'package-lock.json' '*.tsbuildinfo' '.pnp.js'
find . -name '.pnp' -type f -delete 2>/dev/null || true

# Next.js/Web artifacts
remove_dirs '.next' '.open-next' '.opennext' 'out'

# Cloudflare Workers artifacts
remove_dirs '.wrangler'
find . -type d -path '*/.wrangler-*' -prune -exec rm -rf '{}' + 2>/dev/null || true

# Python artifacts
remove_dirs '__pycache__' '*.egg-info' '.pytest_cache' '.mypy_cache' '.Python' \
  'htmlcov' '.tox' '.venv' 'venv' 'ENV' 'env'
remove_files '*.pyc' '*.pyo' '*.pyd' '*.so' '*.egg' '.coverage' '.coverage.*'
find . -name '*$py.class' -type f -delete 2>/dev/null || true

# Test artifacts
remove_dirs 'test-results' 'playwright-report' 'playwright/.cache'

# Cache directories
remove_dirs '.cache'

# Modal artifacts
remove_dirs '.modal'

echo "✅ Cleanup complete!"
