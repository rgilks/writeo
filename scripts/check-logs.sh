#!/bin/bash
# Safe log checking script - won't get stuck
# Usage: ./scripts/check-logs.sh <api-worker|web> [search-term] [limit]

set -e

WORKER=$1
SEARCH_TERM=${2:-""}
LIMIT=${3:-20}

if [ -z "$WORKER" ]; then
    echo "Usage: $0 <api-worker|web> [search-term] [limit]"
    echo ""
    echo "Examples:"
    echo "  $0 api-worker \"error\" 20"
    echo "  $0 web \"\" 30"
    echo "  $0 api-worker \"Modal\" 20"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}/apps/${WORKER}"

if [ -n "$SEARCH_TERM" ]; then
    echo "Checking logs for '${WORKER}' (search: '${SEARCH_TERM}', limit: ${LIMIT})..."
    wrangler tail --format json --search "$SEARCH_TERM" 2>&1 | head -$LIMIT
else
    echo "Checking recent logs for '${WORKER}' (limit: ${LIMIT})..."
    wrangler tail --format json 2>&1 | head -$LIMIT
fi

echo ""
echo "✓ Log check completed"

