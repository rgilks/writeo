#!/bin/bash
set -e

# Function to check if a port is in use
check_port() {
  lsof -i :$1 >/dev/null 2>&1
}

API_PID=""
WEB_PID=""

# Start API Worker if not running
if check_port 8787; then
  echo "API Worker already running on port 8787"
else
  echo "Starting API Worker..."
  npm run dev --workspace=@writeo/api-worker -- --port 8787 --var API_KEY:test-key-for-mocked-services --var TEST_API_KEY:test-key-for-mocked-services --var USE_MOCK_SERVICES:true &
  API_PID=$!
fi

# Start Web App if not running
if check_port 3000; then
  echo "Web App already running on port 3000"
else
  echo "Starting Web App..."
  npm run dev --workspace=@writeo/web -- --port 3000 &
  WEB_PID=$!
fi

# Cleanup on exit (only kill processes we started)
cleanup() {
  if [ -n "$API_PID" ]; then kill $API_PID 2>/dev/null || true; fi
  if [ -n "$WEB_PID" ]; then kill $WEB_PID 2>/dev/null || true; fi
}
trap cleanup EXIT

# Wait for processes if we started any
if [ -n "$API_PID" ] || [ -n "$WEB_PID" ]; then
  wait $API_PID $WEB_PID
else
  # If both were already running, just hang to keep the script alive (if needed by Playwright)
  # But Playwright usually manages the process lifecycle. If we exit, Playwright might think the server crashed?
  # Actually, if Playwright uses reuseExistingServer: true, it won't even call this script if port 3000 is open.
  # So this script is only called if port 3000 is NOT open.
  # However, if port 8787 IS open but 3000 is NOT, we need to start 3000 and keep running.
  echo "Services are running. Waiting..."
  sleep infinity
fi
