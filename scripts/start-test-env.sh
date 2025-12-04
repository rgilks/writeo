#!/bin/bash
set -e

# Start API Worker
echo "Starting API Worker..."
npm run dev --workspace=@writeo/api-worker -- --port 8787 --var API_KEY:test-key-for-mocked-services --var TEST_API_KEY:test-key-for-mocked-services --var USE_MOCK_SERVICES:true &
API_PID=$!

# Start Web App
echo "Starting Web App..."
npm run dev --workspace=@writeo/web -- --port 3000 &
WEB_PID=$!

# Cleanup on exit
trap "kill $API_PID $WEB_PID" EXIT

# Wait for processes
wait $API_PID $WEB_PID
