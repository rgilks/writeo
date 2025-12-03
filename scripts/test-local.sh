#!/bin/bash
set -e

# Start API Worker
echo "Starting API Worker..."
cd apps/api-worker
npm run dev &
API_PID=$!
cd ../..

# Start Web App
echo "Starting Web App..."
cd apps/web
npm run dev &
WEB_PID=$!
cd ../..

# Wait for services to be ready
echo "Waiting for services to be ready..."
# Simple sleep for now, ideally we should poll the health endpoints
sleep 10

# Run Tests
echo "Running Tests..."
npm test

# Cleanup
echo "Cleaning up..."
kill $API_PID
kill $WEB_PID
