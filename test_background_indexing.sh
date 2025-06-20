#!/bin/bash
set -e

echo "Testing background indexing system..."

# Start the server in the background
echo "Starting server..."
cargo run -p breeze-server -- serve &
SERVER_PID=$!

# Give the server time to start
sleep 3

# Submit an indexing task
echo -e "\nSubmitting indexing task for current directory..."
RESPONSE=$(curl -s -X POST http://localhost:3000/api/v1/index/project \
  -H "Content-Type: application/json" \
  -d "{\"path\": \"$(pwd)\"}")

echo "Response: $RESPONSE"

# Extract task ID from response
TASK_ID=$(echo $RESPONSE | jq -r '.task_id')
echo "Task ID: $TASK_ID"

# Check task status a few times
for i in {1..10}; do
  echo -e "\nChecking task status (attempt $i)..."
  STATUS=$(curl -s http://localhost:3000/api/v1/tasks/$TASK_ID)
  echo "Status: $STATUS"
  
  # Check if task is completed
  if echo "$STATUS" | jq -e '.status == "completed"' > /dev/null; then
    echo -e "\nTask completed successfully!"
    echo "Files indexed: $(echo $STATUS | jq -r '.files_indexed')"
    break
  elif echo "$STATUS" | jq -e '.status == "failed"' > /dev/null; then
    echo -e "\nTask failed!"
    echo "Error: $(echo $STATUS | jq -r '.error')"
    break
  fi
  
  sleep 2
done

# List recent tasks
echo -e "\nListing recent tasks..."
curl -s http://localhost:3000/api/v1/tasks?limit=5 | jq '.'

# Kill the server
echo -e "\nStopping server..."
kill $SERVER_PID 2>/dev/null || true

echo -e "\nTest completed!"