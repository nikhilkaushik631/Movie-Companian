#!/bin/bash
# Startup script for Railway/Render deployment
# Ensures proper port binding and worker configuration for free tier

set -e

# Use PORT env variable from hosting platform, default to 8000
PORT=${PORT:-8000}

echo "Starting Cinemizer Backend on port $PORT..."

# Run with 1 worker for free tier resource limits
exec uvicorn main:app \
  --host 0.0.0.0 \
  --port $PORT \
  --workers 1 \
  --access-log \
  --log-level info
