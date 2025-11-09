FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Update package lists and install dependencies
RUN apt-get update && \
    apt-get install -y gcc g++ curl build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN echo '#!/bin/bash' > start.sh && \
    echo 'set -e' >> start.sh && \
    echo 'echo "Starting FastAPI server..."' >> start.sh && \
    echo 'uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info &' >> start.sh && \
    echo 'echo "Waiting for FastAPI to start..."' >> start.sh && \
    echo 'sleep 5' >> start.sh && \
    echo 'until curl -f http://localhost:8000/health 2>/dev/null || curl -f http://localhost:8000 2>/dev/null; do' >> start.sh && \
    echo '    echo "Waiting for FastAPI to be ready..."' >> start.sh && \
    echo '    sleep 3' >> start.sh && \
    echo 'done' >> start.sh && \
    echo 'echo "FastAPI is ready!"' >> start.sh && \
    echo 'exec tail -f /dev/null' >> start.sh

RUN chmod +x start.sh

# Expose ports
EXPOSE 8000

# Run the startup script
CMD ["/bin/sh", "start.sh"]