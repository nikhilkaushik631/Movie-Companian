FROM python:3.13-slim

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
    echo 'echo "Starting FastAPI server..."' >> start.sh && \
    echo 'python API.py &' >> start.sh && \
    echo 'echo "Waiting for FastAPI to start..."' >> start.sh && \
    echo 'sleep 10' >> start.sh && \
    echo 'until curl -f http://localhost:8000/health 2>/dev/null || curl -f http://localhost:8000 2>/dev/null; do' >> start.sh && \
    echo '    echo "Waiting for FastAPI to be ready..."' >> start.sh && \
    echo '    sleep 5' >> start.sh && \
    echo 'done' >> start.sh && \
    echo 'echo "FastAPI is ready! Starting Gradio app..."' >> start.sh && \
    echo 'exec python Gradio_App.py' >> start.sh

RUN chmod +x start.sh

# Expose ports
EXPOSE 8000 7860

# Run the startup script
CMD ["/bin/sh", "start.sh"]