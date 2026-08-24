FROM python:3.12-slim

# Install system dependencies required for native C/C++ extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and group
RUN groupadd -r appgroup && useradd -r -g appgroup -u 1000 appuser

WORKDIR /app

# HuggingFace & SentenceTransformers model cache
ENV HF_HOME=/app/rag_models
ENV SENTENCE_TRANSFORMERS_HOME=/app/rag_models

# 1. Install local shared package
COPY dependencies/ /app/dependencies/
RUN pip install --no-cache-dir /app/dependencies

# 2. Install worker dependencies
COPY services/worker/requirements.txt /app/worker/requirements.txt
RUN pip install --no-cache-dir -r /app/worker/requirements.txt

# 3. Copy worker source code
COPY services/worker/ /app/worker/

# Create directories used by mounted volumes
RUN mkdir -p /app/rag_models /app/data

# 4. Runtime configuration
WORKDIR /app/worker
ENV PYTHONPATH=/app/worker

# Drop root privileges
RUN chown -R appuser:appgroup /app
USER appuser

# Production container entrypoint
CMD ["python", "agent_runner.py"]