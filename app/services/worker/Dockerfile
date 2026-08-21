FROM python:3.12-slim

# Install system dependencies required for native C/C++ extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and group
RUN groupadd -r appgroup && useradd -r -g appgroup -u 1000 appuser

WORKDIR /app

# Redirect HuggingFace & SentenceTransformers model cache
ENV HF_HOME=/app/rag_models
ENV SENTENCE_TRANSFORMERS_HOME=/app/rag_models

# 1. Copy and install local shared package
COPY dependencies/ /app/dependencies/
RUN pip install --no-cache-dir /app/dependencies

# 2. Copy and install worker service requirements
COPY services/worker/requirements.txt /app/worker/requirements.txt
RUN pip install --no-cache-dir -r /app/worker/requirements.txt

# 3. Copy worker source code into /app/worker/
COPY services/worker/ /app/worker/

# Create volume mount target directories
RUN mkdir -p /app/rag_models /app/data

# 4. Set execution directory, PYTHONPATH, permissions, and drop root privileges
WORKDIR /app/worker
ENV PYTHONPATH=/app/worker

RUN chown -R appuser:appgroup /app
USER appuser

ENV AGENT_MODE=start

CMD ["sh", "-c", "python agent_runner.py $AGENT_MODE"]