FROM python:3.12-slim

# ─────────────────────────────────────────────
# System dependencies
# ─────────────────────────────────────────────

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


# ─────────────────────────────────────────────
# Non-root user
# ─────────────────────────────────────────────

RUN groupadd -r appgroup \
    && useradd -r -g appgroup -u 1000 appuser


# ─────────────────────────────────────────────
# Application root
# ─────────────────────────────────────────────

WORKDIR /app


# ─────────────────────────────────────────────
# HuggingFace / SentenceTransformer cache
# ─────────────────────────────────────────────

ENV HF_HOME=/app/rag_models
ENV SENTENCE_TRANSFORMERS_HOME=/app/rag_models


# ─────────────────────────────────────────────
# Install ingestion dependencies
# ─────────────────────────────────────────────

COPY requirements/ingest.txt /tmp/ingest.txt

RUN pip install --no-cache-dir -r /tmp/ingest.txt


# ─────────────────────────────────────────────
# Copy shared code
# ─────────────────────────────────────────────

COPY app/dependencies/shared /app/shared/


# ─────────────────────────────────────────────
# Copy only worker code required by ingestion
# ─────────────────────────────────────────────

COPY app/services/worker/config /app/worker/config
COPY app/services/worker/rag /app/worker/rag


# ─────────────────────────────────────────────
# Runtime directories
# ─────────────────────────────────────────────

RUN mkdir -p /app/data /app/rag_models


# ─────────────────────────────────────────────
# Python import path
# ─────────────────────────────────────────────

ENV PYTHONPATH=/app:/app/worker

WORKDIR /app/worker


# ─────────────────────────────────────────────
# Permissions
# ─────────────────────────────────────────────

RUN chown -R appuser:appgroup /app

USER appuser


# ─────────────────────────────────────────────
# Ingestion entrypoint
# ─────────────────────────────────────────────

CMD ["python", "-m", "rag.ingest"]