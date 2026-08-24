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
# Install ingestion-specific dependencies
# ─────────────────────────────────────────────

COPY requirements/ingest.txt /tmp/ingest.txt

RUN pip install --no-cache-dir -r /tmp/ingest.txt


# ─────────────────────────────────────────────
# Copy only shared code required by ingestion
# ─────────────────────────────────────────────

COPY app/dependencies/shared/__init__.py /app/shared/__init__.py
COPY app/dependencies/shared/call_context.py /app/shared/call_context.py

COPY app/dependencies/shared/config/ \
     /app/shared/config/

COPY app/dependencies/shared/logging_setup/ \
     /app/shared/logging_setup/

COPY app/dependencies/shared/infra/__init__.py \
     /app/shared/infra/__init__.py

COPY app/dependencies/shared/infra/postgres/ \
     /app/shared/infra/postgres/

COPY app/dependencies/shared/infra/vector_store/ \
     /app/shared/infra/vector_store/


# ─────────────────────────────────────────────
# Copy worker config + RAG code required by ingestion
# ─────────────────────────────────────────────

COPY app/services/worker/config/ \
     /app/worker/config/

COPY app/services/worker/rag/ \
     /app/worker/rag/


# ─────────────────────────────────────────────
# Copy root environment file
# ─────────────────────────────────────────────

COPY .env /app/worker/.env


# ─────────────────────────────────────────────
# Runtime directories
# ─────────────────────────────────────────────

RUN mkdir -p /app/data /app/rag_models


# ─────────────────────────────────────────────
# Python configuration
# ─────────────────────────────────────────────

ENV PYTHONPATH=/app/worker:/app

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