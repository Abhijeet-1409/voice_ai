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
# Install shared package
# ─────────────────────────────────────────────

COPY app/dependencies/ /app/dependencies/

RUN pip install --no-cache-dir /app/dependencies


# ─────────────────────────────────────────────
# Install worker-specific dependencies
# ─────────────────────────────────────────────

COPY requirements/worker.txt /tmp/worker.txt

RUN pip install --no-cache-dir -r /tmp/worker.txt


# ─────────────────────────────────────────────
# Copy complete worker application
# ─────────────────────────────────────────────

COPY app/services/worker/ /app/worker/


# ─────────────────────────────────────────────
# Runtime directories
# ─────────────────────────────────────────────

RUN mkdir -p /app/data /app/rag_models


# ─────────────────────────────────────────────
# Python runtime configuration
# ─────────────────────────────────────────────

WORKDIR /app/worker

ENV PYTHONPATH=/app/worker


# ─────────────────────────────────────────────
# Permissions
# ─────────────────────────────────────────────

RUN chown -R appuser:appgroup /app

USER appuser


# ─────────────────────────────────────────────
# Worker entrypoint
# ─────────────────────────────────────────────

CMD ["python", "agent_runner.py"]