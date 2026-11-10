FROM python:3.12-slim

# ─────────────────────────────────────────────
# System dependencies
# ─────────────────────────────────────────────

RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*


# ─────────────────────────────────────────────
# Non-root user
# ─────────────────────────────────────────────

RUN groupadd -r appgroup \
    && useradd -r -g appgroup -u 1000 appuser


# ─────────────────────────────────────────────
# Application root and Python runtime
# ─────────────────────────────────────────────

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app


# ─────────────────────────────────────────────
# Install Prompt Cache Service dependencies
# ─────────────────────────────────────────────

COPY requirements/prompt_cache_service.txt /tmp/prompt_cache_service.txt

RUN pip install --no-cache-dir --default-timeout=120 --retries=10 \
    -r /tmp/prompt_cache_service.txt


# ─────────────────────────────────────────────
# Install the shared package
# ─────────────────────────────────────────────

COPY app/packages/shared /tmp/packages/shared

RUN pip install --no-cache-dir --default-timeout=120 --retries=10 \
    /tmp/packages/shared


# ─────────────────────────────────────────────
# Copy Prompt Cache Service source
#
# Keep the /app/services/... source under /app/app/services/...
# because the service's Celery import path is app.services....
# PYTHONPATH=/app makes the outer /app directory the import root.
# ─────────────────────────────────────────────

COPY app/services/prompt_cache_service /app/app/services/prompt_cache_service


# ─────────────────────────────────────────────
# Copy the Celery Beat schedule adapter
# ─────────────────────────────────────────────

COPY docker/prompt_cache_beat.py /app/prompt_cache_beat.py


# ─────────────────────────────────────────────
# Runtime directories and permissions
# ─────────────────────────────────────────────

RUN mkdir -p /app/credentials \
    && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000


# ─────────────────────────────────────────────
# Default process: Prompt Cache Service API
# Compose overrides this command for outbox, Celery worker,
# and recovery scheduler containers.
# ─────────────────────────────────────────────

CMD ["uvicorn", "app.services.prompt_cache_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
